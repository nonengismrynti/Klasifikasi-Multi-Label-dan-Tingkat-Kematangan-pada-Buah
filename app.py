import streamlit as st                      # UI web
import torch                                # PyTorch core
import torch.nn as nn                       # Modul neural network
from safetensors.torch import safe_open     # Loader bobot .safetensors
import torchvision.transforms as transforms # Transformasi gambar
from PIL import Image                       # Baca gambar
import gdown                                # Download dari Google Drive
import os                                   # Utilitas file/path
import math                                 # Untuk hitung entropi
import traceback                            # Tampilkan traceback saat error
import numpy as np                          # ← DIPAKAI untuk agregasi antar-crop

# --- 1) Setup umum ---
MODEL_URL  = 'https://drive.google.com/uc?id=1GPPxPSpodNGJHSeWyrTVwRoSjwW3HaA8'  # Link model di Drive
MODEL_PATH = 'model_3.safetensors'                                               # Nama file lokal model

# Daftar label (urutan penting: harus cocok dengan urutan output model)
LABELS = [
    'alpukat_matang', 'alpukat_mentah',
    'belimbing_matang', 'belimbing_mentah',
    'mangga_matang', 'mangga_mentah'
]

# ---- Parameter model + inferensi (mengikuti retrain-5) ----
NUM_HEADS   = 10                # Jumlah attention heads (sesuai retrain-5)
NUM_LAYERS  = 4                 # Banyak InteractionBlock
HIDDEN_DIM  = 640               # Dimensi embedding
PATCH_SIZE  = 14                # Ukuran patch conv
IMAGE_SIZE  = 210               # Ukuran input ke model
THRESHOLDS  = [0.01, 0.01, 0.01, 0.06, 0.02, 0.01]  # Threshold per-kelas hasil tuning validasi

# Parameter sliding-window inferensi (sesuai arahan dosen)
WIN_FRACS    = (1.0, 0.7)       # Ukuran jendela relatif terhadap sisi terpendek (100% dan 70%)
STRIDE_FRAC  = 0.5              # Overlap 50% (stride = 0.5 * window)

# --- 2) Download model bila belum ada / korup ---
def download_model():
    with st.spinner('🔄 Mengunduh ulang model dari Google Drive...'):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)  # Unduh file model

# Cek keberadaan & ukuran file model (antisipasi file korup)
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 50000:
    if os.path.exists(MODEL_PATH):
        st.warning("📦 Ukuran file model terlalu kecil, kemungkinan korup. Mengunduh ulang...")
        os.remove(MODEL_PATH)                 # Hapus file korup
    download_model()                           # Unduh ulang

# --- 3) Komponen model (harus identik dengan saat training) ---
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, emb_size):
        super().__init__()
        self.proj = nn.Conv2d(3, emb_size, kernel_size=patch_size, stride=patch_size)  # Conv untuk ubah gambar→patch
    def forward(self, x):
        x = self.proj(x)                       # [B, E, H/ps, W/ps]
        x = x.flatten(2).transpose(1, 2)       # [B, N, E] (token patch)
        return x

class WordEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
    def forward(self, x):
        return x                               # Dummy: pass-through (pakai tensor nol saat inferensi)

class FeatureFusion(nn.Module):
    def forward(self, v, t):
        return torch.cat([v, t[:, :v.size(1), :]], dim=-1)  # Gabung visual+teks di dim fitur → [B, N, 2E]

class ScaleTransformation(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)            # Proyeksi 2E → E
    def forward(self, x):
        return self.linear(x)

class ChannelUnification(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)                       # LayerNorm token
    def forward(self, x):
        return self.norm(x)

class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=NUM_HEADS):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)  # Self-attention
    def forward(self, x):
        return self.attn(x, x, x)[0]

class CrossScaleAggregation(nn.Module):
    def forward(self, x):
        return x.mean(dim=1, keepdim=True)                  # Agregasi sederhana antar token → [B, 1, E]

class HamburgerHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)            # Linear head penyesuaian
    def forward(self, x):
        return self.linear(x)

class MLPClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),              # Hidden 256 + ReLU
            nn.Linear(256, num_classes)                     # Output logits [B, C]
        )
    def forward(self, x):
        return self.mlp(x)

# Arsitektur HSVLT (mengikuti retrain-5)
class HSVLTModel(nn.Module):
    def __init__(self, img_size=210, patch_size=14, emb_size=HIDDEN_DIM,
                 num_classes=6, num_heads=NUM_HEADS, num_layers=NUM_LAYERS):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, emb_size)             # Patch embedding
        self.word_embed  = WordEmbedding(emb_size)                                    # Word embedding (dummy)
        self.concat      = FeatureFusion()                                            # Concatenate visual+text
        self.scale_transform     = ScaleTransformation(emb_size * 2, emb_size)        # 2E → E
        self.channel_unification = ChannelUnification(emb_size)                       # LayerNorm
        self.interaction_blocks  = nn.Sequential(                                     # Tumpuk self-attn
            *[InteractionBlock(emb_size, num_heads=num_heads) for _ in range(num_layers)]
        )
        self.csa        = CrossScaleAggregation()                                     # CSA sederhana
        self.head       = HamburgerHead(emb_size, emb_size)                           # Linear head
        self.classifier = MLPClassifier(emb_size, num_classes)                        # Klasifier multi-label

    def forward(self, image, text):
        image_feat = self.patch_embed(image)      # [B, N, E]
        text_feat  = self.word_embed(text)        # [B, N, E] (dummy zeros)
        x = self.concat(image_feat, text_feat)    # [B, N, 2E]
        x = self.scale_transform(x)               # [B, N, E]
        x = self.channel_unification(x)           # [B, N, E]
        x = self.interaction_blocks(x)            # [B, N, E]
        x = self.csa(x)                           # [B, 1, E]
        x = self.head(x)                          # [B, 1, E]
        x = x.mean(dim=1)                         # [B, E] (hilangkan dimensi token)
        return self.classifier(x)                 # [B, C] logits

# --- 4) Load model ke device ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Pakai GPU jika ada

# Validasi panjang THRESHOLDS vs LABELS
if len(THRESHOLDS) != len(LABELS):
    st.error("Panjang THRESHOLDS tidak sama dengan jumlah LABELS.")
    st.stop()

try:
    # Baca bobot dari .safetensors (lebih aman dari .pt)
    with safe_open(MODEL_PATH, framework="pt", device=device) as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}  # Ambil semua tensor

    # Inisialisasi arsitektur (harus identik dengan saat training)
    model = HSVLTModel(
        patch_size=PATCH_SIZE,
        emb_size=HIDDEN_DIM,
        num_classes=len(LABELS),
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS
    ).to(device)

    model.load_state_dict(state_dict, strict=True)  # Load bobot ke model
    model.eval()                                    # Set evaluasi (non-training)

except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")         # Tampilkan error ramah
    st.code(traceback.format_exc())                 # Tampilkan traceback teknis
    st.stop()                                       # Hentikan app bila gagal load

# --- 5) Transformasi gambar (harus sama dengan training) ---
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),    # Resize ke 210×210
    transforms.ToTensor(),                          # Ke tensor [0..1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],# Normalisasi (ImageNet mean)
                         std =[0.229, 0.224, 0.225])# dan std
])

# --- 5.1) Fungsi inferensi SLIDING WINDOW  ---
def sliding_window_infer(image_pil, model, transform, device,
                         hidden_dim, patch_size,
                         win_fracs=WIN_FRACS, stride_frac=STRIDE_FRAC):
    """
    Memotong gambar jadi beberapa crop overlap (multi-skala),
    prediksi per crop, lalu agregasi skor dengan MAX antar-crop.
    Mengembalikan:
      - probs_max: vektor probabilitas per label setelah agregasi
      - num_crops: berapa crop yang diproses
    """
    W, H = image_pil.size                     # Ambil lebar & tinggi gambar asli
    short = min(W, H)                         # Sisi terpendek (basis ukuran jendela)
    crops = []                                # List untuk menyimpan potongan gambar

    for wf in win_fracs:                      # Loop tiap skala jendela (mis. 1.0 & 0.7)
        win = max(int(short * wf), 64)        # Ukuran jendela (min 64 px biar gak terlalu kecil)
        step = max(1, int(win * stride_frac)) # Langkah geser = stride (overlap ditentukan oleh stride_frac)

        # Geser jendela secara horizontal & vertikal
        for top in range(0, max(H - win + 1, 1), step):
            for left in range(0, max(W - win + 1, 1), step):
                box = (left, top, left + win, top + win)  # (x1, y1, x2, y2)
                crops.append(image_pil.crop(box))         # Tambah crop persegi

    crops.append(image_pil)                    # Tambahkan full image sebagai crop tambahan

    num_tokens = (IMAGE_SIZE // patch_size) ** 2                   # 225 token (15×15)
    probs_list = []                                                # List untuk simpan prediksi setiap crop
    model.eval()                                                   # Pastikan model eval
    with torch.no_grad():                                          # Non-grad untuk inferensi
        for crop in crops:
            x = transform(crop).unsqueeze(0).to(device)            # Transform + batch size 1
            dummy_text = torch.zeros((1, num_tokens, hidden_dim),  # Dummy text zeros (konsisten training)
                                      device=device)
            logits = model(x, dummy_text)                          # Forward → logits [1, C]
            p = torch.sigmoid(logits).cpu().numpy()[0]             # Sigmoid → probabilitas [C]
            probs_list.append(p)                                   # Simpan

    probs_crops = np.stack(probs_list, axis=0)     # [num_crops, num_classes]
    probs_max   = probs_crops.max(axis=0)          # Agregasi antar-crop: ambil nilai maksimum
    return probs_max, probs_crops.shape[0]         # Kembalikan vektor prob & jumlah crop

# --- 6) UI Streamlit ---
st.title("🍉 Klasifikasi Multi-Label Buah")        # Judul aplikasi
st.write("Upload gambar buah; sistem akan mendeteksi beberapa label sekaligus.")  # Deskripsi singkat

uploaded_file = st.file_uploader("Unggah gambar buah", type=['jpg', 'jpeg', 'png'])  # Uploader

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')      # Baca gambar sebagai RGB
    st.image(image, caption="Gambar Input", use_container_width=True)  # Tampilkan gambar

    # === INFERENSI DENGAN SLIDING WINDOW (arah dosen) ===
    probs_vec, num_crops = sliding_window_infer(
        image_pil=image, model=model, transform=transform, device=device,
        hidden_dim=HIDDEN_DIM, patch_size=PATCH_SIZE,
        win_fracs=WIN_FRACS, stride_frac=STRIDE_FRAC
    )

    probs = probs_vec.tolist()                             # Ubah ke list python agar mudah dipakai

    # --- Terapkan threshold per-kelas untuk menentukan label aktif ---
    detected_labels = [
        (label, prob)                                      # Simpan (nama label, skor)
        for label,
