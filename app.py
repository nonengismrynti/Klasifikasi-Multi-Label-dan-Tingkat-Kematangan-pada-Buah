import streamlit as st                      # UI web
import torch                                # PyTorch core
import torch.nn as nn                       # Modul neural network
from safetensors.torch import safe_open     # Loader bobot .safetensors
import torchvision.transforms as transforms # Transformasi gambar
from PIL import Image                       # Baca gambar
import gdown                                # Download dari Google Drive
import os                                   # Utilitas file/path
import math                                 # Untuk hitung entropi (informasi)
import traceback                            # Tampilkan traceback saat error
import numpy as np                          # Agregasi antar-crop (sliding window)

# ========================
# 1) Setup umum & opsi UI
# ========================
MODEL_URL  = 'https://drive.google.com/uc?id=1GPPxPSpodNGJHSeWyrTVwRoSjwW3HaA8'  # Link model di Drive
MODEL_PATH = 'model_3.safetensors'                                               # Nama file lokal model

LABELS = [                                   # Urutan label HARUS sama seperti saat training
    'alpukat_matang', 'alpukat_mentah',
    'belimbing_matang', 'belimbing_mentah',
    'mangga_matang', 'mangga_mentah'
]

SHOW_VOTES = False                           # ← Tampilkan "votes" di UI? (False = disembunyikan)

# ======================================================
# 2) Parameter model + inferensi (mengikuti retrain-5)
# ======================================================
NUM_HEADS   = 10                             # Jumlah attention heads
NUM_LAYERS  = 4                              # Banyak InteractionBlock
HIDDEN_DIM  = 640                            # Dimensi embedding
PATCH_SIZE  = 14                             # Ukuran patch conv
IMAGE_SIZE  = 210                            # Ukuran input ke model
THRESHOLDS  = [0.01, 0.01, 0.01, 0.06, 0.02, 0.01]  # Threshold per-kelas (hasil tuning validasi)

# Sliding-window (dipasang di tahap prediksi/aplikasi sesuai arahan dosen)
WIN_FRACS   = (1.0, 0.7, 0.5, 0.4)           # Skala jendela relatif terhadap sisi terpendek
STRIDE_FRAC = 0.33                           # Overlap ~67% (stride = 0.33 * window)
MIN_VOTES   = 2                              # Minimal jumlah crop yang “setuju” agar label dihitung

# ==========================================
# 3) Download model bila belum ada/terdeteksi korup
# ==========================================
def download_model():                        # Fungsi unduh model dari Google Drive
    with st.spinner('🔄 Mengunduh ulang model dari Google Drive...'):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 50000:  # Cek eksistensi/ukuran minimal
    if os.path.exists(MODEL_PATH):              # Jika file ada tapi kecil → kemungkinan korup
        st.warning("📦 Ukuran file model terlalu kecil, kemungkinan korup. Mengunduh ulang...")
        os.remove(MODEL_PATH)                   # Hapus file korup
    download_model()                            # Unduh ulang

# ======================================
# 4) Komponen model (identik dgn training)
# ======================================
class PatchEmbedding(nn.Module):                # Ubah gambar → token patch
    def __init__(self, img_size, patch_size, emb_size):
        super().__init__()
        self.proj = nn.Conv2d(3, emb_size, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)                        # [B, E, H/ps, W/ps]
        x = x.flatten(2).transpose(1, 2)        # [B, N, E]
        return x

class WordEmbedding(nn.Module):                 # Dummy (kita pakai tensor nol saat inferensi)
    def __init__(self, dim): super().__init__()
    def forward(self, x): return x

class FeatureFusion(nn.Module):                 # Gabung visual + “teks” (dummy)
    def forward(self, v, t):
        return torch.cat([v, t[:, :v.size(1), :]], dim=-1)  # → [B, N, 2E]

class ScaleTransformation(nn.Module):           # Proyeksi 2E → E
    def __init__(self, in_dim, out_dim):
        super().__init__(); self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.linear(x)

class ChannelUnification(nn.Module):            # LayerNorm token
    def __init__(self, dim):
        super().__init__(); self.norm = nn.LayerNorm(dim)
    def forward(self, x): return self.norm(x)

class InteractionBlock(nn.Module):              # Self-attention block
    def __init__(self, dim, num_heads=NUM_HEADS):
        super().__init__(); self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
    def forward(self, x): return self.attn(x, x, x)[0]

class CrossScaleAggregation(nn.Module):         # Agregasi sederhana antar token
    def forward(self, x): return x.mean(dim=1, keepdim=True)  # [B, 1, E]

class HamburgerHead(nn.Module):                 # Linear head penyesuaian
    def __init__(self, in_dim, out_dim):
        super().__init__(); self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.linear(x)

class MLPClassifier(nn.Module):                 # Klasifier multi-label (logits)
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, num_classes))
    def forward(self, x): return self.mlp(x)

class HSVLTModel(nn.Module):                    # Rangkaian lengkap model
    def __init__(self, img_size=210, patch_size=14, emb_size=HIDDEN_DIM,
                 num_classes=6, num_heads=NUM_HEADS, num_layers=NUM_LAYERS):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, emb_size)
        self.word_embed  = WordEmbedding(emb_size)
        self.concat      = FeatureFusion()
        self.scale_transform     = ScaleTransformation(emb_size * 2, emb_size)
        self.channel_unification = ChannelUnification(emb_size)
        self.interaction_blocks  = nn.Sequential(
            *[InteractionBlock(emb_size, num_heads=num_heads) for _ in range(num_layers)]
        )
        self.csa        = CrossScaleAggregation()
        self.head       = HamburgerHead(emb_size, emb_size)
        self.classifier = MLPClassifier(emb_size, num_classes)

    def forward(self, image, text):
        image_feat = self.patch_embed(image)    # [B, N, E]
        text_feat  = self.word_embed(text)      # [B, N, E] (dummy zeros)
        x = self.concat(image_feat, text_feat)  # [B, N, 2E]
        x = self.scale_transform(x)             # [B, N, E]
        x = self.channel_unification(x)         # [B, N, E]
        x = self.interaction_blocks(x)          # [B, N, E]
        x = self.csa(x)                         # [B, 1, E]
        x = self.head(x)                        # [B, 1, E]
        x = x.mean(dim=1)                       # [B, E]
        return self.classifier(x)               # [B, C] logits

# ==================
# 5) Load model
# ==================
device = 'cuda' if torch.cuda.is_available() else 'cpu'   # Pilih device (GPU jika ada)
if len(THRESHOLDS) != len(LABELS):                         # Validasi panjang threshold
    st.error("Panjang THRESHOLDS tidak sama dengan jumlah LABELS."); st.stop()

try:
    with safe_open(MODEL_PATH, framework="pt", device=device) as f:  # Buka file .safetensors
        state_dict = {k: f.get_tensor(k) for k in f.keys()}          # Ambil semua tensor
    model = HSVLTModel(                                              # Inisialisasi arsitektur
        patch_size=PATCH_SIZE, emb_size=HIDDEN_DIM, num_classes=len(LABELS),
        num_heads=NUM_HEADS, num_layers=NUM_LAYERS
    ).to(device)
    model.load_state_dict(state_dict, strict=True)                   # Load bobot
    model.eval()                                                     # Mode evaluasi
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")                           # Pesan gagal
    st.code(traceback.format_exc()); st.stop()                       # Tampilkan traceback & stop

# ===========================
# 6) Transformasi gambar
# ===========================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),                    # Resize ke 210x210
    transforms.ToTensor(),                                          # Ke tensor [0..1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],                # Normalisasi (ImageNet)
                         std =[0.229, 0.224, 0.225])
])

# ==========================================================
# 7) Sliding-window inference + agregasi (MAX antar-crop)
# ==========================================================
def sliding_window_infer(image_pil, model, transform, device,
                         hidden_dim, patch_size,
                         win_fracs=WIN_FRACS, stride_frac=STRIDE_FRAC,
                         include_full=False):
    """
    Potong gambar menjadi beberapa crop overlap (multi-skala),
    prediksi per-crop, lalu agregasi probabilitas per kelas
    dengan operasi MAX antar-crop.
    """
    W, H = image_pil.size                                          # Lebar & tinggi asli
    short = min(W, H)                                              # Sisi terpendek (basis window)
    crops = []                                                     # Kumpulan crop

    for wf in win_fracs:                                           # Loop tiap skala window
        win  = max(int(short * wf), 64)                            # Ukuran window (min 64 px)
        step = max(1, int(win * stride_frac))                      # Stride (overlap tinggi → step kecil)
        for top in range(0, max(H - win + 1, 1), step):            # Iterasi vertikal
            for left in range(0, max(W - win + 1, 1), step):       # Iterasi horizontal
                crops.append(image_pil.crop((left, top, left + win, top + win)))  # Ambil crop persegi

    if include_full:                                               # Opsional: tambahkan full image
        crops.append(image_pil)

    num_tokens = (IMAGE_SIZE // patch_size) ** 2                   # 225 token (15x15)
    probs_list = []                                                # Daftar probabilitas per-crop
    model.eval()                                                   # Pastikan eval
    with torch.no_grad():                                          # Non-grad untuk inferensi
        for crop in crops:
            x = transform(crop).unsqueeze(0).to(device)            # Transform + batch=1
            dummy_text = torch.zeros((1, num_tokens, hidden_dim),  # Dummy text zeros (konsisten training)
                                      device=device)
            logits = model(x, dummy_text)                          # Forward → logits [1, C]
            p = torch.sigmoid(logits).cpu().numpy()[0]             # Sigmoid → probabilitas [C]
            probs_list.append(p)                                   # Simpan hasil

    probs_crops = np.stack(probs_list, axis=0)                     # [N, C] semua crop
    probs_max   = probs_crops.max(axis=0)                          # Agregasi MAX antar-crop → [C]
    return probs_max, probs_crops                                   # Kembalikan vektor & matriks

# ==========
# 8) UI App
# ==========
st.title("🍉 Klasifikasi Multi-Label Buah")                         # Judul
st.write("Upload gambar buah; sistem akan mendeteksi beberapa label sekaligus.")  # Deskripsi

uploaded_file = st.file_uploader("Unggah gambar buah", type=['jpg', 'jpeg', 'png'])  # Uploader

if uploaded_file is not None:                                       # Jika ada file diunggah
    image = Image.open(uploaded_file).convert('RGB')                # Baca sebagai RGB
    st.image(image, caption="Gambar Input", use_container_width=True)  # Tampilkan

    # (1) Prediksi dengan Sliding-Window
    probs_max, probs_crops = sliding_window_infer(
        image_pil=image, model=model, transform=transform, device=device,
        hidden_dim=HIDDEN_DIM, patch_size=PATCH_SIZE,
        win_fracs=WIN_FRACS, stride_frac=STRIDE_FRAC, include_full=False
    )
    probs = probs_max.tolist()                                      # Ke list Python

    # (2) Hitung votes: jumlah crop yang melewati threshold per-kelas
    votes = (probs_crops >= np.array(THRESHOLDS)).sum(axis=0).astype(int)  # [C]

    # (3) Ambil kandidat: skor ≥ threshold dan votes ≥ MIN_VOTES
    #     (TANPA aturan eksklusif matang vs mentah)
    detections = [
        (lbl, float(p), int(v))                     # tuple (nama_label, prob, votes)
        for lbl, p, thr, v in zip(LABELS, probs, THRESHOLDS, votes.tolist())
        if (p >= thr and v >= MIN_VOTES)            # syarat lulus
    ]

    # Urutkan dari probabilitas tertinggi ke terendah
    detections.sort(key=lambda x: x[1], reverse=True)

    # (4) Tampilkan hasil — votes disembunyikan jika SHOW_VOTES=False
    st.subheader("🔍 Label Terdeteksi:")
    if not detections:
        st.warning("🚫 Tidak ada label yang memenuhi kriteria (threshold + votes).")
    else:
        st.write(f"Total label terdeteksi: **{len(detections)}**")
        for label, prob, v in detections:
            if SHOW_VOTES:
                st.write(f"✅ *{label}* ({prob:.2%}) — votes: {v}")
            else:
                st.write(f"✅ *{label}* ({prob:.2%})")

    # (5) Panel detail (opsional: votes hanya jika SHOW_VOTES=True)
    with st.expander("📊 Lihat Semua Probabilitas"):
        mean_prob = float(np.mean(probs))
        entropy   = -float(np.mean([p * math.log(p + 1e-8) for p in probs]))
        st.write(f"🪟 crops: {probs_crops.shape[0]} | mean_prob: {mean_prob:.3f} | entropy: {entropy:.3f}")
        for i, lbl in enumerate(LABELS):
            pass_thr = "✓" if probs[i] >= THRESHOLDS[i] else "✗"
            line = f"{lbl}: {probs[i]:.2%} (thr {THRESHOLDS[i]:.2f}) {pass_thr}"
            if SHOW_VOTES:
                line += f" | votes={int(votes[i])}"
            st.write(line)

    # (6) Panel detail (opsional tampilkan votes sesuai flag)
    with st.expander("📊 Lihat Semua Probabilitas"):
        mean_prob = float(np.mean(probs))                            # Rata-rata probabilitas
        entropy   = -float(np.mean([p * math.log(p + 1e-8) for p in probs]))  # Entropi sederhana
        st.write(f"🪟 crops: {probs_crops.shape[0]} | mean_prob: {mean_prob:.3f} | entropy: {entropy:.3f}")
        for i, lbl in enumerate(LABELS):
            pass_thr = "✓" if probs[i] >= THRESHOLDS[i] else "✗"    # Lolos threshold?
            line = f"{lbl}: {probs[i]:.2%} (thr {THRESHOLDS[i]:.2f}) {pass_thr}"
            if SHOW_VOTES:
                line += f" | votes={int(votes[i])}"                  # Tambah votes jika ingin
            st.write(line)
