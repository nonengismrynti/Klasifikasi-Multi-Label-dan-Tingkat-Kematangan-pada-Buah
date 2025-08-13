import streamlit as st
import torch
import torch.nn as nn
from safetensors.torch import safe_open
import torchvision.transforms as transforms
from PIL import Image
import gdown, os, math, traceback
import numpy as np

# --- 1. Setup ---
MODEL_URL  = 'https://drive.google.com/uc?id=1GPPxPSpodNGJHSeWyrTVwRoSjwW3HaA8'
MODEL_PATH = 'model_3.safetensors'

LABELS = [
    'alpukat_matang','alpukat_mentah',
    'belimbing_matang','belimbing_mentah',
    'mangga_matang','mangga_mentah'
]

# ==== PARAM ====
NUM_HEADS=10; NUM_LAYERS=4; HIDDEN_DIM=640; PATCH_SIZE=14; IMAGE_SIZE=210
THRESHOLDS = [0.01, 0.01, 0.01, 0.06, 0.02, 0.01]  # dari validasi

# Param sliding-window (HANYA crop di prediksi)
WIN_FRAC   = 0.70          # ukuran jendela = 70% sisi terpendek (boleh kamu ganti)
STRIDE_FRAC= 0.33          # overlap besar (stride = 0.33 * window)
INCLUDE_FULL = True        # tambahkan full image sebagai crop ekstra

# --- 2. Download model bila belum ada / korup ---
def download_model():
    with st.spinner('🔄 Mengunduh ulang model dari Google Drive...'):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 50000:
    if os.path.exists(MODEL_PATH):
        st.warning("📦 Ukuran file model terlalu kecil, kemungkinan korup. Mengunduh ulang...")
        os.remove(MODEL_PATH)
    download_model()

# --- 3. Komponen Model (identik training) ---
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, emb_size):
        super().__init__(); self.proj = nn.Conv2d(3, emb_size, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)                  # [B,E,H/ps,W/ps]
        return x.flatten(2).transpose(1,2)# [B,N,E]

class WordEmbedding(nn.Module):
    def __init__(self, dim): super().__init__()
    def forward(self, x): return x        # dummy

class FeatureFusion(nn.Module):
    def forward(self, v, t): return torch.cat([v, t[:, :v.size(1), :]], dim=-1)

class ScaleTransformation(nn.Module):
    def __init__(self, in_dim, out_dim): super().__init__(); self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.linear(x)

class ChannelUnification(nn.Module):
    def __init__(self, dim): super().__init__(); self.norm = nn.LayerNorm(dim)
    def forward(self, x): return self.norm(x)

class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=NUM_HEADS):
        super().__init__(); self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
    def forward(self, x): return self.attn(x,x,x)[0]

class CrossScaleAggregation(nn.Module):
    def forward(self, x): return x.mean(dim=1, keepdim=True)

class HamburgerHead(nn.Module):
    def __init__(self, in_dim, out_dim): super().__init__(); self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.linear(x)

class MLPClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__(); self.mlp = nn.Sequential(nn.Linear(in_dim,256), nn.ReLU(), nn.Linear(256,num_classes))
    def forward(self, x): return self.mlp(x)

class HSVLTModel(nn.Module):
    def __init__(self, img_size=210, patch_size=14, emb_size=HIDDEN_DIM, num_classes=6, num_heads=NUM_HEADS, num_layers=NUM_LAYERS):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, emb_size)
        self.word_embed  = WordEmbedding(emb_size)
        self.concat = FeatureFusion()
        self.scale_transform     = ScaleTransformation(emb_size*2, emb_size)
        self.channel_unification = ChannelUnification(emb_size)
        self.interaction_blocks  = nn.Sequential(*[InteractionBlock(emb_size, num_heads=num_heads) for _ in range(num_layers)])
        self.csa = CrossScaleAggregation()
        self.head = HamburgerHead(emb_size, emb_size)
        self.classifier = MLPClassifier(emb_size, num_classes)
    def forward(self, image, text):
        v = self.patch_embed(image); t = self.word_embed(text)
        x = self.concat(v,t); x = self.scale_transform(x); x = self.channel_unification(x)
        x = self.interaction_blocks(x); x = self.csa(x); x = self.head(x)
        return self.classifier(x.mean(dim=1))      # [B,C]

# --- 4. Load Model ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if len(THRESHOLDS) != len(LABELS):
    st.error("Panjang THRESHOLDS tidak sama dengan jumlah LABELS."); st.stop()

try:
    with safe_open(MODEL_PATH, framework="pt", device=device) as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    model = HSVLTModel(patch_size=PATCH_SIZE, emb_size=HIDDEN_DIM, num_classes=len(LABELS),
                       num_heads=NUM_HEADS, num_layers=NUM_LAYERS).to(device)
    model.load_state_dict(state_dict, strict=True); model.eval()
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}"); st.code(traceback.format_exc()); st.stop()

# --- 5. Transformasi gambar ---
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ---------- INI BAGIAN YANG DIMINTA DOSEN: fungsi sliding_window untuk crop ----------
def sliding_window_crops(image_pil, win_frac=WIN_FRAC, stride_frac=STRIDE_FRAC, include_full=INCLUDE_FULL):
    """
    HANYA memotong gambar menjadi potongan persegi (crop) dengan jendela geser.
    - win_frac: ukuran jendela = win_frac * sisi terpendek gambar
    - stride_frac: langkah geser relatif terhadap ukuran jendela (semakin kecil → overlap semakin besar)
    - include_full: jika True, tambahkan 1 crop lagi = full image (opsional)
    Return:
      crops (list[PIL.Image]), boxes (list[tuple(x1,y1,x2,y2)])
    """
    W, H = image_pil.size                          # dimensi aslinya
    short = min(W, H)                              # sisi terpendek buat skala jendela
    win   = max(int(short * win_frac), 64)         # ukuran jendela (min 64px)
    step  = max(1, int(win * stride_frac))         # stride (mis. 0.33*win → overlap ~67%)

    crops, boxes = [], []
    for top in range(0, max(H - win + 1, 1), step):        # geser vertikal
        for left in range(0, max(W - win + 1, 1), step):   # geser horizontal
            box = (left, top, left + win, top + win)       # koordinat crop
            boxes.append(box); crops.append(image_pil.crop(box))

    if include_full:                                       # opsional: tambah full image
        boxes.append((0,0,W,H)); crops.append(image_pil)

    return crops, boxes
# --------------------------------------------------------------------------------------

# --- 6. UI ---
st.title("🍉 Klasifikasi Multi-Label Buah (pakai Sliding Window crop)")
st.write("Fungsi sliding_window: memotong gambar jadi beberapa crop saat prediksi, lalu kita gabungkan hasilnya.")

uploaded_file = st.file_uploader("Unggah gambar buah", type=['jpg','jpeg','png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar Input", use_container_width=True)

    # === INFERENSI: pakai crop dari sliding_window ===
    crops, boxes = sliding_window_crops(image)              # <<— INI inti permintaan
    num_tokens = (IMAGE_SIZE // PATCH_SIZE) ** 2
    probs_list = []                                         # simpan probabilitas per crop

    with torch.no_grad():
        for crop in crops:
            x = transform(crop).unsqueeze(0).to(device)     # [1,3,210,210]
            dummy_text = torch.zeros((1, num_tokens, HIDDEN_DIM), device=device)  # [1,225,640]
            logits = model(x, dummy_text)                   # [1,C]
            p = torch.sigmoid(logits).cpu().numpy()[0]      # [C]
            probs_list.append(p)

    # Agregasi sederhana: MAX antar-crop per kelas (tanpa voting/aturan tambahan)
    probs = np.max(np.stack(probs_list, axis=0), axis=0).tolist()

    # Terapkan threshold per-kelas → label aktif
    detected = [(lbl, prob) for lbl, prob, thr in zip(LABELS, probs, THRESHOLDS) if prob >= thr]
    detected.sort(key=lambda x: x[1], reverse=True)

    st.subheader("🔍 Label Terdeteksi:")
    if not detected:
        st.warning("🚫 Tidak ada label melewati ambang.")
    else:
        for lbl, p in detected:
            st.write(f"✅ *{lbl}* ({p:.2%})")

    # Panel info tambahan
    with st.expander("📊 Detail"):
        st.write(f"🪟 window: {int(WIN_FRAC*min(image.size))} px | stride: {STRIDE_FRAC:.2f}×window | crops: {len(crops)}")
        mean_prob = float(np.mean(probs))
        entropy   = -float(np.mean([p * math.log(p + 1e-8) for p in probs]))
        st.write(f"mean_prob: {mean_prob:.3f} | entropy: {entropy:.3f}")
        for lbl, p, thr in zip(LABELS, probs, THRESHOLDS):
            st.write(f"{lbl}: {p:.2%} (thr {thr:.2f}) {'✓' if p>=thr else '✗'}")
