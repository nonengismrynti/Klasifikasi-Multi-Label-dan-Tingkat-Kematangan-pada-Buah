import streamlit as st
import torch
import torch.nn as nn
from safetensors.torch import safe_open
import torchvision.transforms as transforms
from PIL import Image
import gdown
import os
import math
import numpy as np
import traceback  # <-- tambah

# --- 1. Setup ---
MODEL_URL = 'https://drive.google.com/uc?id=1GPPxPSpodNGJHSeWyrTVwRoSjwW3HaA8'
MODEL_PATH = 'model_3.safetensors'

LABELS = [
    'alpukat_matang', 'alpukat_mentah',
    'belimbing_matang', 'belimbing_mentah',
    'mangga_matang', 'mangga_mentah'
]

# ==== PARAM ====
NUM_HEADS  = 10
NUM_LAYERS = 4
HIDDEN_DIM = 640
PATCH_SIZE = 14
IMAGE_SIZE = 210
# pakai threshold per-kelas (dari tuning test retrain-5)
THRESHOLDS = [0.01, 0.01, 0.01, 0.06, 0.02, 0.01]

# --- 2. Download model ---
def download_model():
    with st.spinner('🔄 Mengunduh ulang model dari Google Drive...'):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 50000:
    if os.path.exists(MODEL_PATH):
        st.warning("📦 Ukuran file model terlalu kecil, kemungkinan korup. Mengunduh ulang...")
        os.remove(MODEL_PATH)
    download_model()

# --- 3. Komponen Model ---
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, emb_size):
        super().__init__()
        self.proj = nn.Conv2d(3, emb_size, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)                          # [B, C, H/ps, W/ps]
        x = x.flatten(2).transpose(1, 2)          # [B, num_patches, emb_size]
        return x

class WordEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
    def forward(self, x):
        return x                                   # x = dummy_text [B, 225, 640]

class FeatureFusion(nn.Module):
    def forward(self, v, t):                      # v = [B, N, E], t = [B, N, E]
        return torch.cat([v, t[:, :v.size(1), :]], dim=-1)

class ScaleTransformation(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.linear(x)

class ChannelUnification(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        return self.norm(x)

class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=NUM_HEADS):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
    def forward(self, x):
        return self.attn(x, x, x)[0]

class CrossScaleAggregation(nn.Module):
    def forward(self, x):
        return x.mean(dim=1, keepdim=True)  # [B, 1, D]

class HamburgerHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.linear(x)

class MLPClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.mlp(x)

# ==== MODEL ====
class HSVLTModel(nn.Module):
    def __init__(self, img_size=210, patch_size=14, emb_size=HIDDEN_DIM,
                 num_classes=6, num_heads=NUM_HEADS, num_layers=NUM_LAYERS):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, emb_size)
        self.word_embed = WordEmbedding(emb_size)
        self.concat = FeatureFusion()
        self.scale_transform = ScaleTransformation(emb_size * 2, emb_size)
        self.channel_unification = ChannelUnification(emb_size)
        self.interaction_blocks = nn.Sequential(
            *[InteractionBlock(emb_size, num_heads=num_heads) for _ in range(num_layers)]
        )
        self.csa = CrossScaleAggregation()
        self.head = HamburgerHead(emb_size, emb_size)
        self.classifier = MLPClassifier(emb_size, num_classes)

    def forward(self, image, text):
        image_feat = self.patch_embed(image)         # [B, N, E]
        text_feat  = self.word_embed(text)           # [B, N, E]
        x = self.concat(image_feat, text_feat)
        x = self.scale_transform(x)
        x = self.channel_unification(x)
        x = self.interaction_blocks(x)
        x = self.csa(x)
        x = self.head(x)
        x = x.mean(dim=1)                            # [B, E]
        return self.classifier(x)                    # [B, num_classes]

# --- 4. Load Model ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Safety: cek konsistensi panjang THRESHOLDS
if len(THRESHOLDS) != len(LABELS):
    st.error("Panjang THRESHOLDS tidak sama dengan jumlah LABELS.")
    st.stop()

try:
    with safe_open(MODEL_PATH, framework="pt", device=device) as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}

    model = HSVLTModel(
        patch_size=PATCH_SIZE,
        emb_size=HIDDEN_DIM,
        num_classes=len(LABELS),
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.code(traceback.format_exc())
    st.stop()

# --- 5. Transformasi Gambar ---
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==========================================================
# 6. Sliding Window (cropping per-bagian saat prediksi)
# ==========================================================
def _gen_starts(total: int, size: int, stride: int):
    """Buat titik awal agar ujung kanan/bawah selalu ter-cover."""
    if total <= size:
        return [0]
    starts = list(range(0, total - size + 1, stride))
    if starts[-1] != total - size:
        starts.append(total - size)
    return starts

def sliding_window_crops(pil_img: Image.Image, window_size: int, overlap: float):
    """
    Return:
      crops: list[PIL.Image]
      rects: list[(x0,y0,x1,y1)]
    """
    w, h = pil_img.size
    size = min(window_size, w, h)  # jangan lebih besar dari sisi terkecil
    stride = max(1, int(size * (1 - overlap)))

    xs = _gen_starts(w, size, stride)
    ys = _gen_starts(h, size, stride)

    crops, rects = [], []
    for y in ys:
        for x in xs:
            rect = (x, y, x + size, y + size)
            crops.append(pil_img.crop(rect))
            rects.append(rect)
    return crops, rects, size, stride, (len(xs), len(ys))

def infer_with_optional_sliding(pil_img: Image.Image,
                                use_sliding: bool,
                                window_size: int,
                                overlap: float,
                                agg: str = "max"):
    """
    Menghasilkan vektor probabilitas per-kelas.
    Jika use_sliding=True, akan agregasi dari semua jendela (max/mean).
    """
    num_tokens = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 225

    if not use_sliding:
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        dummy_text = torch.zeros((1, num_tokens, HIDDEN_DIM), device=device)
        with torch.no_grad():
            logits = model(input_tensor, dummy_text)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        detail = {"num_windows": 1, "grid": (1, 1)}
        return probs, detail

    # Sliding window path
    crops, rects, size, stride, grid = sliding_window_crops(pil_img, window_size, overlap)

    # batching biar cepat
    tensors = [transform(c).unsqueeze(0) for c in crops]
    batch = torch.cat(tensors, dim=0).to(device)

    with torch.no_grad():
        dummy_text = torch.zeros((batch.size(0), num_tokens, HIDDEN_DIM), device=device)
        logits = model(batch, dummy_text)
        probs_all = torch.sigmoid(logits).cpu().numpy()  # [N, C]

    if agg == "mean":
        probs = probs_all.mean(axis=0)
    else:  # default max
        probs = probs_all.max(axis=0)

    detail = {
        "num_windows": probs_all.shape[0],
        "grid": grid,
        "stride": stride,
        "win_size": size,
    }
    return probs, detail

# --- 7. Streamlit UI ---
st.title("🍉 Klasifikasi Multi-Label Buah")
st.write("Upload gambar buah, sistem akan mendeteksi beberapa label sekaligus. Jika bukan buah, akan ditolak.")
st.caption("🔧 Sliding window diterapkan *saat pengujian/prediksi* untuk memotong gambar jadi beberapa bagian agar pembacaan objek terpisah.")

uploaded_file = st.file_uploader("Unggah gambar buah", type=['jpg', 'jpeg', 'png'])

# Panel pengaturan sliding
use_sliding = st.checkbox("Aktifkan sliding window (crop per-bagian)", value=True)
agg = st.selectbox("Metode agregasi skor antar-jendela", ["max", "mean"], index=0,
                   help="max = ambil skor tertinggi per-kelas; mean = rata-rata.")
win_size_px = None
overlap = None

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar Input", use_container_width=True)

    if use_sliding:
        w, h = image.size
        default_ws = min(max(210, min(w, h)//2), min(w, h))  # default aman
        win_size_px = st.slider("Ukuran jendela (px)", 128, min(1024, min(w, h)), default_ws, step=16)
        overlap = st.slider("Overlap antar-jendela", 0.0, 0.9, 0.30, step=0.05)

    # === infer (single atau sliding) ===
    probs, detail = infer_with_optional_sliding(
        image,
        use_sliding=use_sliding,
        window_size=win_size_px if use_sliding else IMAGE_SIZE,
        overlap=overlap if use_sliding else 0.0,
        agg=agg
    )
    probs = probs.tolist()

    # --- pakai threshold per-kelas ---
    detected_labels = [
        (label, prob) for label, prob, thr in zip(LABELS, probs, THRESHOLDS) if prob >= thr
    ]
    detected_labels.sort(key=lambda x: x[1], reverse=True)

    # Statistik tambahan
    max_prob = float(np.max(probs))
    second_max_prob = float(np.partition(probs, -2)[-2]) if len(probs) > 1 else 0.0
    mean_prob = float(np.mean(probs))

    # OOD sederhana: jumlah label >= threshold
    high_conf_count = sum(int(p >= t) for p, t in zip(probs, THRESHOLDS))
    is_ood = (high_conf_count < 1)

    st.subheader("🔍 Label Terdeteksi:")
    if is_ood:
        st.warning("🚫 Gambar tidak mengandung buah yang dikenali.")
    else:
        if detected_labels:
            for label, prob in detected_labels:
                st.write(f"✅ *{label}* ({prob:.2%})")
        else:
            st.warning("🚫 Tidak ada label yang melewati ambang batas.")

    with st.expander("📊 Lihat Semua Probabilitas"):
        entropy = -sum([p * math.log(p + 1e-8) for p in probs]) / len(probs)
        st.write(f"📊 mean_prob: {mean_prob:.3f} | entropy: {entropy:.3f}")
        if use_sliding:
            st.write(f"🪟 windows: {detail['num_windows']} | grid: {detail['grid']} | "
                     f"win_size: {detail.get('win_size','-')} | stride: {detail.get('stride','-')}")
            st.caption("Catatan: skor kelas adalah agregasi dari semua jendela.")
        for label, prob, thr in zip(LABELS, probs, THRESHOLDS):
            pass_thr = "✓" if prob >= thr else "✗"
            st.write(f"{label}: {prob:.2%} (thr {thr:.2f}) {pass_thr}")
