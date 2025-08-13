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
import numpy as np                          # Agregasi antar-crop

# --- 1) Setup umum ---
MODEL_URL  = 'https://drive.google.com/uc?id=1GPPxPSpodNGJHSeWyrTVwRoSjwW3HaA8'  # Link model di Drive
MODEL_PATH = 'model_3.safetensors'                                               # Nama file lokal model

# Urutan label harus sama dengan saat training
LABELS = [
    'alpukat_matang', 'alpukat_mentah',
    'belimbing_matang', 'belimbing_mentah',
    'mangga_matang', 'mangga_mentah'
]

# ---- Parameter model + inferensi (mengikuti retrain-5) ----
NUM_HEADS   = 10               # Jumlah attention heads
NUM_LAYERS  = 4                # Banyak InteractionBlock
HIDDEN_DIM  = 640              # Dimensi embedding
PATCH_SIZE  = 14               # Ukuran patch conv
IMAGE_SIZE  = 210              # Ukuran input ke model
THRESHOLDS  = [0.01, 0.01, 0.01, 0.06, 0.02, 0.01]  # Threshold per-kelas (dari validasi)

# Sliding-window (disarankan pembimbing: dilakukan di tahap prediksi/aplikasi)
WIN_FRACS   = (1.0, 0.7, 0.5, 0.4)  # Tambah skala kecil supaya tiap buah lebih terisolasi
STRIDE_FRAC = 0.33                  # Overlap ~67% biar gak lompat objek
MIN_VOTES   = 2                     # Minimal jumlah crop yang "setuju" agar label dihitung

# --- 2) Download model bila belum ada / korup ---
def download_model():
    with st.spinner('🔄 Mengunduh ulang model dari Google Drive...'):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 50000:
    if os.path.exists(MODEL_PATH):
        st.warning("📦 Ukuran file model terlalu kecil, kemungkinan korup. Mengunduh ulang...")
        os.remove(MODEL_PATH)
    download_model()

# --- 3) Komponen model (identik dengan saat training) ---
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, emb_size):
        super().__init__()
        self.proj = nn.Conv2d(3, emb_size, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)                 # [B, E, H/ps, W/ps]
        x = x.flatten(2).transpose(1, 2) # [B, N, E]
        return x

class WordEmbedding(nn.Module):
    def __init__(self, dim): super().__init__()
    def forward(self, x): return x       # dummy (kita pakai tensor nol di inferensi)

class FeatureFusion(nn.Module):
    def forward(self, v, t):             # v,t: [B, N, E]
        return torch.cat([v, t[:, :v.size(1), :]], dim=-1)  # → [B, N, 2E]

class ScaleTransformation(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__(); self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.linear(x)             # 2E → E

class ChannelUnification(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.norm = nn.LayerNorm(dim)
    def forward(self, x): return self.norm(x)

class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=NUM_HEADS):
        super().__init__(); self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
    def forward(self, x): return self.attn(x, x, x)[0]

class CrossScaleAggregation(nn.Module):
    def forward(self, x): return x.mean(dim=1, keepdim=True)  # [B, 1, E]

class HamburgerHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__(); self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.linear(x)

class MLPClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, num_classes))
    def forward(self, x): return self.mlp(x)

class HSVLTModel(nn.Module):
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
        image_feat = self.patch_embed(image)     # [B, N, E]
        text_feat  = self.word_embed(text)       # [B, N, E]
        x = self.concat(image_feat, text_feat)   # [B, N, 2E]
        x = self.scale_transform(x)              # [B, N, E]
        x = self.channel_unification(x)          # [B, N, E]
        x = self.interaction_blocks(x)           # [B, N, E]
        x = self.csa(x)                          # [B, 1, E]
        x = self.head(x)                         # [B, 1, E]
        x = x.mean(dim=1)                        # [B, E]
        return self.classifier(x)                # [B, C] (logits)

# --- 4) Load model ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if len(THRESHOLDS) != len(LABELS):
    st.error("Panjang THRESHOLDS tidak sama dengan jumlah LABELS."); st.stop()

try:
    with safe_open(MODEL_PATH, framework="pt", device=device) as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    model = HSVLTModel(
        patch_size=PATCH_SIZE, emb_size=HIDDEN_DIM, num_classes=len(LABELS),
        num_heads=NUM_HEADS, num_layers=NUM_LAYERS
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.code(traceback.format_exc()); st.stop()

# --- 5) Transformasi gambar ---
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

# --- 5.1) Sliding-window inference + agregasi ---
def sliding_window_infer(image_pil, model, transform, device,
                         hidden_dim, patch_size,
                         win_fracs=WIN_FRACS, stride_frac=STRIDE_FRAC,
                         include_full=False):
    """Potong gambar (multi-skala, overlap), prediksi per-crop, kembalikan:
       - probs_max:  max antar-crop per kelas [C]
       - probs_crops: semua probabilitas per crop [N, C]
    """
    W, H = image_pil.size
    short = min(W, H)
    crops = []

    for wf in win_fracs:
        win  = max(int(short * wf), 64)            # ukuran jendela
        step = max(1, int(win * stride_frac))      # stride (overlap)
        for top in range(0, max(H - win + 1, 1), step):
            for left in range(0, max(W - win + 1, 1), step):
                crops.append(image_pil.crop((left, top, left + win, top + win)))

    if include_full:
        crops.append(image_pil)                    # opsional: tambahkan full image

    num_tokens = (IMAGE_SIZE // patch_size) ** 2
    probs_list = []
    model.eval()
    with torch.no_grad():
        for crop in crops:
            x = transform(crop).unsqueeze(0).to(device)
            dummy_text = torch.zeros((1, num_tokens, hidden_dim), device=device)  # konsisten dgn training
            p = torch.sigmoid(model(x, dummy_text)).cpu().numpy()[0]              # [C]
            probs_list.append(p)

    probs_crops = np.stack(probs_list, axis=0)     # [N, C]
    probs_max   = probs_crops.max(axis=0)          # agregasi MAX antar-crop
    return probs_max, probs_crops

# --- 6) UI ---
st.title("🍉 Klasifikasi Multi-Label Buah")
st.write("Upload gambar buah; sistem akan mendeteksi beberapa label sekaligus.")

uploaded_file = st.file_uploader("Unggah gambar buah", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar Input", use_container_width=True)

    # 1) Sliding-window (multi-skala, overlap besar)
    probs_max, probs_crops = sliding_window_infer(
        image_pil=image, model=model, transform=transform, device=device,
        hidden_dim=HIDDEN_DIM, patch_size=PATCH_SIZE,
        win_fracs=WIN_FRACS, stride_frac=STRIDE_FRAC, include_full=False
    )
    probs = probs_max.tolist()

    # 2) Votes: berapa crop yang ≥ threshold per-kelas
    votes = (probs_crops >= np.array(THRESHOLDS)).sum(axis=0).astype(int)  # [C]

    # 3) Kandidat = skor ≥ threshold & votes cukup
    candidates = {
        lbl: (float(p), int(v))
        for lbl, p, thr, v in zip(LABELS, probs, THRESHOLDS, votes.tolist())
        if (p >= thr and v >= MIN_VOTES)
    }

    # 4) Aturan eksklusif matang vs mentah (per buah pilih satu)
    pairs = [
        ('alpukat_matang', 'alpukat_mentah'),
        ('belimbing_matang', 'belimbing_mentah'),
        ('mangga_matang', 'mangga_mentah')
    ]
    final_labels = []
    for a, b in pairs:
        has_a, has_b = a in candidates, b in candidates
        if has_a and has_b:
            chosen = a if candidates[a][0] >= candidates[b][0] else b
            final_labels.append((chosen, *candidates[chosen]))  # (label, prob, votes)
        elif has_a:
            final_labels.append((a, *candidates[a]))
        elif has_b:
            final_labels.append((b, *candidates[b]))

    final_labels.sort(key=lambda x: x[1], reverse=True)

    # 5) Tampilkan
    st.subheader("🔍 Label Terdeteksi:")
    if not final_labels:
        st.warning("🚫 Tidak ada label yang memenuhi kriteria (threshold + votes).")
    else:
        for label, prob, v in final_labels:
            st.write(f"✅ *{label}* ({prob:.2%}) — votes: {v}")

    # Panel detail
    with st.expander("📊 Lihat Semua Probabilitas"):
        mean_prob = float(np.mean(probs))
        entropy   = -float(np.mean([p * math.log(p + 1e-8) for p in probs]))
        st.write(f"🪟 crops: {probs_crops.shape[0]} | mean_prob: {mean_prob:.3f} | entropy: {entropy:.3f}")
        for i, lbl in enumerate(LABELS):
            pass_thr = "✓" if probs[i] >= THRESHOLDS[i] else "✗"
            st.write(f"{lbl}: {probs[i]:.2%} (thr {THRESHOLDS[i]:.2f}) | votes={int(votes[i])} {pass_thr}")
