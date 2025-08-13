import streamlit as st                      # UI web
import torch                                # PyTorch core
import torch.nn as nn                       # Modul neural network
from safetensors.torch import safe_open     # Loader bobot .safetensors
import torchvision.transforms as transforms # Transformasi gambar
from PIL import Image                       # Baca gambar
import gdown                                # Download dari Google Drive
import os                                   # Utilitas file/path
import math                                 # Hitung entropi
import traceback                            # Tampilkan traceback saat error
import numpy as np                          # Operasi numerik/agregasi

# =========================
# 1) Setup & parameter
# =========================
MODEL_URL  = 'https://drive.google.com/uc?id=1GPPxPSpodNGJHSeWyrTVwRoSjwW3HaA8'  # Link model
MODEL_PATH = 'model_3.safetensors'                                               # Nama file lokal

# Urutan label HARUS sama dengan saat training
LABELS = [
    'alpukat_matang', 'alpukat_mentah',
    'belimbing_matang', 'belimbing_mentah',
    'mangga_matang', 'mangga_mentah'
]
IDX = {lbl: i for i, lbl in enumerate(LABELS)}                                   # Map label→index

# Param arsitektur (mengikuti retrain-5)
NUM_HEADS   = 10
NUM_LAYERS  = 4
HIDDEN_DIM  = 640
PATCH_SIZE  = 14
IMAGE_SIZE  = 210

# Threshold dasar hasil validasi (longgar)
BASE_THRESHOLDS = [0.01, 0.01, 0.01, 0.06, 0.02, 0.01]

# Floor threshold untuk deployment (ketatin terutama Mangga)
THRESHOLDS_FLOOR = {
    'alpukat_matang':   0.15,
    'alpukat_mentah':   0.15,
    'belimbing_matang': 0.12,
    'belimbing_mentah': 0.12,
    'mangga_matang':    0.30,
    'mangga_mentah':    0.25
}
THRESHOLDS = [max(b, THRESHOLDS_FLOOR[l]) for b, l in zip(BASE_THRESHOLDS, LABELS)]  # Ambang final

# Sliding-window di tahap prediksi
WIN_FRACS        = (1.0, 0.7, 0.5, 0.4)   # Skala jendela (relatif sisi terpendek)
STRIDE_FRAC      = 0.33                   # Overlap ~67% (lebih rapat → lebih stabil)
INCLUDE_FULL_IMG = True                   # Ikutkan full image sebagai “bukti negatif”
PERCENTILE_Q     = 80                     # Agregasi = percentile 80% (tahan outlier)
MIN_PROP         = 0.15                   # Minimal proporsi crop yang setuju (≥ threshold)
MIN_VOTES_ABS    = 3                      # Minimal votes absolut
EXCLUSIVE_PER_FRUIT = True                # Satu buah: pilih matang ATAU mentah (bukan dua2nya)
CONSISTENCY_MARGIN  = 0.04                # Selisih minimal biar “pemenang” jelas

# =========================
# 2) Download model bila perlu
# =========================
def download_model():
    with st.spinner('🔄 Mengunduh ulang model dari Google Drive...'):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 50000:
    if os.path.exists(MODEL_PATH):
        st.warning("📦 File model terlalu kecil/korup. Mengunduh ulang...")
        os.remove(MODEL_PATH)
    download_model()

# =========================
# 3) Arsitektur (identik training)
# =========================
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, emb_size):
        super().__init__(); self.proj = nn.Conv2d(3, emb_size, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)                 # [B, E, H/ps, W/ps]
        x = x.flatten(2).transpose(1, 2) # [B, N, E]
        return x

class WordEmbedding(nn.Module):
    def __init__(self, dim): super().__init__()   # dummy
    def forward(self, x): return x                # pass-through

class FeatureFusion(nn.Module):
    def forward(self, v, t):                      # v,t: [B, N, E]
        return torch.cat([v, t[:, :v.size(1), :]], dim=-1)  # [B, N, 2E]

class ScaleTransformation(nn.Module):
    def __init__(self, in_dim, out_dim): super().__init__(); self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.linear(x)              # 2E → E

class ChannelUnification(nn.Module):
    def __init__(self, dim): super().__init__(); self.norm = nn.LayerNorm(dim)
    def forward(self, x): return self.norm(x)

class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=NUM_HEADS):
        super().__init__(); self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
    def forward(self, x): return self.attn(x, x, x)[0]

class CrossScaleAggregation(nn.Module):
    def forward(self, x): return x.mean(dim=1, keepdim=True)  # [B, 1, E]

class HamburgerHead(nn.Module):
    def __init__(self, in_dim, out_dim): super().__init__(); self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.linear(x)

class MLPClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__(); self.mlp = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, num_classes))
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
        self.interaction_blocks  = nn.Sequential(*[InteractionBlock(emb_size, num_heads=num_heads) for _ in range(num_layers)])
        self.csa        = CrossScaleAggregation()
        self.head       = HamburgerHead(emb_size, emb_size)
        self.classifier = MLPClassifier(emb_size, num_classes)
    def forward(self, image, text):
        v = self.patch_embed(image)                    # [B, N, E]
        t = self.word_embed(text)                      # [B, N, E]
        x = self.concat(v, t)                          # [B, N, 2E]
        x = self.scale_transform(x); x = self.channel_unification(x)
        x = self.interaction_blocks(x); x = self.csa(x); x = self.head(x)
        x = x.mean(dim=1)                              # [B, E]
        return self.classifier(x)                      # [B, C]

# =========================
# 4) Load model
# =========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if len(THRESHOLDS) != len(LABELS):
    st.error("Panjang THRESHOLDS tidak sama dengan jumlah LABELS."); st.stop()

try:
    with safe_open(MODEL_PATH, framework="pt", device=device) as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    model = HSVLTModel(patch_size=PATCH_SIZE, emb_size=HIDDEN_DIM,
                       num_classes=len(LABELS), num_heads=NUM_HEADS, num_layers=NUM_LAYERS).to(device)
    model.load_state_dict(state_dict, strict=True); model.eval()
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}"); st.code(traceback.format_exc()); st.stop()

# =========================
# 5) Transformasi gambar
# =========================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])
])

# =========================
# 5.1) Sliding-window (return semua skor crop)
# =========================
def get_crop_probs(image_pil, model, transform, device,
                   hidden_dim, patch_size,
                   win_fracs=WIN_FRACS, stride_frac=STRIDE_FRAC, include_full=INCLUDE_FULL_IMG):
    W, H = image_pil.size                                 # Ukuran asli
    short = min(W, H)                                     # Sisi terpendek
    crops = []                                            # Kumpulan crop

    # Buat crop multi-skala dengan overlap
    for wf in win_fracs:
        win  = max(int(short * wf), 64)                   # Ukuran jendela
        step = max(1, int(win * stride_frac))             # Stride (overlap)
        for top in range(0, max(H - win + 1, 1), step):
            for left in range(0, max(W - win + 1, 1), step):
                crops.append(image_pil.crop((left, top, left + win, top + win)))

    if include_full:
        crops.append(image_pil)                           # Tambahkan full image (bukti negatif)

    num_tokens = (IMAGE_SIZE // patch_size) ** 2          # 225 token dummy
    probs_list = []                                       # Tempat semua skor [N, C]

    with torch.no_grad():                                 # Non-grad (cepat)
        for crop in crops:
            x = transform(crop).unsqueeze(0).to(device)   # Transform + batch=1
            dummy_text = torch.zeros((1, num_tokens, hidden_dim), device=device)  # Dummy zeros
            p = torch.sigmoid(model(x, dummy_text)).cpu().numpy()[0]              # Prob [C]
            probs_list.append(p)

    return np.stack(probs_list, axis=0)                   # [N, C]

# =========================
# 5.2) Agregasi robust (percentile)
# =========================
def aggregate_percentile(probs_crops, q=PERCENTILE_Q):
    return np.percentile(probs_crops, q, axis=0)          # Ambil percentile per kelas → [C]

# =========================
# 6) UI
# =========================
st.title("🍉 Klasifikasi Multi-Label Buah")
st.write("Upload gambar; sistem akan mendeteksi beberapa label sekaligus.")

uploaded_file = st.file_uploader("Unggah gambar buah", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')      # Baca RGB
    st.image(image, caption="Gambar Input", use_container_width=True)

    # A) Prediksi semua crop
    probs_crops = get_crop_probs(
        image_pil=image, model=model, transform=transform, device=device,
        hidden_dim=HIDDEN_DIM, patch_size=PATCH_SIZE,
        win_fracs=WIN_FRACS, stride_frac=STRIDE_FRAC, include_full=INCLUDE_FULL_IMG
    )

    # B) Agregasi robust → skor per kelas
    probs_vec = aggregate_percentile(probs_crops, q=PERCENTILE_Q)  # [C]
    probs = probs_vec.tolist()                                     # ke list

    # C) Hitung votes per kelas (berapa crop ≥ threshold)
    votes = (probs_crops >= np.array(THRESHOLDS)).sum(axis=0).astype(int)  # [C]
    need_votes = max(MIN_VOTES_ABS, int(MIN_PROP * probs_crops.shape[0]))  # Ambang dinamis

    # D) Seleksi per-buah (hadir jika salah satu state kuat & cukup konsensus)
    fruits = [
        ('alpukat_matang', 'alpukat_mentah'),
        ('belimbing_matang', 'belimbing_mentah'),
        ('mangga_matang', 'mangga_mentah')
    ]
    final = []  # (label, prob)

    for a, b in fruits:
        ia, ib = IDX[a], IDX[b]                                  # index dua state
        pa, pb = probs[ia], probs[ib]                             # skor agregat
        va, vb = votes[ia], votes[ib]                             # votes
        tha, thb = THRESHOLDS[ia], THRESHOLDS[ib]                 # ambang

        # Buah dianggap ADA kalau salah satu state ≥ threshold & votes cukup
        present = ((pa >= tha and va >= need_votes) or (pb >= thb and vb >= need_votes))
        if not present:
            continue                                             # skip buah ini

        if EXCLUSIVE_PER_FRUIT:
            # Pilih state yang lebih kuat dan beda skor cukup (biar gak dobel)
            if pa >= pb + CONSISTENCY_MARGIN:
                final.append((a, pa))
            elif pb >= pa + CONSISTENCY_MARGIN:
                final.append((b, pb))
            else:
                # Kalau bedanya tipis, pilih yang lebih besar saja
                chosen = (a, pa) if pa >= pb else (b, pb)
                final.append(chosen)
        else:
            # Boleh muncul dua-duanya (jarang dipakai)
            if pa >= tha and va >= need_votes: final.append((a, pa))
            if pb >= thb and vb >= need_votes: final.append((b, pb))

    # E) Urutkan & tampilkan
    final.sort(key=lambda x: x[1], reverse=True)
    st.subheader("🔍 Label Terdeteksi:")
    if not final:
        st.warning("🚫 Tidak ada label yang memenuhi kriteria (threshold + konsensus).")
    else:
        for lbl, p in final:
            st.write(f"✅ *{lbl}* ({p:.2%})")                    # tanpa menampilkan votes

    # F) Panel detail (kalau mau ngintip angka mentah)
    with st.expander("📊 Detail Probabilitas & Konsensus"):
        mean_prob = float(np.mean(probs))
        entropy   = -float(np.mean([p * math.log(p + 1e-8) for p in probs]))
        st.write(f"🪟 crops={probs_crops.shape[0]} | q={PERCENTILE_Q} | need_votes≥{need_votes} | mean_prob={mean_prob:.3f} | entropy={entropy:.3f}")
        for i, lbl in enumerate(LABELS):
            pass_thr = "✓" if probs[i] >= THRESHOLDS[i] else "✗"
            st.write(f"{lbl}: {probs[i]:.2%} (thr {THRESHOLDS[i]:.2f}) | votes={int(votes[i])} {pass_thr}")
