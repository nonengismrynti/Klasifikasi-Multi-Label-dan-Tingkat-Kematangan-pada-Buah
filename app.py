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

# =========================
# 1) Setup umum & parameter
# =========================
MODEL_URL  = 'https://drive.google.com/uc?id=1GPPxPSpodNGJHSeWyrTVwRoSjwW3HaA8'  # Link model di Drive
MODEL_PATH = 'model_3.safetensors'                                               # Nama file lokal model

# Urutan label HARUS sama dengan saat training
LABELS = [
    'alpukat_matang', 'alpukat_mentah',
    'belimbing_matang', 'belimbing_mentah',
    'mangga_matang', 'mangga_mentah'
]

# ---- Parameter model (mengikuti retrain-5) ----
NUM_HEADS   = 10               # Jumlah attention heads
NUM_LAYERS  = 4                # Banyak InteractionBlock
HIDDEN_DIM  = 640              # Dimensi embedding
PATCH_SIZE  = 14               # Ukuran patch conv
IMAGE_SIZE  = 210              # Ukuran input ke model

# ---- Threshold dasar dari tuning (longgar) ----
BASE_THRESHOLDS = [0.01, 0.01, 0.01, 0.06, 0.02, 0.01]  # ambang awal hasil validasi

# ---- Floor threshold untuk deployment (lebih ketat utk kelas rawan FP) ----
THRESHOLDS_FLOOR = {
    'alpukat_matang':   0.15,   # naikin sedikit
    'alpukat_mentah':   0.15,   # naikin sedikit
    'belimbing_matang': 0.05,   # tetap longgar (mudah dibaca)
    'belimbing_mentah': 0.05,   # tetap longgar
    'mangga_matang':    0.25,   # ketatin karena sering false positive
    'mangga_mentah':    0.20    # ketatin
}
# Gabungkan base & floor → pakai nilai yang lebih tinggi per kelas
THRESHOLDS = [max(b, THRESHOLDS_FLOOR[lbl]) for b, lbl in zip(BASE_THRESHOLDS, LABELS)]

# ---- Sliding-window di tahap prediksi (saran dosen) ----
WIN_FRACS        = (1.0, 0.7, 0.5, 0.4)   # multi-skala (porsi sisi terpendek)
STRIDE_FRAC      = 0.33                   # overlap ~67%
INCLUDE_FULL_IMG = False                  # jangan pakai full image (objek dominan suka “menang”)
TOPK             = 3                      # agregasi: rata-rata top-K per kelas (tahan outlier)
MIN_PROP         = 0.10                   # minimal proporsi crop yg “setuju” (≥ threshold)
MIN_VOTES_ABS    = 2                      # minimal votes absolut
EXCLUSIVE_PER_FRUIT = False               # False = matang & mentah bisa muncul bersamaan

# =========================
# 2) Download model bila perlu
# =========================
def download_model():
    with st.spinner('🔄 Mengunduh ulang model dari Google Drive...'):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)  # Unduh file

# Cek file model (antisipasi korup)
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 50000:
    if os.path.exists(MODEL_PATH):
        st.warning("📦 Ukuran file model terlalu kecil, kemungkinan korup. Mengunduh ulang...")
        os.remove(MODEL_PATH)             # Hapus file korup
    download_model()                      # Unduh ulang

# =========================
# 3) Definisi arsitektur (identik training)
# =========================
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, emb_size):
        super().__init__()
        self.proj = nn.Conv2d(3, emb_size, kernel_size=patch_size, stride=patch_size)  # Conv patch
    def forward(self, x):
        x = self.proj(x)                 # [B, E, H/ps, W/ps]
        x = x.flatten(2).transpose(1, 2) # [B, N, E]
        return x

class WordEmbedding(nn.Module):
    def __init__(self, dim): super().__init__()         # dummy class
    def forward(self, x): return x                      # pass-through

class FeatureFusion(nn.Module):
    def forward(self, v, t):                            # v,t: [B, N, E]
        return torch.cat([v, t[:, :v.size(1), :]], dim=-1)  # → [B, N, 2E]

class ScaleTransformation(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__(); self.linear = nn.Linear(in_dim, out_dim)  # 2E → E
    def forward(self, x): return self.linear(x)

class ChannelUnification(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.norm = nn.LayerNorm(dim)  # LayerNorm
    def forward(self, x): return self.norm(x)

class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=NUM_HEADS):
        super().__init__(); self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
    def forward(self, x): return self.attn(x, x, x)[0]     # Self-attention

class CrossScaleAggregation(nn.Module):
    def forward(self, x): return x.mean(dim=1, keepdim=True)  # [B, 1, E]

class HamburgerHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__(); self.linear = nn.Linear(in_dim, out_dim)  # Linear head
    def forward(self, x): return self.linear(x)

class MLPClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, num_classes))  # MLP
    def forward(self, x): return self.mlp(x)

class HSVLTModel(nn.Module):
    def __init__(self, img_size=210, patch_size=14, emb_size=HIDDEN_DIM,
                 num_classes=6, num_heads=NUM_HEADS, num_layers=NUM_LAYERS):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, emb_size)      # Patch emb
        self.word_embed  = WordEmbedding(emb_size)                             # Word emb (dummy)
        self.concat      = FeatureFusion()                                     # Fuse vis+text
        self.scale_transform     = ScaleTransformation(emb_size * 2, emb_size) # 2E→E
        self.channel_unification = ChannelUnification(emb_size)                # Norm
        self.interaction_blocks  = nn.Sequential(                              # tumpuk attention
            *[InteractionBlock(emb_size, num_heads=num_heads) for _ in range(num_layers)]
        )
        self.csa        = CrossScaleAggregation()                              # CSA sederhana
        self.head       = HamburgerHead(emb_size, emb_size)                    # Linear head
        self.classifier = MLPClassifier(emb_size, num_classes)                 # Klasifier

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
        return self.classifier(x)                # [B, C] logits

# =========================
# 4) Load model ke device
# =========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # pilih GPU kalau ada
if len(THRESHOLDS) != len(LABELS):                       # sanity check thresholds
    st.error("Panjang THRESHOLDS tidak sama dengan jumlah LABELS."); st.stop()

try:
    with safe_open(MODEL_PATH, framework="pt", device=device) as f:  # buka .safetensors
        state_dict = {k: f.get_tensor(k) for k in f.keys()}          # load semua tensor
    model = HSVLTModel(                                              # inisialisasi arsitektur
        patch_size=PATCH_SIZE, emb_size=HIDDEN_DIM, num_classes=len(LABELS),
        num_heads=NUM_HEADS, num_layers=NUM_LAYERS
    ).to(device)
    model.load_state_dict(state_dict, strict=True)                   # muat bobot
    model.eval()                                                     # mode eval
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")                           # tampilkan error
    st.code(traceback.format_exc()); st.stop()                       # tampilkan traceback & stop

# =========================
# 5) Transformasi gambar
# =========================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),                     # resize ke 210x210
    transforms.ToTensor(),                                           # ke tensor [0..1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],                 # normalisasi ImageNet
                         std =[0.229, 0.224, 0.225])
])

# =========================
# 5.1) Sliding-window + agregasi robust
# =========================
def sliding_window_infer(image_pil, model, transform, device,
                         hidden_dim, patch_size,
                         win_fracs=WIN_FRACS, stride_frac=STRIDE_FRAC,
                         include_full=INCLUDE_FULL_IMG, topk=TOPK):
    """
    Potong gambar (multi-skala, overlap), prediksi per-crop,
    lalu agregasi per kelas pakai mean dari top-K skor (lebih tahan outlier).
    """
    W, H = image_pil.size                                         # ukuran asli
    short = min(W, H)                                             # sisi terpendek
    crops = []                                                    # list crop

    # Bangun crop persegi di berbagai skala
    for wf in win_fracs:
        win  = max(int(short * wf), 64)                           # ukuran jendela (min 64px)
        step = max(1, int(win * stride_frac))                     # stride (overlap)
        for top in range(0, max(H - win + 1, 1), step):           # loop vertikal
            for left in range(0, max(W - win + 1, 1), step):      # loop horizontal
                crops.append(image_pil.crop((left, top, left + win, top + win)))  # tambah crop

    if include_full:
        crops.append(image_pil)                                    # opsional: tambah full image

    num_tokens = (IMAGE_SIZE // patch_size) ** 2                   # jumlah token dummy (225)
    probs_list = []                                                # simpan prediksi setiap crop

    with torch.no_grad():                                          # non-grad (lebih cepat/hemat)
        for crop in crops:
            x = transform(crop).unsqueeze(0).to(device)            # transform + batch=1
            dummy_text = torch.zeros((1, num_tokens, hidden_dim), device=device)  # dummy zeros
            p = torch.sigmoid(model(x, dummy_text)).cpu().numpy()[0]              # prob [C]
            probs_list.append(p)                                   # simpan

    probs_crops = np.stack(probs_list, axis=0)                     # [N, C] semua crop
    K = min(topk, probs_crops.shape[0])                            # K tidak > jumlah crop
    topk_sorted = np.sort(probs_crops, axis=0)[-K:, :]             # ambil K skor tertinggi per kelas
    probs_topk_mean = topk_sorted.mean(axis=0)                     # rata-rata top-K per kelas → [C]
    return probs_topk_mean, probs_crops                            # kembalikan agregat & raw

# =========================
# 6) UI
# =========================
st.title("🍉 Klasifikasi Multi-Label Buah")                         # judul app
st.write("Upload gambar buah; sistem akan mendeteksi beberapa label sekaligus.")  # deskripsi

uploaded_file = st.file_uploader("Unggah gambar buah", type=['jpg', 'jpeg', 'png'])  # uploader

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')               # baca jadi RGB
    st.image(image, caption="Gambar Input", use_container_width=True)  # tampilkan

    # 1) Sliding-window (multi-skala + agregasi robust)
    probs_vec, probs_crops = sliding_window_infer(
        image_pil=image, model=model, transform=transform, device=device,
        hidden_dim=HIDDEN_DIM, patch_size=PATCH_SIZE,
        win_fracs=WIN_FRACS, stride_frac=STRIDE_FRAC,
        include_full=INCLUDE_FULL_IMG, topk=TOPK
    )
    probs = probs_vec.tolist()                                     # ke list python

    # 2) Votes berbasis proporsi (berapa crop ≥ threshold per kelas)
    votes = (probs_crops >= np.array(THRESHOLDS)).sum(axis=0).astype(int)       # [C]
    need_votes = max(MIN_VOTES_ABS, int(MIN_PROP * probs_crops.shape[0]))       # ambang dinamis

    # 3) Kandidat = skor ≥ threshold & votes cukup
    candidates = {
        lbl: float(p)
        for lbl, p, thr, v in zip(LABELS, probs, THRESHOLDS, votes.tolist())
        if (p >= thr and v >= need_votes)
    }

    # 4) (Opsional) Aturan eksklusif matang vs mentah per buah
    pairs = [
        ('alpukat_matang', 'alpukat_mentah'),
        ('belimbing_matang', 'belimbing_mentah'),
        ('mangga_matang', 'mangga_mentah')
    ]
    final_labels = []                                              # hasil akhir utk ditampilkan
    if EXCLUSIVE_PER_FRUIT:                                        # jika eksklusif → pilih 1
        for a, b in pairs:
            has_a, has_b = a in candidates, b in candidates
            if has_a and has_b:
                chosen = a if candidates[a] >= candidates[b] else b
                final_labels.append((chosen, candidates[chosen]))  # (label, prob)
            elif has_a:
                final_labels.append((a, candidates[a]))
            elif has_b:
                final_labels.append((b, candidates[b]))
    else:
        final_labels = sorted(candidates.items(), key=lambda x: x[1], reverse=True)  # non-eksklusif

    # 5) Tampilkan (tanpa menulis jumlah votes di daftar utama)
    st.subheader("🔍 Label Terdeteksi:")
    if not final_labels:
        st.warning("🚫 Tidak ada label yang memenuhi kriteria (threshold + konsensus).")
    else:
        for label, prob in final_labels:
            st.write(f"✅ *{label}* ({prob:.2%})")                # hanya label + persen

    # 6) Panel detail (opsional): tampilkan statistik lengkap
    with st.expander("📊 Detail Probabilitas & Konsensus"):
        mean_prob = float(np.mean(probs))                         # rata-rata prob
        entropy   = -float(np.mean([p * math.log(p + 1e-8) for p in probs]))  # entropi rata2
        st.write(f"🪟 crops: {probs_crops.shape[0]} | topK={TOPK} | need_votes≥{need_votes} | mean_prob={mean_prob:.3f} | entropy={entropy:.3f}")
        for i, lbl in enumerate(LABELS):
            pass_thr = "✓" if probs[i] >= THRESHOLDS[i] else "✗"  # lulus threshold?
            st.write(f"{lbl}: {probs[i]:.2%} (thr {THRESHOLDS[i]:.2f}) | votes={int(votes[i])} {pass_thr}")
