# ===========================
# 0) IMPORT & KONFIGURASI
# ===========================
import os, math, traceback
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import gdown
from PIL import Image
from safetensors.torch import safe_open
import torchvision.transforms as transforms

# ---------------------------
# Konfigurasi umum
# ---------------------------
MODEL_URL  = 'https://drive.google.com/uc?id=1GPPxPSpodNGJHSeWyrTVwRoSjwW3HaA8'  # URL model
MODEL_PATH = 'model_3.safetensors'                                               # Nama file lokal

LABELS = [                                                                       # Urutan label = saat training
    'alpukat_matang', 'alpukat_mentah',
    'belimbing_matang', 'belimbing_mentah',
    'mangga_matang',   'mangga_mentah'
]
THRESHOLDS = [0.01, 0.01, 0.01, 0.06, 0.02, 0.01]                                # Ambang per-kelas
SHOW_VOTES = False                                                                # Tampilkan votes di UI?

# Hyperparameter arsitektur (sesuai retrain-5)
NUM_HEADS, NUM_LAYERS, HIDDEN_DIM = 10, 4, 640
PATCH_SIZE, IMAGE_SIZE = 14, 210

# Sliding-window & pascaproses
WIN_FRACS    = (1.0, 0.7, 0.5, 0.4)    # Skala window relatif sisi terpendek
STRIDE_FRAC  = 0.33                     # Overlap besar (stride = 0.33*win)
IOU_NMS      = 0.30                     # NMS per kelas (agresif)
RADIUS_FRAC  = 0.30                     # Radius klaster (30% sisi terpendek)
IOU_EPS      = 0.05                     # IoU kecil untuk bantu klaster (kalau jarak pusat pas-pasan)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===========================
# 1) PENGUNDUHAN MODEL
# ===========================
def download_model():                                                        # Unduh model jika perlu
    with st.spinner('🔄 Mengunduh model...'):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 50000:   # Cegah file korup
    if os.path.exists(MODEL_PATH):
        st.warning("📦 File model korup. Mengunduh ulang...")
        os.remove(MODEL_PATH)
    download_model()

# ===========================
# 2) DEFINISI MODEL
# ===========================
class PatchEmbedding(nn.Module):                                             # Gambar → token patch
    def __init__(self, img_size, patch_size, emb_size):
        super().__init__()
        self.proj = nn.Conv2d(3, emb_size, patch_size, patch_size)
    def forward(self, x):
        x = self.proj(x)                                                     # [B,E,H/ps,W/ps]
        return x.flatten(2).transpose(1, 2)                                  # [B,N,E]

class WordEmbedding(nn.Module):                                              # Dummy (pass-through)
    def __init__(self, dim): super().__init__()
    def forward(self, x): return x

class FeatureFusion(nn.Module):                                              # Gabung visual + “teks”
    def forward(self, v, t): return torch.cat([v, t[:, :v.size(1), :]], -1)

class ScaleTransformation(nn.Module):                                        # 2E → E
    def __init__(self, in_dim, out_dim):
        super().__init__(); self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.linear(x)

class ChannelUnification(nn.Module):                                         # LayerNorm token
    def __init__(self, dim):
        super().__init__(); self.norm = nn.LayerNorm(dim)
    def forward(self, x): return self.norm(x)

class InteractionBlock(nn.Module):                                           # Self-attention
    def __init__(self, dim, num_heads=NUM_HEADS):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
    def forward(self, x): return self.attn(x, x, x)[0]

class CrossScaleAggregation(nn.Module):                                      # Agregasi token
    def forward(self, x): return x.mean(1, keepdim=True)                     # [B,1,E]

class HamburgerHead(nn.Module):                                              # Head linear
    def __init__(self, in_dim, out_dim):
        super().__init__(); self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.linear(x)

class MLPClassifier(nn.Module):                                              # Klasifier multi-label
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, num_classes))
    def forward(self, x): return self.mlp(x)

class HSVLTModel(nn.Module):                                                 # Rangkaian lengkap model
    def __init__(self, patch_size, emb_size, num_classes, num_heads, num_layers):
        super().__init__()
        self.patch_embed = PatchEmbedding(IMAGE_SIZE, patch_size, emb_size)
        self.word_embed  = WordEmbedding(emb_size)
        self.concat      = FeatureFusion()
        self.scale_transform     = ScaleTransformation(emb_size * 2, emb_size)
        self.channel_unification = ChannelUnification(emb_size)
        self.interaction_blocks  = nn.Sequential(*[InteractionBlock(emb_size, num_heads) for _ in range(num_layers)])
        self.csa        = CrossScaleAggregation()
        self.head       = HamburgerHead(emb_size, emb_size)
        self.classifier = MLPClassifier(emb_size, num_classes)
    def forward(self, image, text):
        x = self.concat(self.patch_embed(image), self.word_embed(text))
        x = self.scale_transform(x)
        x = self.channel_unification(x)
        x = self.interaction_blocks(x)
        x = self.csa(x)
        x = self.head(x).mean(1)
        return self.classifier(x)

# ===========================
# 3) LOAD MODEL
# ===========================
try:
    with safe_open(MODEL_PATH, framework="pt", device=device) as f:          # Baca bobot
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    model = HSVLTModel(PATCH_SIZE, HIDDEN_DIM, len(LABELS), NUM_HEADS, NUM_LAYERS).to(device)
    model.load_state_dict(state_dict)                                        # Pasang bobot
    model.eval()
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.code(traceback.format_exc()); st.stop()

# ===========================
# 4) TRANSFORMASI GAMBAR
# ===========================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ===========================
# 5) UTIL: SLIDING, NMS, CLUSTER
# ===========================
def gen_windows(image_pil):
    """Buat semua crop sliding-window + koordinat (xyxy)."""
    W, H = image_pil.size
    short = min(W, H)
    boxes, crops = [], []
    for wf in WIN_FRACS:
        win  = max(int(short * wf), 64)
        step = max(1, int(win * STRIDE_FRAC))
        for top in range(0, max(H - win + 1, 1), step):
            for left in range(0, max(W - win + 1, 1), step):
                box = (left, top, left + win, top + win)
                boxes.append(box); crops.append(image_pil.crop(box))
    return crops, np.array(boxes, np.float32)

def run_preds_on_crops(crops):
    """Prediksi per-crop (prob kelas)."""
    num_tokens = (IMAGE_SIZE // PATCH_SIZE) ** 2
    out = []
    with torch.no_grad():
        for crop in crops:
            x = transform(crop).unsqueeze(0).to(device)
            dummy_text = torch.zeros((1, num_tokens, HIDDEN_DIM), device=device)
            p = torch.sigmoid(model(x, dummy_text)).cpu().numpy()[0]
            out.append(p)
    return np.stack(out) if out else np.zeros((0, len(LABELS)), dtype=np.float32)

def iou_xyxy(a, b):
    """Hitung IoU dua box (xyxy)."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter  = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-8)

def nms_per_class(boxes, scores, thr):
    """Greedy NMS per kelas."""
    idxs = np.argsort(-scores)
    keep = []
    while idxs.size:
        i = idxs[0]; keep.append(i)
        rest = []
        for j in idxs[1:]:
            if iou_xyxy(boxes[i], boxes[j]) <= thr:
                rest.append(j)
        idxs = np.array(rest, dtype=int)
    return keep

def fam_of(label): return label.split('_')[0]                               # “keluarga” buah

def close_enough(box_a, box_b, radius, iou_eps=IOU_EPS):
    """Dua box dianggap 1 objek jika pusatnya dekat ATAU IoU kecil tapi masih beririsan."""
    cx1 = (box_a[0] + box_a[2]) / 2; cy1 = (box_a[1] + box_a[3]) / 2
    cx2 = (box_b[0] + box_b[2]) / 2; cy2 = (box_b[1] + box_b[3]) / 2
    center_dist = ((cx1 - cx2)**2 + (cy1 - cy2)**2) ** 0.5
    return (center_dist <= radius) or (iou_xyxy(box_a, box_b) >= iou_eps)

def cluster_per_family(dets, img_wh, radius_frac=RADIUS_FRAC):
    """
    Klaster deteksi per keluarga menggunakan kedekatan pusat / IoU kecil.
    Satu klaster ≈ satu buah nyata.
    """
    if not dets: return []
    W, H = img_wh; radius = radius_frac * min(W, H)
    # urut skor desc biar yang kuat jadi pusat klaster
    order = np.argsort([-d['score'] for d in dets])
    used  = [False]*len(dets)
    clusters = []
    for i in order:
        if used[i]: continue
        di = dets[i]; fam = di['fam']
        group = [i]; used[i] = True
        for j in order:
            if used[j]: continue
            dj = dets[j]
            if dj['fam'] != fam: continue
            if close_enough(di['box'], dj['box'], radius):
                used[j] = True; group.append(j)
        clusters.append(group)
    return clusters

def single_object_gate(dets, thr=0.50):
    """Jika semua box tumpuk di satu lokasi → pilih 1 label terbaik."""
    if len(dets) <= 1: return dets
    base = dets[0]["box"]
    same_spot = all(iou_xyxy(base, d["box"]) >= thr for d in dets[1:])
    return [max(dets, key=lambda d: d["score"])] if same_spot else dets

def postprocess(boxes, probs_crops, img_size, min_votes):
    """
    1) Top-1 gating + threshold + votes
    2) NMS per kelas
    3) Klaster per keluarga (gabung duplikasi)
    4) Ambil 1 label terbaik per klaster (otomatis pilih matang/mentah yang paling kuat)
    5) Single-object gate
    """
    if probs_crops.size == 0:
        return []

    # --- Top-1 gating: hanya kelas tertinggi per crop yang dipertimbangkan
    top_idx = np.argmax(probs_crops, axis=1)

    # --- Kumpulkan deteksi awal per kelas + NMS
    dets = []
    for c, label in enumerate(LABELS):
        scores = probs_crops[:, c]
        mask = (scores >= THRESHOLDS[c]) & (top_idx == c)
        if mask.sum() < min_votes:
            continue
        b = boxes[mask]; s = scores[mask]
        if b.size == 0: continue
        keep = nms_per_class(b, s, IOU_NMS)
        for k in keep:
            dets.append({"label": label, "fam": fam_of(label),
                         "c": c, "score": float(s[k]), "box": b[k]})

    if not dets:
        return []

    # --- Klaster per keluarga → 1 buah = 1 klaster
    clusters = cluster_per_family(dets, img_size, radius_frac=RADIUS_FRAC)

    # --- Ambil satu label terbaik per klaster (sekalian resolve matang vs mentah)
    chosen = []
    for group in clusters:
        best = max((dets[i] for i in group), key=lambda d: d['score'])
        chosen.append(best)

    # --- Single-object gate (kalau memang 1 objek saja)
    chosen = single_object_gate(chosen, thr=0.50)

    # Urutkan skor desc untuk tampilan
    return sorted(chosen, key=lambda d: d["score"], reverse=True)

# ===========================
# 6) UI
# ===========================
st.title("🍉 Klasifikasi Multi-Label Buah")
uploaded_file = st.file_uploader("Unggah gambar buah", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar Input", use_container_width=True)

    # 1) Sliding-window
    crops, boxes  = gen_windows(image)
    probs_crops   = run_preds_on_crops(crops)
    num_crops     = probs_crops.shape[0]

    # 2) Votes dinamis (ketat): ≥ max(3, 12% dari jumlah crop)
    MIN_VOTES_dyn = max(3, int(math.ceil(0.12 * max(1, num_crops))))

    # 3) Pascaproses (NMS + cluster → 1 label per objek)
    final_dets = postprocess(boxes, probs_crops, image.size, MIN_VOTES_dyn)

    # 4) Tampilkan
    st.subheader("🔍 Label Terdeteksi:")
    if not final_dets:
        st.warning("🚫 Tidak ada label terdeteksi.")
    else:
        st.write(f"Total label: **{len(final_dets)}**")
        for det in final_dets:
            line = f"✅ *{det['label']}* ({det['score']:.2%})"
            if SHOW_VOTES:
                # votes ditampilkan global per kelas (opsional)
                pass
            st.write(line)

    # 5) Panel analitik (opsional)
    with st.expander("📊 Detail Probabilitas & Konsensus"):
        probs_max = probs_crops.max(0) if num_crops else np.zeros(len(LABELS))
        mean_prob = float(np.mean(probs_max)) if num_crops else 0.0
        entropy   = -float(np.mean([p * math.log(p + 1e-8) for p in probs_max])) if num_crops else 0.0
        st.write(f"🪟 crops: {num_crops} | min_votes: {MIN_VOTES_dyn} | mean_prob: {mean_prob:.3f} | entropy: {entropy:.3f}")
        for i, lbl in enumerate(LABELS):
            pass_thr = "✓" if probs_max[i] >= THRESHOLDS[i] else "✗"
            st.write(f"{lbl}: {probs_max[i]:.2%} (thr {THRESHOLDS[i]:.2f}) {pass_thr}")
