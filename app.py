# ===========================
# 0) IMPORT & KONFIGURASI
# ===========================
import os, math, traceback                           # utilitas & log
import numpy as np                                   # operasi array
import torch, torch.nn as nn                         # PyTorch
import streamlit as st                               # UI
import gdown                                         # unduh GDrive
from PIL import Image                                # gambar
from safetensors.torch import safe_open              # load .safetensors
import torchvision.transforms as transforms          # transformasi

# ---------------------------
# Konfigurasi umum
# ---------------------------
MODEL_URL  = 'https://drive.google.com/uc?id=1GPPxPSpodNGJHSeWyrTVwRoSjwW3HaA8'
MODEL_PATH = 'model_3.safetensors'

LABELS = [
    'alpukat_matang', 'alpukat_mentah',
    'belimbing_matang', 'belimbing_mentah',
    'mangga_matang', 'mangga_mentah'
]
THRESHOLDS = [0.01, 0.01, 0.01, 0.06, 0.02, 0.01]     # ambang per kelas
SHOW_VOTES = False

# Hyperparameter arsitektur (sesuai retrain-5)
NUM_HEADS, NUM_LAYERS, HIDDEN_DIM = 10, 4, 640
PATCH_SIZE, IMAGE_SIZE = 14, 210

# Sliding-window & pascaproses
WIN_FRACS, STRIDE_FRAC = (1.0, 0.7, 0.5, 0.4), 0.33   # skala & overlap
IOU_NMS   = 0.30                                      # ➜ lebih agresif
IOU_PAIR  = 0.30                                      # matang vs mentah
IOU_XFAM  = 0.50                                      # antar keluarga (alpukat vs mangga vs belimbing)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===========================
# 1) PENGUNDUHAN MODEL
# ===========================
def download_model():
    with st.spinner('🔄 Mengunduh model...'):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 50000:
    if os.path.exists(MODEL_PATH):
        st.warning("📦 File model korup. Mengunduh ulang...")
        os.remove(MODEL_PATH)
    download_model()

# ===========================
# 2) DEFINISI MODEL
# ===========================
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, emb_size):
        super().__init__()
        self.proj = nn.Conv2d(3, emb_size, patch_size, patch_size)   # conv stride=patch
    def forward(self, x):
        x = self.proj(x)                                             # [B,E,H/ps,W/ps]
        return x.flatten(2).transpose(1, 2)                          # [B,N,E]

class WordEmbedding(nn.Module):
    def __init__(self, dim): super().__init__()
    def forward(self, x): return x                                   # dummy (zeros saat inferensi)

class FeatureFusion(nn.Module):
    def forward(self, v, t): return torch.cat([v, t[:, :v.size(1), :]], -1)

class ScaleTransformation(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__(); self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.linear(x)

class ChannelUnification(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.norm = nn.LayerNorm(dim)
    def forward(self, x): return self.norm(x)

class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=NUM_HEADS):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
    def forward(self, x): return self.attn(x, x, x)[0]

class CrossScaleAggregation(nn.Module):
    def forward(self, x): return x.mean(1, keepdim=True)             # [B,1,E]

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
    with safe_open(MODEL_PATH, framework="pt", device=device) as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    model = HSVLTModel(PATCH_SIZE, HIDDEN_DIM, len(LABELS), NUM_HEADS, NUM_LAYERS).to(device)
    model.load_state_dict(state_dict)
    model.eval()
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.code(traceback.format_exc()); st.stop()

# ===========================
# 4) TRANSFORMASI
# ===========================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ===========================
# 5) UTIL SLIDING WINDOW + NMS
# ===========================
def gen_windows(image_pil):
    """Generate crops + koordinat (xyxy)."""
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
    """Prediksi per-crop."""
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
    """IoU kotak xyxy."""
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
            if iou_xyxy(boxes[i], boxes[j]) <= thr:  # buang yang IoU besar
                rest.append(j)
        idxs = np.array(rest, dtype=int)
    return keep

def merge_pairs(detections):
    """Gabung matang/mentah per keluarga."""
    pairs = [('alpukat_matang', 'alpukat_mentah'),
             ('belimbing_matang', 'belimbing_mentah'),
             ('mangga_matang',   'mangga_mentah')]
    final = []
    for a, b in pairs:
        A = [d for d in detections if d["label"] == a]
        B = [d for d in detections if d["label"] == b]
        used_B = set()
        for da in A:
            best_j, best_iou = -1, 0.0
            for j, db in enumerate(B):
                if j in used_B: continue
                iou = iou_xyxy(da["box"], db["box"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= IOU_PAIR:
                chosen = da if da["score"] >= B[best_j]["score"] else B[best_j]
                final.append(chosen); used_B.add(best_j)
            else:
                final.append(da)
        final.extend(db for j, db in enumerate(B) if j not in used_B)
    for d in final: d["fam"] = d["label"].split('_')[0]              # tambah info keluarga
    return final

def cross_family_suppress(dets, thr=IOU_XFAM):
    """Tekan overlap antar keluarga berbeda."""
    dets = sorted(dets, key=lambda d: d["score"], reverse=True)
    kept = []
    for d in dets:
        ok = True
        for k in kept:
            if d["fam"] != k["fam"] and iou_xyxy(d["box"], k["box"]) >= thr:
                ok = False; break
        if ok: kept.append(d)
    return kept

def single_object_gate(dets, thr=0.50):
    """Jika semua box di 1 lokasi, ambil 1 label skor tertinggi."""
    if len(dets) <= 1: return dets
    base = dets[0]["box"]
    same_spot = all(iou_xyxy(base, d["box"]) >= thr for d in dets[1:])
    return [max(dets, key=lambda d: d["score"])] if same_spot else dets

def postprocess_with_nms(boxes, probs_crops, votes, min_votes, iou_nms=IOU_NMS):
    """Filter → NMS per kelas → merge matang/mentah → suppress antar keluarga → gate single-object."""
    detections = []
    for c, label in enumerate(LABELS):
        scores = probs_crops[:, c]
        mask = (scores >= THRESHOLDS[c]) & (votes[c] >= min_votes)   # ➜ pakai votes
        if not mask.any(): continue
        b = boxes[mask]; s = scores[mask]
        keep = nms_per_class(b, s, iou_nms)
        for k in keep:
            detections.append({"label": label, "c": c, "score": float(s[k]), "box": b[k]})
    dets = merge_pairs(detections)
    dets = cross_family_suppress(dets, thr=IOU_XFAM)
    dets = single_object_gate(dets, thr=0.50)
    return sorted(dets, key=lambda d: d["score"], reverse=True)

# ===========================
# 6) UI
# ===========================
st.title("🍉 Klasifikasi Multi-Label Buah")
uploaded_file = st.file_uploader("Unggah gambar buah", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar Input", use_container_width=True)

    # 1) Sliding-window
    crops, boxes     = gen_windows(image)
    probs_crops      = run_preds_on_crops(crops)
    num_crops        = probs_crops.shape[0]

    # 2) Konsensus votes per kelas
    votes            = (probs_crops >= np.array(THRESHOLDS)).sum(0).astype(int)
    MIN_VOTES_dyn    = max(3, int(math.ceil(0.05 * max(1, num_crops))))  # ≥3 atau 5% crops

    # 3) NMS + suppression
    final_dets = postprocess_with_nms(boxes, probs_crops, votes, MIN_VOTES_dyn, iou_nms=IOU_NMS)

    # 4) Tampilkan
    st.subheader("🔍 Label Terdeteksi:")
    if not final_dets:
        st.warning("🚫 Tidak ada label terdeteksi.")
    else:
        st.write(f"Total label: **{len(final_dets)}**")
        for det in final_dets:
            line = f"✅ *{det['label']}* ({det['score']:.2%})"
            if SHOW_VOTES: line += f" | votes={votes[det['c']]}"
            st.write(line)

    # 5) Panel analitik (opsional)
    with st.expander("📊 Detail Probabilitas & Konsensus"):
        probs_max = probs_crops.max(0) if num_crops else np.zeros(len(LABELS))
        mean_prob = float(np.mean(probs_max)) if num_crops else 0.0
        entropy   = -float(np.mean([p * math.log(p + 1e-8) for p in probs_max])) if num_crops else 0.0
        st.write(f"🪟 crops: {num_crops} | min_votes: {MIN_VOTES_dyn} | mean_prob: {mean_prob:.3f} | entropy: {entropy:.3f}")
        for i, lbl in enumerate(LABELS):
            pass_thr = "✓" if probs_max[i] >= THRESHOLDS[i] else "✗"
            line = f"{lbl}: {probs_max[i]:.2%} (thr {THRESHOLDS[i]:.2f}) {pass_thr}"
            if SHOW_VOTES: line += f" | votes={votes[i]}"
            st.write(line)
