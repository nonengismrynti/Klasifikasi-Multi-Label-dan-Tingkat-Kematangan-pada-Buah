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
MODEL_URL  = 'https://drive.google.com/uc?id=1GPPxPSpodNGJHSeWyrTVwRoSjwW3HaA8'
MODEL_PATH = 'model_3.safetensors'

LABELS = [
    'alpukat_matang', 'alpukat_mentah',
    'belimbing_matang', 'belimbing_mentah',
    'mangga_matang', 'mangga_mentah'
]
THRESHOLDS = [0.01, 0.01, 0.01, 0.06, 0.02, 0.01]      # ambang per kelas (hasil tuning)
SHOW_VOTES = False                                     # tampilkan votes di UI

# Hyperparameter arsitektur (sesuai retrain-5)
NUM_HEADS, NUM_LAYERS, HIDDEN_DIM = 10, 4, 640
PATCH_SIZE, IMAGE_SIZE = 14, 210

# Sliding-window & pascaproses
WIN_FRACS    = (1.0, 0.7, 0.5, 0.4)    # skala jendela relatif sisi pendek
STRIDE_FRAC  = 0.33                    # overlap besar (stride = 0.33*win)
IOU_NMS      = 0.30                    # NMS per kelas (agresif)
IOU_PAIR     = 0.30                    # merge matang vs mentah
IOU_XFAM     = 0.50                    # suppress antar keluarga buah
CENTER_FRAC  = 0.18                    # radius cluster = 18% sisi pendek

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===========================
# 1) PENGUNDUHAN MODEL
# ===========================
def download_model():
    """Unduh model dari Google Drive jika tidak ada/korup."""
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
    """Ubah gambar → token patch."""
    def __init__(self, img_size, patch_size, emb_size):
        super().__init__()
        self.proj = nn.Conv2d(3, emb_size, patch_size, patch_size)   # stride=patch
    def forward(self, x):
        x = self.proj(x)                                             # [B,E,H/ps,W/ps]
        return x.flatten(2).transpose(1, 2)                          # [B,N,E]

class WordEmbedding(nn.Module):
    """Dummy: saat inferensi kita isi nol; modul ini pass-through saja."""
    def __init__(self, dim): super().__init__()
    def forward(self, x): return x

class FeatureFusion(nn.Module):
    """Gabungkan visual + 'teks' (dummy)."""
    def forward(self, v, t): return torch.cat([v, t[:, :v.size(1), :]], -1)

class ScaleTransformation(nn.Module):
    """Proyeksi 2E → E."""
    def __init__(self, in_dim, out_dim):
        super().__init__(); self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.linear(x)

class ChannelUnification(nn.Module):
    """LayerNorm token."""
    def __init__(self, dim):
        super().__init__(); self.norm = nn.LayerNorm(dim)
    def forward(self, x): return self.norm(x)

class InteractionBlock(nn.Module):
    """Self-attention block."""
    def __init__(self, dim, num_heads=NUM_HEADS):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
    def forward(self, x): return self.attn(x, x, x)[0]

class CrossScaleAggregation(nn.Module):
    """Agregasi sederhana antar token."""
    def forward(self, x): return x.mean(1, keepdim=True)             # [B,1,E]

class HamburgerHead(nn.Module):
    """Linear head penyelarasan fitur."""
    def __init__(self, in_dim, out_dim):
        super().__init__(); self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.linear(x)

class MLPClassifier(nn.Module):
    """Klasifier multi-label (logits)."""
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, num_classes))
    def forward(self, x): return self.mlp(x)

class HSVLTModel(nn.Module):
    """Rangkaian lengkap model (identik saat training)."""
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
# 5) UTIL SLIDING WINDOW + NMS + CLUSTER
# ===========================
def gen_windows(image_pil):
    """Buat semua crop sliding-window + koordinat (xyxy)."""
    W, H = image_pil.size
    short = min(W, H)
    boxes, crops = [], []
    for wf in WIN_FRACS:
        win  = max(int(short * wf), 64)              # min 64 px
        step = max(1, int(win * STRIDE_FRAC))        # stride kecil → overlap besar
        for top in range(0, max(H - win + 1, 1), step):
            for left in range(0, max(W - win + 1, 1), step):
                box = (left, top, left + win, top + win)
                boxes.append(box); crops.append(image_pil.crop(box))
    return crops, np.array(boxes, np.float32)

def run_preds_on_crops(crops):
    """Prediksi per-crop (probabilitas multi-label)."""
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
    """Greedy NMS per kelas (buang yang overlap besar)."""
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

def merge_pairs(detections):
    """
    Gabungkan matang vs mentah per keluarga:
    jika overlap ≥ IOU_PAIR, ambil yang skornya lebih tinggi.
    """
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
    for d in final: d["fam"] = d["label"].split('_')[0]             # info keluarga
    return final

def cross_family_suppress(dets, thr=IOU_XFAM):
    """Tekan overlap antar keluarga berbeda (mis. belimbing vs mangga)."""
    dets = sorted(dets, key=lambda d: d["score"], reverse=True)
    kept = []
    for d in dets:
        ok = True
        for k in kept:
            if d["fam"] != k["fam"] and iou_xyxy(d["box"], k["box"]) >= thr:
                ok = False; break
        if ok: kept.append(d)
    return kept

def cluster_by_center(dets, img_wh, frac=CENTER_FRAC):
    """
    Gabungkan deteksi yang pusatnya berdekatan walau IoU kecil.
    – radius = frac × sisi terpendek gambar
    – kelompokkan per 'fam' (alpukat/belimbing/mangga)
    – pilih satu label terbaik (skor tertinggi) per klaster
    """
    W, H = img_wh; radius = frac * min(W, H)
    if not dets: return dets
    order = np.argsort([-d['score'] for d in dets])      # urut skor desc
    used  = [False]*len(dets)
    clusters = []
    for i in order:
        if used[i]: continue
        di = dets[i]; cx = (di['box'][0]+di['box'][2])/2; cy = (di['box'][1]+di['box'][3])/2
        fam = di['fam']; group = [i]; used[i] = True
        for j in order:
            if used[j]: continue
            dj = dets[j]
            if dj['fam'] != fam: continue
            cxj = (dj['box'][0]+dj['box'][2])/2; cyj = (dj['box'][1]+dj['box'][3])/2
            if ((cx-cxj)**2 + (cy-cyj)**2)**0.5 <= radius:
                used[j] = True; group.append(j)
        # Ambil satu label terbaik dari klaster
        best = max((dets[k] for k in group), key=lambda d: d['score'])
        clusters.append(best)
    return clusters

def single_object_gate(dets, thr=0.50):
    """Jika semua box menumpuk di satu lokasi → pilih 1 label tertinggi."""
    if len(dets) <= 1: return dets
    base = dets[0]["box"]
    same_spot = all(iou_xyxy(base, d["box"]) >= thr for d in dets[1:])
    return [max(dets, key=lambda d: d["score"])] if same_spot else dets

def postprocess_with_nms(boxes, probs_crops, img_size, min_votes,
                         iou_nms=IOU_NMS):
    """
    1) Top-1 gating + threshold + votes
    2) NMS per kelas
    3) Merge matang/mentah
    4) Cross-family suppress
    5) Center-clustering (gabung duplikasi non-overlap)
    6) Single-object gate
    """
    if probs_crops.size == 0:
        return []

    # --- Top-1 gating: hanya hitung crop untuk kelas yg jadi top di crop tsb
    top_idx = np.argmax(probs_crops, axis=1)

    detections = []
    for c, label in enumerate(LABELS):
        scores = probs_crops[:, c]
        keep_mask = (scores >= THRESHOLDS[c]) & (top_idx == c)
        if keep_mask.sum() < min_votes:        # syarat votes ketat
            continue
        b = boxes[keep_mask]; s = scores[keep_mask]
        if b.size == 0: continue
        keep = nms_per_class(b, s, iou_nms)    # NMS per kelas
        for k in keep:
            detections.append({"label": label, "c": c, "score": float(s[k]), "box": b[k]})

    # Tahap gabungan & penekanan
    dets = merge_pairs(detections)
    dets = cross_family_suppress(dets, thr=IOU_XFAM)
    dets = cluster_by_center(dets, img_size, frac=CENTER_FRAC)   # inti anti-duplikasi
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
    crops, boxes = gen_windows(image)
    probs_crops  = run_preds_on_crops(crops)
    num_crops    = probs_crops.shape[0]

    # 2) Votes dinamis (lebih ketat: ≥ max(3, 10% jumlah crop))
    MIN_VOTES_dyn = max(3, int(math.ceil(0.10 * max(1, num_crops))))

    # 3) NMS + suppress + cluster
    final_dets = postprocess_with_nms(boxes, probs_crops, image.size, MIN_VOTES_dyn, iou_nms=IOU_NMS)

    # 4) Tampilkan
    st.subheader("🔍 Label Terdeteksi:")
    if not final_dets:
        st.warning("🚫 Tidak ada label terdeteksi.")
    else:
        st.write(f"Total label: **{len(final_dets)}**")
        for det in final_dets:
            st.write(f"✅ *{det['label']}* ({det['score']:.2%})")

    # 5) Panel analitik (opsional)
    with st.expander("📊 Detail Probabilitas & Konsensus"):
        probs_max = probs_crops.max(0) if num_crops else np.zeros(len(LABELS))
        mean_prob = float(np.mean(probs_max)) if num_crops else 0.0
        entropy   = -float(np.mean([p * math.log(p + 1e-8) for p in probs_max])) if num_crops else 0.0
        st.write(f"🪟 crops: {num_crops} | min_votes: {MIN_VOTES_dyn} | mean_prob: {mean_prob:.3f} | entropy: {entropy:.3f}")
        for i, lbl in enumerate(LABELS):
            pass_thr = "✓" if probs_max[i] >= THRESHOLDS[i] else "✗"
            st.write(f"{lbl}: {probs_max[i]:.2%} (thr {THRESHOLDS[i]:.2f}) {pass_thr}")
