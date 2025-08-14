import streamlit as st                      # UI web
import torch                                # PyTorch core
import torch.nn as nn                       # Modul neural network
from safetensors.torch import safe_open     # Loader bobot .safetensors
import torchvision.transforms as transforms # Transformasi gambar
from PIL import Image                       # Baca gambar
import gdown                                # Unduh dari Google Drive
import os, math, traceback
import numpy as np

# ========================
# 1) Setup umum & opsi UI
# ========================
MODEL_URL  = 'https://drive.google.com/uc?id=1GPPxPSpodNGJHSeWyrTVwRoSjwW3HaA8'
MODEL_PATH = 'model_3.safetensors'

LABELS = [
    'alpukat_matang', 'alpukat_mentah',
    'belimbing_matang', 'belimbing_mentah',
    'mangga_matang', 'mangga_mentah'
]

SHOW_VOTES = False

# ======================================================
# 2) Parameter model + inferensi (ikut retrain-5)
# ======================================================
NUM_HEADS   = 10
NUM_LAYERS  = 4
HIDDEN_DIM  = 640
PATCH_SIZE  = 14
IMAGE_SIZE  = 210
THRESHOLDS  = [0.01, 0.01, 0.01, 0.06, 0.02, 0.01]

# Sliding-window
WIN_FRACS   = (1.0, 0.7, 0.5, 0.4)
STRIDE_FRAC = 0.33

# NMS & merging
IOU_NMS  = 0.50    # NMS per-kelas (agak agresif biar nggak duplikat)
IOU_PAIR = 0.30    # Matang vs mentah → satu buah bila IoU >= 0.3

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===========================
# 3) Download model jika perlu
# ===========================
def download_model():
    with st.spinner('🔄 Mengunduh model dari Google Drive...'):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 50_000:
    if os.path.exists(MODEL_PATH):
        st.warning("📦 File model kecil/korup. Mengunduh ulang...")
        os.remove(MODEL_PATH)
    download_model()

# ===========================
# 4) Definisi model (identik training)
# ===========================
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, emb_size):
        super().__init__()
        self.proj = nn.Conv2d(3, emb_size, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)                 # [B,E,H/ps,W/ps]
        return x.flatten(2).transpose(1, 2)

class WordEmbedding(nn.Module):
    def __init__(self, dim): super().__init__()
    def forward(self, x): return x

class FeatureFusion(nn.Module):
    def forward(self, v, t): return torch.cat([v, t[:, :v.size(1), :]], dim=-1)

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
        super().__init__(); self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
    def forward(self, x): return self.attn(x, x, x)[0]

class CrossScaleAggregation(nn.Module):
    def forward(self, x): return x.mean(dim=1, keepdim=True)

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
        v = self.patch_embed(image)
        t = self.word_embed(text)
        x = self.concat(v, t)
        x = self.scale_transform(x)
        x = self.channel_unification(x)
        x = self.interaction_blocks(x)
        x = self.csa(x)
        x = self.head(x).mean(dim=1)
        return self.classifier(x)

# ==================
# 5) Load model
# ==================
if len(THRESHOLDS) != len(LABELS):
    st.error("Panjang THRESHOLDS ≠ jumlah LABELS"); st.stop()

try:
    with safe_open(MODEL_PATH, framework="pt", device=device) as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    model = HSVLTModel(patch_size=PATCH_SIZE, emb_size=HIDDEN_DIM,
                       num_classes=len(LABELS), num_heads=NUM_HEADS,
                       num_layers=NUM_LAYERS).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.code(traceback.format_exc()); st.stop()

# ===========================
# 6) Transformasi gambar
# ===========================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

# ==========================================================
# 7) Sliding-window inference (kembalikan probs & boxes)
# ==========================================================
def sliding_window_infer(image_pil, model, transform, device,
                         hidden_dim, patch_size,
                         win_fracs=WIN_FRACS, stride_frac=STRIDE_FRAC,
                         include_full=False):
    W, H = image_pil.size
    short = min(W, H)
    crops, boxes = [], []
    for wf in win_fracs:
        win  = max(int(short * wf), 64)
        step = max(1, int(win * stride_frac))
        for top in range(0, max(H - win + 1, 1), step):
            for left in range(0, max(W - win + 1, 1), step):
                boxes.append((left, top, left + win, top + win))
                crops.append(image_pil.crop((left, top, left + win, top + win)))
    if include_full:
        boxes.append((0, 0, W, H))
        crops.append(image_pil)

    num_tokens = (IMAGE_SIZE // patch_size) ** 2
    probs_list = []
    with torch.no_grad():
        for crop in crops:
            x = transform(crop).unsqueeze(0).to(device)
            dummy = torch.zeros((1, num_tokens, hidden_dim), device=device)
            p = torch.sigmoid(model(x, dummy)).cpu().numpy()[0]
            probs_list.append(p)

    probs_crops = np.stack(probs_list, axis=0)      # [N, C]
    probs_max   = probs_crops.max(axis=0)           # [C]
    boxes       = np.array(boxes, dtype=np.float32) # [N, 4]
    return probs_max, probs_crops, boxes

# ==========================================================
# 8) NMS & merging utils
# ==========================================================
def iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter  = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-8)

def nms_per_class(boxes, scores, thr=IOU_NMS):
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        rest = rest[[iou_xyxy(boxes[i], boxes[j]) <= thr for j in rest]]
        order = rest
    return keep

def merge_pairs(dets):
    pairs = [('alpukat_matang', 'alpukat_mentah'),
             ('belimbing_matang', 'belimbing_mentah'),
             ('mangga_matang',   'mangga_mentah')]
    final = []
    for a, b in pairs:
        A = [d for d in dets if d["label"] == a]
        B = [d for d in dets if d["label"] == b]
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
                final.append(chosen)
                used_B.add(best_j)
            else:
                final.append(da)
        final.extend(db for j, db in enumerate(B) if j not in used_B)
    return final

def dedupe_per_label(dets, thr=0.5):
    """NMS kedua per-label (jaga-jaga kalau masih ada duplikat kelas yg sama)."""
    out = []
    for lbl in set(d['label'] for d in dets):
        group = [d for d in dets if d['label'] == lbl]
        if not group: continue
        boxes = np.stack([g['box'] for g in group], 0)
        scores = np.array([g['score'] for g in group], dtype=np.float32)
        keep = nms_per_class(boxes, scores, thr=thr)
        out.extend([group[k] for k in keep])
    return out

def postprocess_with_nms(boxes, probs_crops, votes, min_votes):
    dets = []
    for c, label in enumerate(LABELS):
        scores = probs_crops[:, c]
        mask   = scores >= THRESHOLDS[c]
        if not mask.any() or votes[c] < min_votes:
            continue
        b = boxes[mask]; s = scores[mask]
        keep = nms_per_class(b, s, thr=IOU_NMS)
        for k in keep:
            dets.append({"label": label, "c": c, "score": float(s[k]), "box": b[k]})

    if not dets:
        return []

    dets = merge_pairs(dets)             # matang vs mentah → satu label per buah
    dets = dedupe_per_label(dets, 0.5)   # bersihkan duplikat kelas yg sama (opsional)
    dets.sort(key=lambda d: d["score"], reverse=True)
    return dets

# ==========
# 9) UI App
# ==========
st.title("🍉 Klasifikasi Multi-Label Buah")
st.write("Upload gambar; sistem akan deteksi label per buah (sliding-window + NMS).")

uploaded_file = st.file_uploader("Unggah gambar buah", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar Input", use_container_width=True)

    # (1) Sliding-window → probs per-crop & boxes
    probs_max, probs_crops, boxes = sliding_window_infer(
        image_pil=image, model=model, transform=transform, device=device,
        hidden_dim=HIDDEN_DIM, patch_size=PATCH_SIZE,
        win_fracs=WIN_FRACS, stride_frac=STRIDE_FRAC, include_full=False
    )

    # (2) Votes per-kelas: berapa crop yg ≥ threshold
    thr = np.array(THRESHOLDS, dtype=np.float32)[None, :]      # (1,C)
    votes = (probs_crops >= thr).sum(axis=0).astype(int)       # (C,)

    # (3) Minimal votes dinamis: 5% jumlah crop, min 2
    min_votes_dyn = max(2, int(0.05 * probs_crops.shape[0]))

    # (4) NMS + merge → final deteksi
    final_dets = postprocess_with_nms(boxes, probs_crops, votes, min_votes=min_votes_dyn)

    # (5) Tampilkan
    st.subheader("🔍 Label Terdeteksi:")
    if not final_dets:
        st.warning("🚫 Tidak ada label yang memenuhi kriteria.")
    else:
        st.write(f"Total label terdeteksi: **{len(final_dets)}**")
        for d in final_dets:
            txt = f"✅ *{d['label']}* ({d['score']:.2%})"
            if SHOW_VOTES:
                txt += f" | votes={int(votes[d['c']])}"
            st.write(txt)

    # (6) Panel analitik
    with st.expander("📊 Detail Probabilitas & Konsensus"):
        probs = probs_max.tolist()
        mean_prob = float(np.mean(probs))
        entropy   = -float(np.mean([p * math.log(p + 1e-8) for p in probs]))
        st.write(f"🪟 crops: {probs_crops.shape[0]} | min_votes: {min_votes_dyn} | mean_prob: {mean_prob:.3f} | entropy: {entropy:.3f}")
        for i, lbl in enumerate(LABELS):
            pass_thr = "✓" if probs[i] >= THRESHOLDS[i] else "✗"
            line = f"{lbl}: {probs[i]:.2%} (thr {THRESHOLDS[i]:.2f}) {pass_thr}"
            if SHOW_VOTES:
                line += f" | votes={int(votes[i])}"
            st.write(line)
