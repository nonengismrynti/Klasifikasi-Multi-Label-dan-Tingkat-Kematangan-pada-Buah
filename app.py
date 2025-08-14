
import os
import math
import traceback
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import gdown
from PIL import Image
from safetensors.torch import safe_open
import torchvision.transforms as transforms

# ========================
# Konfigurasi umum
# ========================
MODEL_URL = 'https://drive.google.com/uc?id=1GPPxPSpodNGJHSeWyrTVwRoSjwW3HaA8'
MODEL_PATH = 'model_3.safetensors'

LABELS = [
    'alpukat_matang', 'alpukat_mentah',
    'belimbing_matang', 'belimbing_mentah',
    'mangga_matang', 'mangga_mentah'
]
THRESHOLDS = [0.01, 0.01, 0.01, 0.06, 0.02, 0.01]
SHOW_VOTES = False

NUM_HEADS, NUM_LAYERS, HIDDEN_DIM = 10, 4, 640
PATCH_SIZE, IMAGE_SIZE = 14, 210
WIN_FRACS, STRIDE_FRAC = (1.0, 0.7, 0.5, 0.4), 0.33
IOU_NMS, IOU_PAIR, MIN_VOTES = 0.50, 0.30, 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ========================
# Download model jika perlu
# ========================
def download_model():
    with st.spinner('🔄 Mengunduh model...'):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 50000:
    if os.path.exists(MODEL_PATH):
        st.warning("📦 File model korup. Mengunduh ulang...")
        os.remove(MODEL_PATH)
    download_model()

# ========================
# Definisi model
# ========================
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, emb_size):
        super().__init__()
        self.proj = nn.Conv2d(3, emb_size, patch_size, patch_size)
    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)

class WordEmbedding(nn.Module):
    def __init__(self, dim): super().__init__()
    def forward(self, x): return x

class FeatureFusion(nn.Module):
    def forward(self, v, t): return torch.cat([v, t[:, :v.size(1), :]], -1)

class ScaleTransformation(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.linear(x)

class ChannelUnification(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
    def forward(self, x): return self.norm(x)

class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=NUM_HEADS):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
    def forward(self, x): return self.attn(x, x, x)[0]

class CrossScaleAggregation(nn.Module):
    def forward(self, x): return x.mean(1, keepdim=True)

class HamburgerHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
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
        self.word_embed = WordEmbedding(emb_size)
        self.concat = FeatureFusion()
        self.scale_transform = ScaleTransformation(emb_size * 2, emb_size)
        self.channel_unification = ChannelUnification(emb_size)
        self.interaction_blocks = nn.Sequential(*[InteractionBlock(emb_size, num_heads) for _ in range(num_layers)])
        self.csa = CrossScaleAggregation()
        self.head = HamburgerHead(emb_size, emb_size)
        self.classifier = MLPClassifier(emb_size, num_classes)
    def forward(self, image, text):
        x = self.concat(self.patch_embed(image), self.word_embed(text))
        x = self.scale_transform(x)
        x = self.channel_unification(x)
        x = self.interaction_blocks(x)
        x = self.csa(x)
        x = self.head(x).mean(1)
        return self.classifier(x)

# ========================
# Load model
# ========================
try:
    with safe_open(MODEL_PATH, framework="pt", device=device) as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    model = HSVLTModel(PATCH_SIZE, HIDDEN_DIM, len(LABELS), NUM_HEADS, NUM_LAYERS).to(device)
    model.load_state_dict(state_dict)
    model.eval()
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.code(traceback.format_exc())
    st.stop()

# ========================
# Transformasi gambar
# ========================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ========================
# Fungsi pendukung
# ========================
def gen_windows(image_pil):
    W, H = image_pil.size
    short = min(W, H)
    boxes, crops = [], []
    for wf in WIN_FRACS:
        win = max(int(short * wf), 64)
        step = max(1, int(win * STRIDE_FRAC))
        for top in range(0, max(H - win + 1, 1), step):
            for left in range(0, max(W - win + 1, 1), step):
                box = (left, top, left + win, top + win)
                boxes.append(box)
                crops.append(image_pil.crop(box))
    return crops, np.array(boxes, np.float32)

def run_preds_on_crops(crops):
    num_tokens = (IMAGE_SIZE // PATCH_SIZE) ** 2
    probs = []
    with torch.no_grad():
        for crop in crops:
            x = transform(crop).unsqueeze(0).to(device)
            dummy_text = torch.zeros((1, num_tokens, HIDDEN_DIM), device=device)
            p = torch.sigmoid(model(x, dummy_text)).cpu().numpy()[0]
            probs.append(p)
    return np.stack(probs)

def iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-8)

def nms_per_class(boxes, scores, thr):
    idxs = np.argsort(-scores)
    keep = []
    while idxs.size:
        i = idxs[0]
        keep.append(i)
        idxs = idxs[1:][[iou_xyxy(boxes[i], boxes[j]) <= thr for j in idxs[1:]]]
    return keep

def postprocess_with_nms(boxes, probs_crops):
    detections = []
    for c, label in enumerate(LABELS):
        scores = probs_crops[:, c]
        mask = scores >= THRESHOLDS[c]
        if not mask.any():
            continue
        keep = nms_per_class(boxes[mask], scores[mask], IOU_NMS)
        for k in keep:
            detections.append({"label": label, "c": c, "score": float(scores[mask][k]), "box": boxes[mask][k]})
    final = merge_pairs(detections)
    return sorted(final, key=lambda d: d["score"], reverse=True)

def merge_pairs(detections):
    pairs = [('alpukat_matang', 'alpukat_mentah'),
             ('belimbing_matang', 'belimbing_mentah'),
             ('mangga_matang', 'mangga_mentah')]
    final = []
    for a, b in pairs:
        A = [d for d in detections if d["label"] == a]
        B = [d for d in detections if d["label"] == b]
        used_B = set()
        for da in A:
            best_j, best_iou = -1, 0
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

# ========================
# UI
# ========================
st.title("🍉 Klasifikasi Multi-Label Buah")
uploaded_file = st.file_uploader("Unggah gambar buah", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar Input", use_container_width=True)
    crops, boxes = gen_windows(image)
    probs_crops = run_preds_on_crops(crops)
    votes = (probs_crops >= np.array(THRESHOLDS)).sum(0).astype(int)
    final_dets = postprocess_with_nms(boxes, probs_crops)

    st.subheader("🔍 Label Terdeteksi:")
    if not final_dets:
        st.warning("🚫 Tidak ada label terdeteksi.")
    else:
        st.write(f"Total label: **{len(final_dets)}**")
        for det in final_dets:
            line = f"✅ *{det['label']}* ({det['score']:.2%})"
            if SHOW_VOTES:
                line += f" | votes={votes[det['c']]}"
            st.write(line)

    with st.expander("📊 Detail Probabilitas & Konsensus"):
        probs_max = probs_crops.max(0)
        mean_prob = float(np.mean(probs_max))
        entropy = -float(np.mean([p * math.log(p + 1e-8) for p in probs_max]))
        st.write(f"🪟 crops: {probs_crops.shape[0]} | mean_prob: {mean_prob:.3f} | entropy: {entropy:.3f}")
        for i, lbl in enumerate(LABELS):
            pass_thr = "✓" if probs_max[i] >= THRESHOLDS[i] else "✗"
            line = f"{lbl}: {probs_max[i]:.2%} (thr {THRESHOLDS[i]:.2f}) {pass_thr}"
            if SHOW_VOTES:
                line += f" | votes={votes[i]}"
            st.write(line)