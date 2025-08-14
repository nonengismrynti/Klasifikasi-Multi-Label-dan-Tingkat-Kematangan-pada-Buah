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
import numpy as np                          # Numerik (untuk NMS & agregasi)

# ========================
# 1) Setup umum & opsi UI
# ========================
MODEL_URL  = 'https://drive.google.com/uc?id=1GPPxPSpodNGJHSeWyrTVwRoSjwW3HaA8'  # Link model di Drive
MODEL_PATH = 'model_3.safetensors'                                               # Nama file lokal model

LABELS = [                                   # Urutan label HARUS sama seperti saat training
    'alpukat_matang', 'alpukat_mentah',
    'belimbing_matang', 'belimbing_mentah',
    'mangga_matang', 'mangga_mentah'
]

SHOW_VOTES = False                           # Tampilkan jumlah crop pendukung (opsional)

# ======================================================
# 2) Parameter model + inferensi (mengikuti retrain-5)
# ======================================================
NUM_HEADS   = 10                             # Jumlah attention heads
NUM_LAYERS  = 4                              # Banyak InteractionBlock
HIDDEN_DIM  = 640                            # Dimensi embedding
PATCH_SIZE  = 14                             # Ukuran patch conv
IMAGE_SIZE  = 210                            # Ukuran input ke model
THRESHOLDS  = [0.01, 0.01, 0.01, 0.06, 0.02, 0.01]  # Ambang per kelas (dari validasi)

# ----------------------------- Sliding Window & NMS -----------------------------
WIN_FRACS   = (1.0, 0.7, 0.5, 0.4)           # Skala jendela relatif terhadap sisi terpendek
STRIDE_FRAC = 0.33                           # Overlap ~67% (stride = 0.33 * window)
IOU_NMS     = 0.50                           # Ambang IoU untuk NMS per-kelas
IOU_PAIR    = 0.30                           # Ambang IoU untuk “duel” matang vs mentah (sama objek)
MIN_VOTES   = 1                              # Minimal jumlah crop pendukung (set 1 biar tidak terlalu ketat)

# ==========================================
# 3) Download model bila belum ada/terdeteksi korup
# ==========================================
def download_model():
    """Unduh ulang model jika file tidak ada / korup."""
    with st.spinner('🔄 Mengunduh ulang model dari Google Drive...'):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

# Cek eksistensi/ukuran file model (antisipasi korup)
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 50000:
    if os.path.exists(MODEL_PATH):
        st.warning("📦 Ukuran file model terlalu kecil, kemungkinan korup. Mengunduh ulang...")
        os.remove(MODEL_PATH)
    download_model()

# ======================================
# 4) Komponen model (identik dgn training)
# ======================================
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

# ==================
# 5) Load model
# ==================
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

# ===========================
# 6) Transformasi gambar
# ===========================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize ke 210x210
    transforms.ToTensor(),                        # Ke tensor [0..1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

# ======================================================
# 7) Sliding Window + Prediksi + NMS (inti perubahan)
# ======================================================

def gen_windows(image_pil, win_fracs=WIN_FRACS, stride_frac=STRIDE_FRAC):
    """Buat list crop persegi dan koordinat box-nya (x1,y1,x2,y2)."""
    W, H = image_pil.size
    short = min(W, H)
    boxes, crops = [], []
    for wf in win_fracs:                                  # Beberapa skala jendela
        win  = max(int(short * wf), 64)                   # Minimal 64 px supaya tidak terlalu kecil
        step = max(1, int(win * stride_frac))             # Langkah (semakin kecil => overlap besar)
        for top in range(0, max(H - win + 1, 1), step):
            for left in range(0, max(W - win + 1, 1), step):
                box = (left, top, left + win, top + win)  # (x1,y1,x2,y2)
                boxes.append(box)                         # Simpan koordinat
                crops.append(image_pil.crop(box))         # Simpan crop
    return crops, np.array(boxes, dtype=np.float32)       # Kembalikan crops & boxes

def run_preds_on_crops(crops):
    """Prediksi probabilitas per-crop. Return array [N,C]."""
    num_tokens = (IMAGE_SIZE // PATCH_SIZE) ** 2          # 225 token (15x15)
    probs_list = []
    with torch.no_grad():
        for crop in crops:
            x = transform(crop).unsqueeze(0).to(device)   # Transform & batch=1
            dummy_text = torch.zeros((1, num_tokens, HIDDEN_DIM), device=device)
            logits = model(x, dummy_text)                 # [1,C]
            p = torch.sigmoid(logits).cpu().numpy()[0]    # [C]
            probs_list.append(p)
    return np.stack(probs_list, axis=0)                   # [N,C]

def iou_xyxy(a, b):
    """Hitung IoU antara box a dan b (format xyxy)."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = area_a + area_b - inter + 1e-8
    return inter / union

def nms_per_class(boxes, scores, iou_thr=IOU_NMS):
    """NMS sederhana: urutkan skor turun, buang box yang IoU-nya tinggi."""
    idxs = np.argsort(-scores)              # Indeks dari skor tertinggi ke terendah
    keep = []
    while idxs.size > 0:
        i = idxs[0]; keep.append(i)         # Ambil skor tertinggi
        if idxs.size == 1: break
        rest = idxs[1:]
        ious = np.array([iou_xyxy(boxes[i], boxes[j]) for j in rest])
        idxs = rest[ious <= iou_thr]        # Buang yang overlap besar
    return keep

def postprocess_with_nms(boxes, probs_crops):
    """
    1) Ambil kandidat per kelas berdasarkan threshold.
    2) NMS per kelas untuk hilangkan duplikasi crop di objek yang sama.
    3) Untuk pasangan (matang,mentah), jika box tumpang tindih cukup besar → pilih satu yang skornya lebih tinggi.
    """
    detections = []  # List final deteksi sementara per kelas

    # ---- NMS per kelas ----
    for c, label in enumerate(LABELS):
        scores = probs_crops[:, c]
        mask   = scores >= THRESHOLDS[c]                 # Hanya skor di atas ambang
        if mask.sum() == 0: 
            continue
        b = boxes[mask]
        s = scores[mask]
        keep = nms_per_class(b, s, iou_thr=IOU_NMS)      # Indeks yang dipertahankan
        for k in keep:
            detections.append({
                "label": label, "c": c, 
                "score": float(s[k]), "box": b[k]
            })

    # ---- Gabungkan matang vs mentah (1 buah → 1 label) ----
    pairs = [
        ('alpukat_matang', 'alpukat_mentah'),
        ('belimbing_matang', 'belimbing_mentah'),
        ('mangga_matang', 'mangga_mentah')
    ]

    final = []
    for a, b in pairs:
        # Ambil deteksi yang berkaitan dengan satu jenis buah
        A = [d for d in detections if d["label"] == a]
        B = [d for d in detections if d["label"] == b]

        used_B = set()
        # Cocokkan A vs B berdasarkan IoU
        for i, da in enumerate(A):
            best_j = -1; best_iou = 0.0
            for j, db in enumerate(B):
                if j in used_B: 
                    continue
                iou = iou_xyxy(da["box"], db["box"])
                if iou > best_iou:
                    best_iou = iou; best_j = j
            if best_iou >= IOU_PAIR:
                # Dua label di lokasi sama → pilih skor tertinggi
                chosen = da if da["score"] >= B[best_j]["score"] else B[best_j]
                final.append(chosen)
                used_B.add(best_j)  # tandai sudah dipakai
            else:
                final.append(da)    # Tidak ada pasangan dekat → pakai A

        # Tambahkan sisa B yang belum dipasangkan
        for j, db in enumerate(B):
            if j not in used_B:
                final.append(db)

    # Urutkan hasil akhir berdasarkan skor
    final.sort(key=lambda d: d["score"], reverse=True)
    return final

# ==========
# 8) UI App
# ==========
st.title("🍉 Klasifikasi Multi-Label Buah")
st.write("Upload gambar buah; sistem akan mendeteksi beberapa label sekaligus (sliding_window + NMS).")

uploaded_file = st.file_uploader("Unggah gambar buah", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar Input", use_container_width=True)

    # (1) Buat jendela sliding window & prediksi per-crop
    crops, boxes = gen_windows(image, WIN_FRACS, STRIDE_FRAC)       # crops: list PIL, boxes: [N,4]
    probs_crops  = run_preds_on_crops(crops)                        # [N,C]

    # (2) Votes per kelas (opsional, info)
    votes = (probs_crops >= np.array(THRESHOLDS)).sum(axis=0).astype(int)

    # (3) NMS + penggabungan matang/mentah
    final_dets = postprocess_with_nms(boxes, probs_crops)           # list dict {label, score, box}

    # (4) Tampilkan
    st.subheader("🔍 Label Terdeteksi:")
    if not final_dets:
        st.warning("🚫 Tidak ada label yang memenuhi kriteria.")
    else:
        st.write(f"Total label terdeteksi: **{len(final_dets)}**")
        for det in final_dets:
            if SHOW_VOTES:
                st.write(f"✅ *{det['label']}* ({det['score']:.2%})")
            else:
                st.write(f"✅ *{det['label']}* ({det['score']:.2%})")

    # (5) Panel detail (opsional)
    with st.expander("📊 Detail Probabilitas & Konsensus"):
        # Ambil agregasi maksimum per kelas (hanya untuk ringkasan)
        probs_max = probs_crops.max(axis=0)
        mean_prob = float(np.mean(probs_max))
        entropy   = -float(np.mean([p * math.log(p + 1e-8) for p in probs_max]))
        st.write(f"🪟 crops: {probs_crops.shape[0]} | mean_prob: {mean_prob:.3f} | entropy: {entropy:.3f}")
        for i, lbl in enumerate(LABELS):
            pass_thr = "✓" if probs_max[i] >= THRESHOLDS[i] else "✗"
            line = f"{lbl}: {probs_max[i]:.2%} (thr {THRESHOLDS[i]:.2f}) {pass_thr}"
            if SHOW_VOTES:
                line += f" | votes={int(votes[i])}"
            st.write(line)
