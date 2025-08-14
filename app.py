import streamlit as st                      # UI web
import torch                                # PyTorch core
import torch.nn as nn                       # Modul neural network
from safetensors.torch import safe_open     # Loader bobot .safetensors
import torchvision.transforms as transforms # Transformasi gambar
from PIL import Image, ImageDraw            # Baca & anotasi gambar
import gdown                                # Download dari Google Drive
import os                                   # Utilitas file/path
import math                                 # Untuk hitung entropi (informasi)
import traceback                            # Tampilkan traceback saat error
import numpy as np                          # Agregasi antar-crop (sliding window)

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

SHOW_VOTES = False                           # ← Tampilkan "votes" di UI? (False = disembunyikan)

# ======================================================
# 2) Parameter model + inferensi (mengikuti retrain-5)
# ======================================================
NUM_HEADS   = 10                             # Jumlah attention heads
NUM_LAYERS  = 4                              # Banyak InteractionBlock
HIDDEN_DIM  = 640                            # Dimensi embedding
PATCH_SIZE  = 14                             # Ukuran patch conv
IMAGE_SIZE  = 210                            # Ukuran input ke model
THRESHOLDS  = [0.01, 0.01, 0.01, 0.06, 0.02, 0.01]  # Threshold per-kelas (hasil tuning validasi)

# Sliding-window (dipasang di tahap prediksi/aplikasi sesuai arahan dosen)
WIN_FRACS   = (1.0, 0.7, 0.5, 0.4)           # Skala jendela relatif terhadap sisi terpendek
STRIDE_FRAC = 0.33                           # Overlap ~67% (stride = 0.33 * window)
MIN_VOTES   = 2                              # Minimal jumlah crop yang “setuju” agar label dihitung

# ==========================
# 2b) Parameter NMS (baru)
# ==========================
NMS_IOU            = 0.5                     # IOU threshold untuk NMS per-kelas
NMS_MAX_PER_CLASS  = 5                       # Maksimal box per kelas setelah NMS (boleh disetel)

# ==========================================
# 3) Download model bila belum ada/terdeteksi korup
# ==========================================
def download_model():                        # Fungsi unduh model dari Google Drive
    with st.spinner('🔄 Mengunduh ulang model dari Google Drive...'):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 50000:  # Cek eksistensi/ukuran minimal
    if os.path.exists(MODEL_PATH):              # Jika file ada tapi kecil → kemungkinan korup
        st.warning("📦 Ukuran file model terlalu kecil, kemungkinan korup. Mengunduh ulang...")
        os.remove(MODEL_PATH)                   # Hapus file korup
    download_model()                            # Unduh ulang

# ======================================
# 4) Komponen model (identik dgn training)
# ======================================
class PatchEmbedding(nn.Module):                # Ubah gambar → token patch
    def __init__(self, img_size, patch_size, emb_size):
        super().__init__()
        self.proj = nn.Conv2d(3, emb_size, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)                        # [B, E, H/ps, W/ps]
        x = x.flatten(2).transpose(1, 2)        # [B, N, E]
        return x

class WordEmbedding(nn.Module):                 # Dummy (kita pakai tensor nol saat inferensi)
    def __init__(self, dim): super().__init__()
    def forward(self, x): return x

class FeatureFusion(nn.Module):                 # Gabung visual + “teks” (dummy)
    def forward(self, v, t):
        return torch.cat([v, t[:, :v.size(1), :]], dim=-1)  # → [B, N, 2E]

class ScaleTransformation(nn.Module):           # Proyeksi 2E → E
    def __init__(self, in_dim, out_dim):
        super().__init__(); self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.linear(x)

class ChannelUnification(nn.Module):            # LayerNorm token
    def __init__(self, dim):
        super().__init__(); self.norm = nn.LayerNorm(dim)
    def forward(self, x): return self.norm(x)

class InteractionBlock(nn.Module):              # Self-attention block
    def __init__(self, dim, num_heads=NUM_HEADS):
        super().__init__(); self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
    def forward(self, x): return self.attn(x, x, x)[0]

class CrossScaleAggregation(nn.Module):         # Agregasi sederhana antar token
    def forward(self, x): return x.mean(dim=1, keepdim=True)  # [B, 1, E]

class HamburgerHead(nn.Module):                 # Linear head penyesuaian
    def __init__(self, in_dim, out_dim):
        super().__init__(); self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.linear(x)

class MLPClassifier(nn.Module):                 # Klasifier multi-label (logits)
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, num_classes))
    def forward(self, x): return self.mlp(x)

class HSVLTModel(nn.Module):                    # Rangkaian lengkap model
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
        image_feat = self.patch_embed(image)    # [B, N, E]
        text_feat  = self.word_embed(text)      # [B, N, E] (dummy zeros)
        x = self.concat(image_feat, text_feat)  # [B, N, 2E]
        x = self.scale_transform(x)             # [B, N, E]
        x = self.channel_unification(x)         # [B, N, E]
        x = self.interaction_blocks(x)          # [B, N, E]
        x = self.csa(x)                         # [B, 1, E]
        x = self.head(x)                        # [B, 1, E]
        x = x.mean(dim=1)                       # [B, E]
        return self.classifier(x)               # [B, C] logits

# ==================
# 5) Load model
# ==================
device = 'cuda' if torch.cuda.is_available() else 'cpu'   # Pilih device (GPU jika ada)
if len(THRESHOLDS) != len(LABELS):                         # Validasi panjang threshold
    st.error("Panjang THRESHOLDS tidak sama dengan jumlah LABELS."); st.stop()

try:
    with safe_open(MODEL_PATH, framework="pt", device=device) as f:  # Buka file .safetensors
        state_dict = {k: f.get_tensor(k) for k in f.keys()}          # Ambil semua tensor
    model = HSVLTModel(                                              # Inisialisasi arsitektur
        patch_size=PATCH_SIZE, emb_size=HIDDEN_DIM, num_classes=len(LABELS),
        num_heads=NUM_HEADS, num_layers=NUM_LAYERS
    ).to(device)
    model.load_state_dict(state_dict, strict=True)                   # Load bobot
    model.eval()                                                     # Mode evaluasi
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")                           # Pesan gagal
    st.code(traceback.format_exc()); st.stop()                       # Tampilkan traceback & stop

# ===========================
# 6) Transformasi gambar
# ===========================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),                    # Resize ke 210x210
    transforms.ToTensor(),                                          # Ke tensor [0..1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],                # Normalisasi (ImageNet)
                         std =[0.229, 0.224, 0.225])
])

# ===================================
# 7) Utilitas NMS (baru, class-wise)
# ===================================
def _iou_single(box, boxes):
    """
    box : (x1,y1,x2,y2)
    boxes : [N,4]
    """
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter + 1e-8
    return inter / union

def nms_numpy(boxes, scores, iou_thresh=0.5, max_dets=None):
    """
    NMS sederhana di NumPy. boxes: [N,4], scores: [N]
    return: index yang dipertahankan (relatif ke input)
    """
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if (max_dets is not None) and (len(keep) >= max_dets):
            break
        ious = _iou_single(boxes[i], boxes[order[1:]])
        inds = np.where(ious <= iou_thresh)[0]
        order = order[1:][inds]
    return keep

# ==========================================================
# 8) Sliding-window inference → simpan PROB & BOX per-crop
# ==========================================================
def sliding_window_infer(image_pil, model, transform, device,
                         hidden_dim, patch_size,
                         win_fracs=WIN_FRACS, stride_frac=STRIDE_FRAC,
                         include_full=False):
    """
    Potong gambar menjadi beberapa crop overlap (multi-skala),
    prediksi per-crop, dan kembalikan:
      - probs_crops: [N, C]
      - boxes: [N, 4] dalam koordinat gambar asli (x1,y1,x2,y2)
      - probs_max: agregasi MAX antar-crop per kelas (untuk monitoring)
    """
    W, H = image_pil.size
    short = min(W, H)
    crops, boxes = [], []

    for wf in win_fracs:
        win  = max(int(short * wf), 64)                            # Min window 64
        step = max(1, int(win * stride_frac))                      # Stride
        for top in range(0, max(H - win + 1, 1), step):
            for left in range(0, max(W - win + 1, 1), step):
                crops.append(image_pil.crop((left, top, left + win, top + win)))
                boxes.append((left, top, left + win, top + win))

    if include_full:
        crops.append(image_pil)
        boxes.append((0, 0, W, H))

    num_tokens = (IMAGE_SIZE // patch_size) ** 2                   # 225 token
    probs_list = []
    model.eval()
    with torch.no_grad():
        for crop in crops:
            x = transform(crop).unsqueeze(0).to(device)
            dummy_text = torch.zeros((1, num_tokens, hidden_dim), device=device)
            logits = model(x, dummy_text)                          # [1, C]
            p = torch.sigmoid(logits).cpu().numpy()[0]             # [C]
            probs_list.append(p)

    probs_crops = np.stack(probs_list, axis=0)                     # [N, C]
    probs_max   = probs_crops.max(axis=0)                          # [C]
    return probs_max, probs_crops, np.array(boxes, dtype=np.int32)

# ==================================================
# 9) Post-proses: class-wise NMS di atas crop hasil
# ==================================================
def apply_classwise_nms(probs_crops, boxes, thresholds, labels,
                        iou_thresh=NMS_IOU, max_per_class=NMS_MAX_PER_CLASS,
                        min_votes=MIN_VOTES):
    """
    - Seleksi kandidat per kelas: score >= threshold
    - Hitung votes (banyak crop >= threshold)
    - Terapkan NMS per kelas (pakai IoU)
    return:
      detections: list of dict {label, score, box}
      votes: np.array [C] (sebelum NMS)
    """
    C = len(labels)
    votes = (probs_crops >= np.array(thresholds)).sum(axis=0).astype(int)  # [C]
    detections = []

    for c in range(C):
        # Skip kelas yang sangat lemah (tidak ada crop yang lolos thr atau votes < min_votes)
        mask = probs_crops[:, c] >= thresholds[c]
        if (mask.sum() == 0) or (votes[c] < min_votes):
            continue

        boxes_c   = boxes[mask]
        scores_c  = probs_crops[mask, c]
        keep_idx  = nms_numpy(boxes_c, scores_c, iou_thresh=iou_thresh, max_dets=max_per_class)

        for j in keep_idx:
            detections.append({
                "label": labels[c],
                "score": float(scores_c[j]),
                "box": tuple(map(int, boxes_c[j]))
            })

    # Urutkan global berdasarkan skor menurun
    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections, votes

def draw_detections(image_pil, detections):
    img = image_pil.copy()
    drw = ImageDraw.Draw(img)
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        drw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=3)
        # Teks kecil di pojok kiri atas box
        txt = f"{det['label']} {det['score']:.2f}"
        # background tipis agar terbaca
        tw, th = drw.textlength(txt), 12
        drw.rectangle((x1, y1, x1 + tw + 6, y1 + th + 4), fill=(255, 255, 255))
        drw.text((x1 + 3, y1 + 2), txt, fill=(0, 0, 0))
    return img

# ==========
# 10) UI App
# ==========
st.title("🍉 Klasifikasi Multi-Label Buah.")
st.write("Upload gambar buahnya yaa.")

# Opsi NMS dari sidebar (bisa kamu ubah saat uji)
with st.sidebar:
    st.header("⚙️ Opsi Inference")
    NMS_IOU = st.slider("NMS IoU Threshold", 0.1, 0.9, NMS_IOU, 0.05)
    NMS_MAX_PER_CLASS = st.number_input("Maksimal Box per Kelas", 1, 20, NMS_MAX_PER_CLASS, 1)
    SHOW_BOXES = st.checkbox("Tampilkan bounding box hasil NMS", value=True)

uploaded_file = st.file_uploader("Unggah gambar buah", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar Input", use_container_width=True)

    # (1) Prediksi dengan Sliding-Window → probs per-crop + boxes
    probs_max, probs_crops, boxes = sliding_window_infer(
        image_pil=image, model=model, transform=transform, device=device,
        hidden_dim=HIDDEN_DIM, patch_size=PATCH_SIZE,
        win_fracs=WIN_FRACS, stride_frac=STRIDE_FRAC, include_full=False
    )

    # (2) NMS per-kelas (sesuai arahan dosen: diterapkan SETELAH sliding_window)
    detections, votes = apply_classwise_nms(
        probs_crops=probs_crops,
        boxes=boxes,
        thresholds=THRESHOLDS,
        labels=LABELS,
        iou_thresh=NMS_IOU,
        max_per_class=NMS_MAX_PER_CLASS,
        min_votes=MIN_VOTES
    )

    # (3) Ringkasan label yang terdeteksi (pasca-NMS)
    unique_labels = []
    for d in detections:
        if d["label"] not in unique_labels:
            unique_labels.append(d["label"])

    st.subheader("🔍 Label Terdeteksi (setelah NMS):")
    if not unique_labels:
        st.warning("🚫 Tidak ada label yang memenuhi kriteria (threshold + votes + NMS).")
    else:
        st.write(f"Total label: **{len(unique_labels)}**")
        for lbl in unique_labels:
            # ambil skor tertinggi dari deteksi label tsb
            best = max([d for d in detections if d["label"] == lbl], key=lambda x: x["score"])
            if SHOW_VOTES:
                idx = LABELS.index(lbl)
                st.write(f"✅ *{lbl}* ({best['score']:.2%}) — votes: {int(votes[idx])}")
            else:
                st.write(f"✅ *{lbl}* ({best['score']:.2%})")

    # # (4) Visualisasi box hasil NMS (opsional)
    # if SHOW_BOXES and len(detections) > 0:
    #     st.subheader("🖼️ Bounding Box Hasil NMS")
    #     st.image(draw_detections(image, detections), use_container_width=True)

    # (5) Panel detail (probabilitas per-kelas)
    st.subheader("📊 Probabilitas & Threshold (agregasi MAX untuk monitoring)")
    mean_prob = float(np.mean(probs_max))
    entropy   = -float(np.mean([p * math.log(p + 1e-8) for p in probs_max]))
    st.write(f"🪟 total crops: {probs_crops.shape[0]} | mean_prob: {mean_prob:.3f} | entropy: {entropy:.3f}")
    for i, lbl in enumerate(LABELS):
        pass_thr = "✓" if probs_max[i] >= THRESHOLDS[i] else "✗"
        line = f"{lbl}: {probs_max[i]:.2%} (thr {THRESHOLDS[i]:.2f}) {pass_thr}"
        if SHOW_VOTES:
            line += f" | votes={int(votes[i])}"
        st.write(line)

    # (6) Daftar semua deteksi (box) pasca-NMS
    with st.expander("📦 Semua Deteksi (setelah NMS)"):
        if len(detections) == 0:
            st.write("—")
        else:
            for d in detections:
                st.write(f"- {d['label']} | skor {d['score']:.2%} | box {d['box']}")
