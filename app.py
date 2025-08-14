# ===========================
# 0) IMPORT & KONFIGURASI
# ===========================
import os                                   # utilitas file path
import math                                 # fungsi matematika
import traceback                            # untuk menampilkan stack trace
import numpy as np                          # operasi array & vektor
import torch                                # PyTorch core
import torch.nn as nn                       # modul jaringan saraf
import streamlit as st                      # UI web
import gdown                                # unduh dari Google Drive
from PIL import Image                       # muat gambar
from safetensors.torch import safe_open     # loader bobot .safetensors
import torchvision.transforms as transforms # transformasi gambar

# ---------------------------
# Konfigurasi umum
# ---------------------------
MODEL_URL  = 'https://drive.google.com/uc?id=1GPPxPSpodNGJHSeWyrTVwRoSjwW3HaA8'  # url model
MODEL_PATH = 'model_3.safetensors'                                               # nama file model lokal

LABELS = [                                  # urutan label HARUS sama dgn training
    'alpukat_matang', 'alpukat_mentah',
    'belimbing_matang', 'belimbing_mentah',
    'mangga_matang', 'mangga_mentah'
]
THRESHOLDS = [0.01, 0.01, 0.01, 0.06, 0.02, 0.01]  # ambang per-kelas hasil tuning
SHOW_VOTES = False                                  # tampilkan votes atau tidak

# Hyperparameter arsitektur (sesuai retrain-5)
NUM_HEADS, NUM_LAYERS, HIDDEN_DIM = 10, 4, 640
PATCH_SIZE, IMAGE_SIZE            = 14, 210

# ---------------------------
# Sliding-window & NMS
# ---------------------------
WIN_FRACS     = (1.0, 0.7, 0.5, 0.4)   # skala jendela relatif sisi terpendek
STRIDE_FRAC   = 0.33                   # overlap (semakin kecil semakin banyak crop)
MAX_CROPS     = 1200                   # BATAS KERAS jumlah crop agar tidak OOM
IOU_NMS       = 0.30                   # IoU NMS per-kelas
IOU_PAIR      = 0.30                   # IoU untuk menggabung matang vs mentah
IOU_XFAM      = 0.50                   # supresi antar keluarga buah
DEDUP_SAMECLS = 0.20                   # dedupe sisa di kelas sama
SINGLE_OBJ_IOU      = 0.40             # definisi “satu lokasi” (untuk gate)
SINGLE_OBJ_MIN_FRAC = 0.80             # ≥80% box numpuk → anggap 1 objek

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # pilih GPU kalau ada
torch.set_grad_enabled(False)                             # pastikan eval tanpa grad
try: torch.set_num_threads(1)                             # hemat CPU thread (opsional)
except Exception: pass

# ===========================
# 1) PENGUNDUHAN MODEL
# ===========================
def download_model():                                    # fungsi unduh model
    with st.spinner('🔄 Mengunduh model...'):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

# cek eksistensi/ukuran file (hindari file korup)
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 50000:
    if os.path.exists(MODEL_PATH):
        st.warning("📦 File model korup. Mengunduh ulang...")
        os.remove(MODEL_PATH)
    download_model()

# ===========================
# 2) DEFINISI MODEL
# ===========================
class PatchEmbedding(nn.Module):                         # ubah gambar → token patch
    def __init__(self, img_size, patch_size, emb_size):
        super().__init__()
        self.proj = nn.Conv2d(3, emb_size, patch_size, patch_size)  # conv stride=patch
    def forward(self, x):
        x = self.proj(x)                                             # [B,E,H/ps,W/ps]
        return x.flatten(2).transpose(1, 2)                          # [B,N,E]

class WordEmbedding(nn.Module):                                      # dummy (pakai nol saat inferensi)
    def __init__(self, dim): super().__init__()
    def forward(self, x): return x

class FeatureFusion(nn.Module):                                      # gabung visual+“teks”
    def forward(self, v, t): return torch.cat([v, t[:, :v.size(1), :]], -1)

class ScaleTransformation(nn.Module):                                # proyeksi 2E→E
    def __init__(self, in_dim, out_dim):
        super().__init__(); self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.linear(x)

class ChannelUnification(nn.Module):                                 # LayerNorm token
    def __init__(self, dim):
        super().__init__(); self.norm = nn.LayerNorm(dim)
    def forward(self, x): return self.norm(x)

class InteractionBlock(nn.Module):                                   # self-attention
    def __init__(self, dim, num_heads=NUM_HEADS):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
    def forward(self, x): return self.attn(x, x, x)[0]

class CrossScaleAggregation(nn.Module):                              # agregasi token
    def forward(self, x): return x.mean(1, keepdim=True)             # [B,1,E]

class HamburgerHead(nn.Module):                                      # head linear
    def __init__(self, in_dim, out_dim):
        super().__init__(); self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.linear(x)

class MLPClassifier(nn.Module):                                      # klasifier multi-label
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, num_classes))
    def forward(self, x): return self.mlp(x)

class HSVLTModel(nn.Module):                                         # model lengkap
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
        x = self.concat(self.patch_embed(image), self.word_embed(text))  # gabung
        x = self.scale_transform(x)                                       # 2E→E
        x = self.channel_unification(x)                                   # LN
        x = self.interaction_blocks(x)                                    # self-attn stack
        x = self.csa(x)                                                   # [B,1,E]
        x = self.head(x).mean(1)                                          # [B,E]
        return self.classifier(x)                                         # logits [B,C]

# ===========================
# 3) LOAD MODEL
# ===========================
try:
    with safe_open(MODEL_PATH, framework="pt", device=device) as f:  # buka safetensors
        state_dict = {k: f.get_tensor(k) for k in f.keys()}          # ambil semua tensor
    model = HSVLTModel(PATCH_SIZE, HIDDEN_DIM, len(LABELS), NUM_HEADS, NUM_LAYERS).to(device)
    model.load_state_dict(state_dict)                                # load bobot
    model.eval()                                                     # mode evaluasi
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.code(traceback.format_exc())
    st.stop()

# ===========================
# 4) TRANSFORMASI GAMBAR
# ===========================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),                     # resize ke 210x210
    transforms.ToTensor(),                                           # [0..1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],                 # normalisasi (ImageNet)
                         std =[0.229, 0.224, 0.225])
])

# ===========================
# 5) FUNGSI PENDUKUNG
# ===========================
def iou_xyxy(a, b):
    """Hitung IoU antara dua box xyxy (aman)."""
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter  = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2]-a[0])) * max(0.0, float(a[3]-a[1]))
    area_b = max(0.0, float(b[2]-b[0])) * max(0.0, float(b[3]-b[1]))
    denom  = area_a + area_b - inter + 1e-8
    return (inter / denom) if denom > 0 else 0.0

def estimate_total_crops(W, H, short, stride_frac):
    """Estimasi jumlah crop untuk grid saat ini (tanpa membuat crop)."""
    total = 0
    for wf in WIN_FRACS:
        win  = max(int(short * wf), 64)
        step = max(1, int(win * stride_frac))
        nx   = 1 + max(0, (W - win) // step)   # jumlah posisi horizontal
        ny   = 1 + max(0, (H - win) // step)   # jumlah posisi vertikal
        total += int(nx * ny)
    return total

def gen_boxes(image_pil):
    """Hanya menghasilkan KOORDINAT box (tanpa menyimpan crop PIL)."""
    W, H = image_pil.size
    short = min(W, H)

    # hitung estimasi jumlah crop dengan stride awal
    est = estimate_total_crops(W, H, short, STRIDE_FRAC)

    # jika estimasi > MAX_CROPS, perbesar stride secara adaptif
    stride_scale = 1.0
    if est > MAX_CROPS:
        stride_scale = math.sqrt(est / MAX_CROPS)  # kira2 mengurangi grid di 2D
    boxes = []
    for wf in WIN_FRACS:
        win  = max(int(short * wf), 64)
        step = max(1, int(win * STRIDE_FRAC * stride_scale))
        for top in range(0, max(H - win + 1, 1), step):
            for left in range(0, max(W - win + 1, 1), step):
                boxes.append((left, top, left + win, top + win))
    return np.array(boxes, dtype=np.float32)  # (N,4)

def run_preds_on_boxes(image_pil, boxes):
    """Crop on-the-fly per box → prediksi → simpan probabilitas (hemat memori)."""
    num_tokens = (IMAGE_SIZE // PATCH_SIZE) ** 2              # 225 token
    probs = np.zeros((len(boxes), len(LABELS)), dtype=np.float32)  # pre-alloc float32
    with torch.no_grad():
        for i, (l, t, r, b) in enumerate(boxes):
            crop = image_pil.crop((int(l), int(t), int(r), int(b)))  # crop saat itu juga
            x = transform(crop).unsqueeze(0).to(device)              # transform+batch=1
            dummy_text = torch.zeros((1, num_tokens, HIDDEN_DIM), device=device)
            p = torch.sigmoid(model(x, dummy_text)).cpu().numpy()[0].astype(np.float32)
            probs[i] = p
    return probs

def nms_per_class(boxes, scores, thr):
    """Non-Maximum Suppression per kelas (aman utk 0/1 kandidat)."""
    if len(boxes) == 0:
        return []
    if len(boxes) == 1:
        return [0]
    idxs = np.argsort(-scores)                                   # urut skor desc
    keep = []
    while idxs.size:
        i = int(idxs[0])                                         # ambil skor tertinggi
        keep.append(i)                                           # simpan
        rest_mask = np.asarray([iou_xyxy(boxes[i], boxes[j]) <= thr for j in idxs[1:]], dtype=bool)
        idxs = idxs[1:][rest_mask]                               # buang overlap besar
    return keep

def dedupe_same_class(detections, iou_thr=DEDUP_SAMECLS):
    """Buang duplikat sisa di kelas sama; sisakan box yang saling cukup jauh."""
    out = []
    for c in range(len(LABELS)):
        group = [d for d in detections if d["c"] == c]
        if not group:
            continue
        group.sort(key=lambda d: d["score"], reverse=True)
        kept = []
        for d in group:
            if all(iou_xyxy(d["box"], k["box"]) < iou_thr for k in kept):
                kept.append(d)
        out.extend(kept)
    return out

def merge_pairs(detections):
    """Gabungkan pasangan matang vs mentah per keluarga buah."""
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
                if j in used_B: 
                    continue
                iou = iou_xyxy(da["box"], db["box"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j != -1 and best_iou >= IOU_PAIR:
                chosen = da if da["score"] >= B[best_j]["score"] else B[best_j]
                final.append(chosen)
                used_B.add(best_j)
            else:
                final.append(da)
        final.extend(db for j, db in enumerate(B) if j not in used_B)
    for d in final:
        d["fam"] = d["label"].split('_')[0]                      # anotasi keluarga
    return final

def cross_family_suppress(dets, thr=IOU_XFAM):
    """Supresi antar keluarga buah yang saling overlap besar."""
    dets = sorted(dets, key=lambda d: d["score"], reverse=True)
    kept = []
    for d in dets:
        ok = True
        for k in kept:
            if d["fam"] != k["fam"] and iou_xyxy(d["box"], k["box"]) >= thr:
                ok = False
                break
        if ok:
            kept.append(d)
    return kept

def is_single_object(boxes):
    """Cek apakah mayoritas box menumpuk di satu lokasi (indikasi foto 1 objek)."""
    if len(boxes) == 0:
        return False
    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    base  = boxes[int(np.argmax(areas))]                          # pilih box terluas sbg anchor
    frac  = np.mean([iou_xyxy(base, b) >= SINGLE_OBJ_IOU for b in boxes])
    return bool(frac >= SINGLE_OBJ_MIN_FRAC)

def postprocess_with_nms(boxes, probs_crops, votes, min_votes, iou_nms=IOU_NMS):
    """Pipeline: filter → NMS → dedupe → pair → cross-family → (opsional) single-object gate."""
    detections = []
    thr_arr = np.asarray(THRESHOLDS, dtype=np.float32)            # pastikan dtype konsisten
    for c, label in enumerate(LABELS):
        scores = probs_crops[:, c]
        mask   = (scores >= thr_arr[c])
        if (not np.any(mask)) or votes[c] < min_votes:
            continue
        b_cls = boxes[mask]
        s_cls = scores[mask]
        keep  = nms_per_class(b_cls, s_cls, iou_nms)
        for k in keep:
            detections.append({"label": label, "c": c, "score": float(s_cls[k]), "box": b_cls[k]})

    detections = dedupe_same_class(detections, DEDUP_SAMECLS)     # dedupe sisa di kelas sama
    dets = merge_pairs(detections)                                 # pilih matang/mentah terbaik
    dets = cross_family_suppress(dets, thr=IOU_XFAM)               # tekan antar keluarga

    # Jika grid menunjukkan satu objek besar → tampilkan SATU label skor tertinggi
    if is_single_object([d["box"] for d in dets]):
        dets = [max(dets, key=lambda d: d["score"])]

    return sorted(dets, key=lambda d: d["score"], reverse=True)

# ===========================
# 6) UI STREAMLIT
# ===========================
st.title("🍉 Klasifikasi Multi-Label Buah")
uploaded_file = st.file_uploader("Unggah gambar buah", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert('RGB')                 # baca RGB
        st.image(image, caption="Gambar Input", use_container_width=True)

        # === Sliding-window ringan: hanya BOX, tidak menyimpan PIL crop
        boxes       = gen_boxes(image)                                   # (N,4)
        if boxes.shape[0] == 0:
            st.warning("Tidak ada window yang dihasilkan dari gambar ini.")
            st.stop()

        probs_crops = run_preds_on_boxes(image, boxes)                   # (N,C) float32
        num_crops   = int(probs_crops.shape[0])                          # jumlah crop
        probs_max   = probs_crops.max(0)                                 # info agregasi
        thr_arr     = np.asarray(THRESHOLDS, dtype=np.float32)           # threshold array
        votes       = (probs_crops >= thr_arr).sum(0).astype(int)        # votes per-kelas

        # Votes dinamis: minimal 5% dari jumlah crop, tapi ≥3 (biar stabil)
        MIN_VOTES_dyn = max(3, int(math.ceil(0.05 * num_crops)))

        # === Post-process (NMS + suppression)
        final_dets = postprocess_with_nms(boxes, probs_crops, votes, MIN_VOTES_dyn, iou_nms=IOU_NMS)

        # === Tampilkan hasil
        st.subheader("🔍 Label Terdeteksi:")
        if not final_dets:
            st.warning("🚫 Tidak ada label terdeteksi.")
        else:
            st.write(f"Total label terdeteksi: **{len(final_dets)}**")
            for det in final_dets:
                line = f"✅ *{det['label']}* ({det['score']:.2%})"
                if SHOW_VOTES:
                    line += f" | votes={int(votes[det['c']])}"
                st.write(line)

        # === Panel detail (diagnostik ringan)
        with st.expander("📊 Detail Probabilitas & Konsensus"):
            mean_prob = float(np.mean(probs_max))
            entropy   = -float(np.mean([p * math.log(p + 1e-8) for p in probs_max]))
            st.write(f"🪟 crops: {num_crops} (maks {MAX_CROPS}) | min_votes: {MIN_VOTES_dyn} | mean_prob: {mean_prob:.3f} | entropy: {entropy:.3f}")
            for i, lbl in enumerate(LABELS):
                pass_thr = "✓" if probs_max[i] >= THRESHOLDS[i] else "✗"
                line = f"{lbl}: {probs_max[i]:.2%} (thr {THRESHOLDS[i]:.2f}) {pass_thr}"
                if SHOW_VOTES:
                    line += f" | votes={int(votes[i])}"
                st.write(line)

    except Exception as e:
        st.error("Terjadi error saat menjalankan inferensi.")
        st.exception(e)  # tampilkan traceback lengkap agar tidak muncul “Oh no”.
