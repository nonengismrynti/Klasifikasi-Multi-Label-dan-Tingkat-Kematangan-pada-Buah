# ===========================
# 0) IMPORT & KONFIGURASI
# ===========================
import os                                   # utilitas file path
import math                                 # fungsi matematika (ceil, log)
import traceback                            # untuk menampilkan error stack trace
import numpy as np                          # operasi array & vektor
import torch                                # PyTorch core
import torch.nn as nn                       # modul jaringan saraf
import streamlit as st                      # UI web
import gdown                                # unduh dari Google Drive
from PIL import Image                       # load gambar
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
THRESHOLDS = [0.01, 0.01, 0.01, 0.06, 0.02, 0.01]  # ambang per-kelas hasil tuning validasi
SHOW_VOTES = False                                  # menampilkan jumlah "votes" di UI atau tidak

# Hyperparameter arsitektur (sesuai retrain-5)
NUM_HEADS, NUM_LAYERS, HIDDEN_DIM = 10, 4, 640
PATCH_SIZE, IMAGE_SIZE = 14, 210

# ---------------------------
# Parameter sliding-window & NMS (REVISI)
# ---------------------------
WIN_FRACS, STRIDE_FRAC = (1.0, 0.7, 0.5, 0.4), 0.33     # skala jendela & overlap
IOU_NMS   = 0.30                                        # NMS lebih ketat supaya tidak banjir duplikat
IOU_PAIR  = 0.30                                        # IoU untuk merge matang vs mentah
IOU_XFAM  = 0.50                                        # IoU suppression antar "keluarga buah"
DEDUP_IOU_SAMECLASS = 0.20                              # dedupe tambahan per-kelas setelah NMS
SINGLE_OBJ_IOU      = 0.40                              # definisi overlap "1 lokasi" utk single-object
SINGLE_OBJ_MIN_FRAC = 0.80                              # ≥80% box menumpuk → anggap 1 objek

device = 'cuda' if torch.cuda.is_available() else 'cpu' # pilih GPU kalau ada

# ===========================
# 1) PENGUNDUHAN MODEL
# ===========================
def download_model():                                    # fungsi unduh model
    with st.spinner('🔄 Mengunduh model...'):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

# cek eksistensi / ukuran file (hindari file korup)
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

class FeatureFusion(nn.Module):                                      # gabung visual+teks
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

class MLPClassifier(nn.Module):                                      # klasifier multilabel
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
    with safe_open(MODEL_PATH, framework="pt", device=device) as f:  # buka file safetensors
        state_dict = {k: f.get_tensor(k) for k in f.keys()}          # ambil semua tensor
    model = HSVLTModel(PATCH_SIZE, HIDDEN_DIM, len(LABELS), NUM_HEADS, NUM_LAYERS).to(device)  # init arsitektur
    model.load_state_dict(state_dict)                                # load bobot
    model.eval()                                                     # mode evaluasi
except Exception as e:                                               # bila gagal load
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
def gen_windows(image_pil):
    """Buat daftar crop sliding-window + koordinat box-nya."""
    W, H = image_pil.size
    short = min(W, H)
    boxes, crops = [], []
    for wf in WIN_FRACS:                                             # untuk tiap skala jendela
        win  = max(int(short * wf), 64)                              # ukuran window (min 64 px)
        step = max(1, int(win * STRIDE_FRAC))                        # stride (overlap tinggi → step kecil)
        for top in range(0, max(H - win + 1, 1), step):              # geser vertikal
            for left in range(0, max(W - win + 1, 1), step):         # geser horizontal
                box = (left, top, left + win, top + win)             # (x1,y1,x2,y2)
                boxes.append(box)                                    # simpan box
                crops.append(image_pil.crop(box))                    # simpan crop persegi
    return crops, np.array(boxes, np.float32)                        # kembalikan list crop & np.array boxes

def run_preds_on_crops(crops):
    """Jalankan prediksi model pada setiap crop (tanpa grad)."""
    num_tokens = (IMAGE_SIZE // PATCH_SIZE) ** 2                     # 225 token (15x15)
    probs = []                                                       # daftar probabilitas per crop
    with torch.no_grad():
        for crop in crops:
            x = transform(crop).unsqueeze(0).to(device)              # transform + batch=1
            dummy_text = torch.zeros((1, num_tokens, HIDDEN_DIM), device=device)  # dummy text zeros
            p = torch.sigmoid(model(x, dummy_text)).cpu().numpy()[0]             # sigmoid → probabilitas [C]
            probs.append(p)                                          # simpan
    return np.stack(probs)                                           # [N, C]

def iou_xyxy(a, b):
    """Hitung IoU antara dua box xyxy."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter  = max(0, x2 - x1) * max(0, y2 - y1)                       # luas irisan
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])                   # luas A
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])                   # luas B
    return inter / (area_a + area_b - inter + 1e-8)                  # IoU aman (hindari div 0)

def nms_per_class(boxes, scores, thr):
    """Non-Maximum Suppression per kelas."""
    idxs = np.argsort(-scores)                                       # urut skor desc
    keep = []                                                        # indeks terpilih
    while idxs.size:
        i = idxs[0]                                                  # ambil skor tertinggi
        keep.append(i)                                               # simpan
        # sisakan hanya box dengan IoU <= thr (yang overlap besar dibuang)
        idxs = idxs[1:][[iou_xyxy(boxes[i], boxes[j]) <= thr for j in idxs[1:]]]
    return keep

# ---------- DEDUPE & SINGLE-OBJECT (REVISI) ----------
def dedupe_same_class(detections, iou_thr=DEDUP_IOU_SAMECLASS):
    """Buang duplikat sisa di kelas yang sama; sisakan box yang saling cukup jauh."""
    out = []
    for c in range(len(LABELS)):
        group = [d for d in detections if d["c"] == c]              # ambil satu kelas
        group.sort(key=lambda d: d["score"], reverse=True)          # urut skor desc
        kept = []
        for d in group:
            if all(iou_xyxy(d["box"], k["box"]) < iou_thr for k in kept):  # cukup jauh dari yang sudah disimpan?
                kept.append(d)
        out.extend(kept)
    return out

def is_single_object(boxes):
    """Cek apakah mayoritas box menumpuk di satu lokasi (indikasi foto 1 objek)."""
    if len(boxes) == 0:
        return False
    # pilih box terluas sebagai anchor
    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    base = boxes[int(np.argmax(areas))]
    # fraksi box yang overlap besar dengan anchor
    frac = np.mean([iou_xyxy(base, b) >= SINGLE_OBJ_IOU for b in boxes])
    return frac >= SINGLE_OBJ_MIN_FRAC

def single_object_gate(dets):
    """Jika 1 objek: kembalikan hanya 1 label dengan skor tertinggi."""
    if not dets:
        return dets
    if is_single_object([d["box"] for d in dets]):
        return [max(dets, key=lambda d: d["score"])]
    return dets

# ---------- PAIRING & CROSS-FAMILY ----------
def merge_pairs(detections):
    """
    Gabungkan pasangan matang vs mentah per keluarga buah.
    Jika overlap >= IOU_PAIR, pertahankan yang skornya lebih tinggi.
    """
    pairs = [('alpukat_matang', 'alpukat_mentah'),
             ('belimbing_matang', 'belimbing_mentah'),
             ('mangga_matang',   'mangga_mentah')]
    final = []
    for a, b in pairs:
        A = [d for d in detections if d["label"] == a]               # daftar deteksi 'a'
        B = [d for d in detections if d["label"] == b]               # daftar deteksi 'b'
        used_B = set()                                               # penanda B yang sudah dipakai
        for da in A:
            best_j, best_iou = -1, 0.0
            for j, db in enumerate(B):
                if j in used_B: continue
                iou = iou_xyxy(da["box"], db["box"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= IOU_PAIR:                                 # overlap cukup besar?
                chosen = da if da["score"] >= B[best_j]["score"] else B[best_j]
                final.append(chosen)                                  # ambil yang skor tertinggi
                used_B.add(best_j)
            else:
                final.append(da)                                      # tidak overlap? simpan apa adanya
        final.extend(db for j, db in enumerate(B) if j not in used_B) # tambahkan sisa B yang belum kepakai
    # tambahkan metadata "fam" = keluarga buah (alpukat/belimbing/mangga)
    for d in final:
        d["fam"] = d["label"].split('_')[0]
    return final

def cross_family_suppress(dets, thr=IOU_XFAM):
    """
    Menekan deteksi antar keluarga buah yang saling overlap besar.
    Urutkan by skor desc, pertahankan yang tertinggi; yang lain (keluarga beda) dibuang bila IoU>=thr.
    """
    dets = sorted(dets, key=lambda d: d["score"], reverse=True)      # urutkan dari skor tertinggi
    kept = []                                                        # hasil akhir
    for d in dets:
        ok = True
        for k in kept:
            if d["fam"] != k["fam"] and iou_xyxy(d["box"], k["box"]) >= thr:  # beda keluarga & overlap besar?
                ok = False
                break
        if ok:
            kept.append(d)
    return kept

def postprocess_with_nms(boxes, probs_crops, votes, min_votes, iou_nms=IOU_NMS):
    """
    Pipeline pasca-proses:
      1) Filter skor per-kelas + syarat votes.
      2) NMS per-kelas (ketat).
      3) Dedupe per-kelas (buang sisa duplikat).
      4) Merge matang vs mentah per keluarga.
      5) Cross-family suppression (antar keluarga buah).
      6) Single-object gate (jika semua box numpuk).
    """
    detections = []                                                  # kumpulan deteksi awal
    for c, label in enumerate(LABELS):
        scores = probs_crops[:, c]                                   # skor semua crop untuk kelas c
        mask = (scores >= THRESHOLDS[c])                             # lulus ambang?
        if not mask.any() or votes[c] < min_votes:                   # cukup votes?
            continue
        keep = nms_per_class(boxes[mask], scores[mask], iou_nms)     # NMS per-kelas (ketat)
        for k in keep:
            detections.append({
                "label": label,
                "c": c,
                "score": float(scores[mask][k]),
                "box":  boxes[mask][k]
            })

    # ---- DEDUPE per-kelas → kurangi sisa duplikat yang masih dekat
    detections = dedupe_same_class(detections, DEDUP_IOU_SAMECLASS)

    # ---- Pairing matang/mentah per keluarga
    dets = merge_pairs(detections)

    # ---- Penekanan antar keluarga (buah berbeda yang numpuk)
    dets = cross_family_suppress(dets, thr=IOU_XFAM)

    # ---- Jika 1 objek → tampilkan hanya 1 label
    dets = single_object_gate(dets)

    return sorted(dets, key=lambda d: d["score"], reverse=True)      # urut skor desc

# ===========================
# 6) UI STREAMLIT
# ===========================
st.title("🍉 Klasifikasi Multi-Label Buah")                          # judul aplikasi
uploaded_file = st.file_uploader("Unggah gambar buah", type=['jpg', 'jpeg', 'png'])  # uploader

if uploaded_file:                                                    # bila ada file
    image = Image.open(uploaded_file).convert('RGB')                 # baca sebagai RGB
    st.image(image, caption="Gambar Input", use_container_width=True)# tampilkan gambar

    # --- Sliding-window: buat crop & jalankan prediksi
    crops, boxes     = gen_windows(image)                            # (list PIL), (N,4) xyxy
    probs_crops      = run_preds_on_crops(crops)                     # (N,C) probabilitas
    num_crops        = probs_crops.shape[0]                          # total crop
    probs_max        = probs_crops.max(0)                            # agregasi MAX antar-crop (info)
    votes            = (probs_crops >= np.array(THRESHOLDS)).sum(0).astype(int)  # votes per kelas

    # --- Votes dinamis: minimal 5% dari jumlah crop, tapi ≥3
    MIN_VOTES_dyn    = max(3, int(math.ceil(0.05 * num_crops)))      # syarat konsensus minimal

    # --- Post-process (NMS + suppression)
    final_dets = postprocess_with_nms(boxes, probs_crops, votes, MIN_VOTES_dyn, iou_nms=IOU_NMS)

    # --- Tampilkan hasil ringkas
    st.subheader("🔍 Label Terdeteksi:")
    if not final_dets:
        st.warning("🚫 Tidak ada label terdeteksi.")
    else:
        st.write(f"Total label terdeteksi: **{len(final_dets)}**")   # jumlah label akhir
        for det in final_dets:
            line = f"✅ *{det['label']}* ({det['score']:.2%})"       # nama label + skor
            if SHOW_VOTES:                                           # opsional tampilkan votes
                line += f" | votes={int(votes[det['c']])}"
            st.write(line)

    # --- Panel detail (opsional analitik)
    with st.expander("📊 Detail Probabilitas & Konsensus"):
        mean_prob = float(np.mean(probs_max))                        # rata-rata prob. (informasi)
        entropy   = -float(np.mean([p * math.log(p + 1e-8) for p in probs_max]))  # entropi sederhana
        st.write(f"🪟 crops: {num_crops} | min_votes: {MIN_VOTES_dyn} | mean_prob: {mean_prob:.3f} | entropy: {entropy:.3f}")
        for i, lbl in enumerate(LABELS):
            pass_thr = "✓" if probs_max[i] >= THRESHOLDS[i] else "✗"              # lulus ambang?
            line = f"{lbl}: {probs_max[i]:.2%} (thr {THRESHOLDS[i]:.2f}) {pass_thr}"
            if SHOW_VOTES:
                line += f" | votes={int(votes[i])}"
            st.write(line)
