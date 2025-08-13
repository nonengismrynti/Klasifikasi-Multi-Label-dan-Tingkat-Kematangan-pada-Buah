import streamlit as st                      # UI web
import torch                                # PyTorch core
import torch.nn as nn                       # Modul neural network
from safetensors.torch import safe_open     # Loader bobot .safetensors
import torchvision.transforms as transforms # Transformasi gambar
from PIL import Image                       # Baca gambar
import gdown                                # Download dari Google Drive
import os                                   # Utilitas file/path
import math                                 # Info metrik tambahan
import traceback                            # Tampilkan traceback saat error
import numpy as np                          # Numpy untuk array & IoU

# ========================
# 1) Setup umum & label
# ========================
MODEL_URL  = 'https://drive.google.com/uc?id=1GPPxPSpodNGJHSeWyrTVwRoSjwW3HaA8'  # Link model di Drive
MODEL_PATH = 'model_3.safetensors'                                               # Nama file lokal model

LABELS = [                                   # Urutan label HARUS sama seperti saat training
    'alpukat_matang', 'alpukat_mentah',
    'belimbing_matang', 'belimbing_mentah',
    'mangga_matang', 'mangga_mentah'
]

# ======================================================
# 2) Parameter model + inferensi (mengikuti retrain-5)
# ======================================================
NUM_HEADS   = 10                             # Jumlah attention heads
NUM_LAYERS  = 4                              # Banyak InteractionBlock
HIDDEN_DIM  = 640                            # Dimensi embedding
PATCH_SIZE  = 14                             # Ukuran patch conv
IMAGE_SIZE  = 210                            # Ukuran input ke model
THRESHOLDS  = [0.01, 0.01, 0.01, 0.06, 0.02, 0.01]  # Ambang per kelas (dari validasi)

# ==========
# Sliding window + clustering
# ==========
WIN_FRACS    = (0.55, 0.7, 0.85, 1.0)        # Ukuran jendela relatif sisi terpendek (beberapa skala)
STRIDE_FRAC  = 0.28                          # Overlap besar (stride = 0.28 * window)
IOU_THR      = 0.35                          # IoU untuk mengelompokkan crop ke objek yang sama
MIN_VOTES    = 2                             # Minimal crop pendukung dalam 1 klaster
MIN_SCORE    = 0.10                          # Minimal skor rata-rata klaster agar valid (filter noise)

# ==========================================
# 3) Download model bila belum ada / korup
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

# =========================================
# 7) Utilitas IoU & sliding window + infer
# =========================================
def iou(box1, box2):                            # Hitung Intersection-over-Union dua kotak
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    inter_x1, inter_y1 = max(x1, a1), max(y1, b1)
    inter_x2, inter_y2 = min(x2, a2), min(y2, b2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (a2 - a1) * (b2 - b1)
    union = area1 + area2 - inter + 1e-8
    return inter / union

def sliding_window_infer(image_pil):            # Potong gambar → prediksi tiap crop → simpan box & skor
    W, H = image_pil.size                       # Lebar & tinggi gambar asli
    short = min(W, H)                           # Sisi terpendek (acuan ukuran window)
    boxes, probs = [], []                       # Simpan bbox & probabilitas per-crop
    num_tokens = (IMAGE_SIZE // PATCH_SIZE) ** 2# Jumlah token dummy text

    with torch.no_grad():                       # Non-grad untuk inferensi
        for wf in WIN_FRACS:                    # Loop beberapa skala jendela
            win  = max(int(short * wf), 64)     # Ukuran jendela (px)
            step = max(1, int(win * STRIDE_FRAC))# Langkah geser (overlap)
            for top in range(0, max(H - win + 1, 1), step):          # Geser vertikal
                for left in range(0, max(W - win + 1, 1), step):     # Geser horizontal
                    box = (left, top, left + win, top + win)         # x1,y1,x2,y2
                    crop = image_pil.crop(box)                       # Ambil crop persegi
                    x = transform(crop).unsqueeze(0).to(device)      # Transform → Tensor [1,3,210,210]
                    dummy_text = torch.zeros((1, num_tokens, HIDDEN_DIM), device=device)  # Dummy text zeros
                    logits = model(x, dummy_text)                    # Forward → logits [1,C]
                    p = torch.sigmoid(logits).cpu().numpy()[0]       # Sigmoid → probabilitas [C]
                    boxes.append(box)                                # Simpan bbox
                    probs.append(p)                                  # Simpan probabilitas

    return np.array(boxes, dtype=np.int32), np.array(probs, dtype=np.float32)  # Kembalikan semua crop

# ============================================================
# 8) Klasterisasi crop → 1 label per objek (winner in cluster)
# ============================================================
def cluster_and_label(boxes, probs):
    """
    - Ambil prediksi terbaik tiap crop (kelas & skor tertinggi).
    - Buang crop dgn skor < threshold kelasnya (reduksi noise).
    - Klasterkan crop berdasarkan IoU>IOU_THR (objek yang sama).
    - Untuk tiap klaster: rata-ratakan skor per-kelas, ambil kelas
      dgn skor rata-rata tertinggi → itulah label objek itu.
    - Hasil: list label (satu per buah/objek).
    """
    if len(boxes) == 0:                         # Jika tidak ada crop, kembalikan kosong
        return []

    # Tentukan kelas terbaik per crop
    top_ids   = probs.argmax(axis=1)            # Index kelas dengan skor tertinggi per crop
    top_scores= probs.max(axis=1)               # Skor tertinggi per crop
    # Ambang dinamis: pakai threshold spesifik kelas
    keep_mask = np.array([top_scores[i] >= THRESHOLDS[top_ids[i]] for i in range(len(top_ids))], dtype=bool)
    boxes  = boxes[keep_mask]                   # Saring bbox yang cukup yakin
    probs  = probs[keep_mask]                   # Saring probabilitasnya juga
    if len(boxes) == 0:                         # Jika semua terbuang, kembalikan kosong
        return []

    # Urutkan crop berdasarkan skor terbaik (desc) agar klaster stabil
    order = np.argsort(-probs.max(axis=1))
    boxes, probs = boxes[order], probs[order]

    clusters = []                               # List klaster (tiap klaster = list index crop)
    assigned = np.zeros(len(boxes), dtype=bool) # Penanda crop sudah masuk klaster

    for i in range(len(boxes)):                 # Greedy clustering
        if assigned[i]:                         # Skip jika sudah dikelompokkan
            continue
        # Buat klaster baru dengan anchor i
        cluster_idx = [i]
        assigned[i] = True
        for j in range(i+1, len(boxes)):        # Cari crop lain yang overlap tinggi
            if assigned[j]: 
                continue
            if iou(boxes[i], boxes[j]) >= IOU_THR:  # Jika IoU cukup besar → dianggap objek yg sama
                cluster_idx.append(j)
                assigned[j] = True
        clusters.append(cluster_idx)            # Simpan klaster

    results = []                                # Hasil akhir (label, skor, jumlah-vote)
    for idxs in clusters:                       # Untuk tiap klaster
        kl_probs = probs[idxs]                  # Ambil semua prob di klaster [n, C]
        votes = len(idxs)                       # Banyak crop pendukung
        mean_scores = kl_probs.mean(axis=0)     # Rata-rata skor per-kelas di klaster [C]
        best_id = int(mean_scores.argmax())     # Pilih kelas dengan skor rata-rata tertinggi
        best_score = float(mean_scores[best_id])# Skor rata-rata kelas terpilih
        # Filter akhir: butuh vote cukup & skor rata-rata di atas threshold & MIN_SCORE umum
        if votes >= MIN_VOTES and best_score >= max(THRESHOLDS[best_id], MIN_SCORE):
            results.append((LABELS[best_id], best_score, votes))  # Simpan sebagai 1 objek terdeteksi

    # Urutkan dari skor tertinggi supaya rapi
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# ==========
# 9) UI App
# ==========
st.title("🍉 Klasifikasi Multi-Label Buah (Sliding Window + Clustering)")
st.write("Target: jumlah label ≈ jumlah buah. 1 buah → 1 label; 3 buah → 3 label; 6 buah → 6 label.")

uploaded_file = st.file_uploader("Unggah gambar buah", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')                # Baca sebagai RGB
    st.image(image, caption="Gambar Input", use_container_width=True)  # Tampilkan

    # (1) Prediksi semua crop (sliding window)
    boxes, probs = sliding_window_infer(image)                      # Dapatkan semua bbox & prob per-crop

    # (2) Klasterkan crop → satu label per objek (buah)
    detections = cluster_and_label(boxes, probs)                    # Hasil: [(label, score, votes), ...]

    # (3) Tampilkan hasil
    st.subheader("🔍 Label Terdeteksi:")
    if not detections:
        st.warning("🚫 Tidak ada objek buah yang lolos kriteria.")
    else:
        for label, score, votes in detections:
            st.write(f"✅ *{label}* — {score:.2%} (votes: {votes})")

    # (4) Panel detail (opsional, metrik ringkas)
    with st.expander("📊 Ringkasan Proses"):
        total_crops = len(boxes)
        st.write(f"🪟 total crop: {total_crops} | klaster terpilih: {len(detections)}")
        # Rata-rata maksimum skor per-crop (indikasi “keyakinan” agregat, tidak memengaruhi hasil)
        if total_crops > 0:
            max_per_crop = probs.max(axis=1)
            st.write(f"📈 rata-rata skor max per-crop: {float(max_per_crop.mean()):.3f}")
