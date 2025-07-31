import streamlit as st
import torch
import torch.nn as nn
from safetensors.torch import safe_open
import torchvision.transforms as transforms
from PIL import Image
import gdown
import os
import math

# --- 1. Setup ---
MODEL_URL = 'https://drive.google.com/uc?id=1PbHLaNkAToSVsVnkGo2N8DhAXyOqvd7F'
MODEL_PATH = 'model_2.safetensors'

LABELS = [
    'alpukat_matang', 'alpukat_mentah',
    'belimbing_matang', 'belimbing_mentah',
    'mangga_matang', 'mangga_mentah'
]

# ✅ Gunakan parameter eksperimen 7
HIDDEN_DIM   = 512
PATCH_SIZE   = 14
IMAGE_SIZE   = 210
NUM_HEADS    = 8   
NUM_LAYERS   = 4
THRESHOLD    = 0.30

# --- 2. Download model ---
def download_model():
    with st.spinner('🔄 Mengunduh ulang model dari Google Drive...'):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 50000:
    if os.path.exists(MODEL_PATH):
        st.warning("📦 Ukuran file model terlalu kecil, kemungkinan korup. Mengunduh ulang...")
        os.remove(MODEL_PATH)
    download_model()

# --- 3. Komponen Model ---
# 🔹 Komponen Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, emb_size):
        super().__init__()
        self.proj = nn.Conv2d(3, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                          # [B, C, H/ps, W/ps]
        x = x.flatten(2).transpose(1, 2)          # [B, num_patches, emb_size]
        return x

# 🔹 Komponen Word Embedding (sederhana → input dummy)
class WordEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
    def forward(self, x): return x               # x = dummy_text [B, 225, 640]

# 🔹 Gabungkan visual dan teks
class FeatureFusion(nn.Module):
    def forward(self, v, t):                     # v = visual [B, N, E], t = text [B, N, E]
        return torch.cat([v, t[:, :v.size(1), :]], dim=-1)  # Gabung di dimensi fitur

# 🔹 Scale Transformation
class ScaleTransformation(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.linear(x)

# 🔹 Channel Normalisasi
class ChannelUnification(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
    def forward(self, x): return self.norm(x)

# 🔹 Attention Block
class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
    def forward(self, x): return self.attn(x, x, x)[0]

# 🔹 CSA Dummy (agregasi skala)
class CrossScaleAggregation(nn.Module):
    def forward(self, x): return x.mean(dim=1, keepdim=True)  # [B, 1, D]

# 🔹 Linear Head
class HamburgerHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.linear(x)

# 🔹 Multi-Label Classifier
class MLPClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.mlp(x)

# ✅ Gabungkan semuanya ke dalam HSVLTModel
class HSVLTModel(nn.Module):
    def __init__(self, img_size=210, patch_size=14, emb_size=512, num_classes=6):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, emb_size)
        self.word_embed = WordEmbedding(emb_size)
        self.concat = FeatureFusion()
        self.scale_transform = ScaleTransformation(emb_size * 2, emb_size)
        self.channel_unification = ChannelUnification(emb_size)
        self.interaction_blocks = nn.Sequential(
            InteractionBlock(emb_size, num_heads=8),
            InteractionBlock(emb_size, num_heads=8),
            InteractionBlock(emb_size, num_heads=8),
            InteractionBlock(emb_size, num_heads=8)
        )
        self.csa = CrossScaleAggregation()
        self.head = HamburgerHead(emb_size, emb_size)
        self.classifier = MLPClassifier(emb_size, num_classes)

    def forward(self, image, text):
        image_feat = self.patch_embed(image)         # [B, N, E]
        text_feat = self.word_embed(text)            # [B, N, E]
        x = self.concat(image_feat, text_feat)       # Gabung visual & teks
        x = self.scale_transform(x)
        x = self.channel_unification(x)
        x = self.interaction_blocks(x)
        x = self.csa(x)
        x = self.head(x)
        x = x.mean(dim=1)                            # Agregasi fitur patch
        return self.classifier(x)                    # [B, num_classes]

import traceback  # Tambahkan ini di atas file jika belum ada

# --- 4. Load Model ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    with safe_open(MODEL_PATH, framework="pt", device=device) as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}

    model = HSVLTModel(
        patch_size=PATCH_SIZE,
        emb_size=HIDDEN_DIM,
        num_classes=len(LABELS)
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.code(traceback.format_exc())  # ← tampilkan traceback asli di UI
    st.stop()



# --- 5. Transformasi Gambar ---
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- 6. Streamlit UI ---
st.title("🍉 Klasifikasi Multilabel Buah")
st.write("Upload gambar buah, sistem akan mendeteksi beberapa label sekaligus. Jika bukan buah, akan ditolak.")

uploaded_file = st.file_uploader("Unggah gambar buah", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar Input", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    # 🧠 Gunakan dummy_text yang distribusinya random, bukan nol
    dummy_text = torch.randn((1, (IMAGE_SIZE // PATCH_SIZE) ** 2, HIDDEN_DIM)).to(device)

with torch.no_grad():
    logits = model(input_tensor, dummy_text)
    probs = torch.sigmoid(logits).cpu().numpy()[0].tolist()


    with torch.no_grad():
        outputs = model(input_tensor, dummy_text)
        probs = torch.sigmoid(outputs).cpu().numpy()[0].tolist()

        # Ambil label di atas threshold
        detected_labels = [(label, prob) for label, prob in zip(LABELS, probs) if prob >= THRESHOLD]
        detected_labels.sort(key=lambda x: x[1], reverse=True)

        # --- OOD DETECTION ---
        max_prob = max(probs)
        sorted_probs = sorted(probs, reverse=True)
        second_max_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        mean_prob = sum(probs) / len(probs)
        high_conf_labels = [(lbl, p) for lbl, p in zip(LABELS, probs) if p > 0.7]

        entropy = -sum([p * math.log(p + 1e-8) for p in probs]) / len(probs)
        high_conf_count = len([p for p in probs if p > 0.2])
        is_ood = (high_conf_count < 2)

        st.subheader("🔍 Label Terdeteksi:")

        if is_ood:
            st.warning("🚫 Gambar tidak mengandung buah yang dikenali.")
        else:
            if detected_labels:
                for label, prob in detected_labels:
                    st.write(f"✅ *{label}* ({prob:.2%})")
            else:
                st.warning("🚫 Tidak ada label yang melewati ambang batas.")

        # ✅ Debugging
        with st.expander("📊 Lihat Semua Probabilitas"):
            st.write(f"📊 mean_prob: {mean_prob:.3f} | entropy: {entropy:.3f}")
            for label, prob in zip(LABELS, probs):
                st.write(f"{label}: {prob:.2%}")
