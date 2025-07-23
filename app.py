import streamlit as st
import torch
import torch.nn as nn
from safetensors.torch import safe_open
import torchvision.transforms as transforms
from PIL import Image
import gdown
import os

# --- 1. Setup ---
MODEL_URL = 'https://drive.google.com/uc?id=1OB3XnCwW2MiEAzPJDMO9bD50_6qG4aR_'
MODEL_PATH = 'model_1.safetensors'

LABELS = [
    'alpukat', 'alpukat_matang', 'alpukat_mentah',
    'belimbing', 'belimbing_matang', 'belimbing_mentah',
    'mangga', 'mangga_matang', 'mangga_mentah'
]

# ✅ Parameter sesuai training checkpoint
HIDDEN_DIM = 640
PATCH_SIZE = 14
IMAGE_SIZE = 210
NUM_HEADS = 10     # 640/10 = 64 per head
NUM_LAYERS = 4     # checkpoint punya 4 layers
THRESHOLD = 0.30

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
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=PATCH_SIZE, emb_size=HIDDEN_DIM):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class FeatureFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, visual_embed, text_embed):
        text_expand = text_embed.mean(dim=1, keepdim=True).repeat(1, visual_embed.size(1), 1)
        fused = torch.cat([visual_embed, text_expand], dim=-1)
        return fused

class ScaleTransformation(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)

class ChannelUnification(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)

class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=NUM_HEADS):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        return attn_output

class HamburgerHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)

class MLPClassifier(nn.Module):
    def __init__(self, in_dim=HIDDEN_DIM, num_classes=9, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)

class HSVLTModel(nn.Module):
    def __init__(self, patch_size=PATCH_SIZE, emb_size=HIDDEN_DIM, num_classes=9):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size=patch_size, emb_size=emb_size)
        self.word_embed = nn.Identity()
        self.concat = FeatureFusion()
        self.scale_transform = ScaleTransformation(emb_size * 2, emb_size)
        self.channel_unification = ChannelUnification(emb_size)
        self.interaction_blocks = nn.Sequential(
            *[InteractionBlock(emb_size, NUM_HEADS) for _ in range(NUM_LAYERS)]
        )
        self.head = HamburgerHead(emb_size, emb_size)
        self.classifier = MLPClassifier(in_dim=emb_size, num_classes=num_classes, hidden_dim=256)

    def forward(self, image):
        B = image.size(0)
        dummy_text = torch.randn(B, 1, HIDDEN_DIM).to(image.device)
        image_feat = self.patch_embed(image)
        x = self.concat(image_feat, dummy_text)
        x = self.scale_transform(x)
        x = self.channel_unification(x)
        x = self.interaction_blocks(x)
        x = self.head(x)
        x = x.mean(dim=1)
        return self.classifier(x)

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

    # ✅ Load dengan strict=False supaya key mismatch diabaikan
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        st.warning(f"⚠️ Missing keys saat load model: {missing}")
    if unexpected:
        st.warning(f"⚠️ Unexpected keys saat load model: {unexpected}")

    model.eval()

except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# --- 5. Transformasi Gambar ---
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# --- 6. Streamlit UI ---
st.title("🍉 Klasifikasi Multilabel Buah")
st.write("Upload gambar buah, sistem akan mendeteksi beberapa label sekaligus.")

uploaded_file = st.file_uploader("Unggah gambar buah", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar Input", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0].tolist()

    # Ambil label di atas threshold
    detected_labels = [(label, prob) for label, prob in zip(LABELS, probs) if prob >= THRESHOLD]
    detected_labels.sort(key=lambda x: x[1], reverse=True)

    # ✅ Hitung statistik tambahan
    max_prob = max(probs)
    mean_prob = sum(probs) / len(probs)
    high_conf_labels = [p for p in probs if p > 0.6]

    st.subheader("🔍 Label Terdeteksi:")

    # ✅ RULE: Jika terlalu sedikit label yang masuk, atau confidence aneh → anggap OOD
    if (max_prob < 0.6) or (mean_prob < 0.15) or len(high_conf_labels) <= 1:
        st.warning("🚫 Tidak ada label yang melewati ambang batas.")
    else:
        if detected_labels:
            for label, prob in detected_labels:
                st.write(f"✅ **{label}** ({prob:.2%})")
        else:
            st.warning("🚫 Tidak ada label yang melewati ambang batas.")

    # ✅ Tetap tampilkan semua probabilitas untuk debugging
    with st.expander("📊 Lihat Semua Probabilitas"):
        for label, prob in zip(LABELS, probs):
            st.write(f"{label}: {prob:.2%}")

