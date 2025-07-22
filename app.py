import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
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
THRESHOLD = 0.3

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
    def __init__(self, in_channels=3, patch_size=14, emb_size=640):
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
    def __init__(self, dim, num_heads=10):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        return attn_output

class CrossScaleAggregation(nn.Module):
    def __init__(self, embed_dim=640, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.linears = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_scales)
        ])

    def forward(self, x):
        B, N, E = x.shape
        scales = []
        for i in range(self.num_scales):
            pool_size = max(1, N // (2 ** i))
            reshaped = x.permute(0, 2, 1)
            pooled = F.adaptive_avg_pool1d(reshaped, output_size=pool_size)
            pooled = pooled.permute(0, 2, 1)
            pooled_mean = pooled.mean(dim=1, keepdim=True)
            scaled = self.linears[i](pooled_mean)
            scales.append(scaled)
        agg = torch.cat(scales, dim=1)
        return agg

class HamburgerHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)

class MLPClassifier(nn.Module):
    def __init__(self, in_dim=640, num_classes=9, hidden_dim=256):  # 256 sesuai checkpoint
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)

class HSVLTModel(nn.Module):
    def __init__(self, img_size=210, patch_size=14, emb_size=640, num_classes=9):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size=patch_size, emb_size=emb_size)
        self.word_embed = nn.Identity()
        self.concat = FeatureFusion()
        self.scale_transform = ScaleTransformation(emb_size * 2, emb_size)
        self.channel_unification = ChannelUnification(emb_size)
        self.interaction_blocks = nn.Sequential(
            InteractionBlock(emb_size),
            InteractionBlock(emb_size),
            InteractionBlock(emb_size),
            InteractionBlock(emb_size)
        )
        self.csa = CrossScaleAggregation(embed_dim=emb_size)
        self.head = HamburgerHead(emb_size, emb_size)  # head.linear
        self.classifier = MLPClassifier(in_dim=emb_size, num_classes=num_classes, hidden_dim=256)

    def forward(self, image):
        B = image.size(0)
        dummy_text = torch.randn(B, 1, 640).to(image.device)
        image_feat = self.patch_embed(image)
        x = self.concat(image_feat, dummy_text)
        x = self.scale_transform(x)
        x = self.channel_unification(x)
        x = self.interaction_blocks(x)
        x = self.csa(x)
        x = self.head(x)
        x = x.mean(dim=1)
        return self.classifier(x)

# --- 4. Load Model ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    with safe_open(MODEL_PATH, framework="pt", device=device) as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    model = HSVLTModel().to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# --- 5. Transformasi Gambar ---
transform = transforms.Compose([
    transforms.Resize((210, 210)),
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

    detected_labels = [(label, prob) for label, prob in zip(LABELS, probs) if prob >= THRESHOLD]
    detected_labels.sort(key=lambda x: x[1], reverse=True)

    st.subheader("🔍 Label Terdeteksi:")
    if detected_labels:
        for label, prob in detected_labels:
            st.write(f"✅ **{label}** ({prob:.2%})")
    else:
        st.warning("🚫 Gambar tidak mengandung buah yang dikenali.")

    with st.expander("📊 Lihat Semua Probabilitas"):
        for label, prob in zip(LABELS, probs):
            st.write(f"{label}: {prob:.2%}")