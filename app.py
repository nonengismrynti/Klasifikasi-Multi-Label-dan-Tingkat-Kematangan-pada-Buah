import streamlit as st
import torch
import torch.nn as nn
from safetensors.torch import safe_open
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights  # <— fruit gate
from PIL import Image
import gdown
import os
import math
import traceback
import numpy as np

# --- 1. Setup ---
MODEL_URL = 'https://drive.google.com/uc?id=1GPPxPSpodNGJHSeWyrTVwRoSjwW3HaA8'
MODEL_PATH = 'model_3.safetensors'

LABELS = [
    'alpukat_matang', 'alpukat_mentah',
    'belimbing_matang', 'belimbing_mentah',
    'mangga_matang', 'mangga_mentah'
]

# ==== PARAM ====
NUM_HEADS  = 10
NUM_LAYERS = 4
HIDDEN_DIM = 640
PATCH_SIZE = 14
IMAGE_SIZE = 210
THRESHOLDS = [0.01, 0.01, 0.01, 0.06, 0.02, 0.01]

# OOD / inference
USE_MULTICROP   = True
GRID_SIDE       = 2                # 2x2 + full image = 5 crops
MIN_VOTES       = 2                # minimal crop yang setuju untuk mengaktifkan satu label
ENABLE_FRUIT_GATE = True           # set False kalau mau memaksa tanpa gate
FRUIT_SUM_MIN   = 0.15             # ambang probabilitas “buah” (ImageNet) agar dianggap gambar buah

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
    def __init__(self, img_size, patch_size, emb_size):
        super().__init__()
        self.proj = nn.Conv2d(3, emb_size, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class WordEmbedding(nn.Module):
    def __init__(self, dim): super().__init__()
    def forward(self, x): return x

class FeatureFusion(nn.Module):
    def forward(self, v, t): return torch.cat([v, t[:, :v.size(1), :]], dim=-1)

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
    def forward(self, x): return x.mean(dim=1, keepdim=True)

class HamburgerHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.linear(x)

class MLPClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.mlp(x)

class HSVLTModel(nn.Module):
    def __init__(self, img_size=210, patch_size=14, emb_size=HIDDEN_DIM,
                 num_classes=6, num_heads=NUM_HEADS, num_layers=NUM_LAYERS):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, emb_size)
        self.word_embed = WordEmbedding(emb_size)
        self.concat = FeatureFusion()
        self.scale_transform = ScaleTransformation(emb_size * 2, emb_size)
        self.channel_unification = ChannelUnification(emb_size)
        self.interaction_blocks = nn.Sequential(
            *[InteractionBlock(emb_size, num_heads=num_heads) for _ in range(num_layers)]
        )
        self.csa = CrossScaleAggregation()
        self.head = HamburgerHead(emb_size, emb_size)
        self.classifier = MLPClassifier(emb_size, num_classes)

    def forward(self, image, text):
        image_feat = self.patch_embed(image)
        text_feat  = self.word_embed(text)
        x = self.concat(image_feat, text_feat)
        x = self.scale_transform(x)
        x = self.channel_unification(x)
        x = self.interaction_blocks(x)
        x = self.csa(x)
        x = self.head(x)
        x = x.mean(dim=1)
        return self.classifier(x)

# --- 4. Load Model ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if len(THRESHOLDS) != len(LABELS):
    st.error("Panjang THRESHOLDS tidak sama dengan jumlah LABELS.")
    st.stop()

try:
    with safe_open(MODEL_PATH, framework="pt", device=device) as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}

    model = HSVLTModel(
        patch_size=PATCH_SIZE,
        emb_size=HIDDEN_DIM,
        num_classes=len(LABELS),
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.code(traceback.format_exc())
    st.stop()

# --- 5. Transformasi Gambar ---
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- 5.1 Fruit gate (ImageNet) ---
FRUIT_GATE_OK = False
if ENABLE_FRUIT_GATE:
    try:
        _weights = ResNet50_Weights.IMAGENET1K_V2
        fruit_gate = resnet50(weights=_weights).to(device).eval()
        imagenet_tf = _weights.transforms()
        imagenet_labels = _weights.meta["categories"]
        FRUIT_KEYWORDS = [
            "avocado","mango","banana","orange","lemon","lime","pineapple","pomegranate",
            "apple","pear","peach","apricot","plum","grape","strawberry","raspberry",
            "blueberry","blackberry","passion","papaya","guava","fig","date","kiwi",
            "cucumber","zucchini","plantain","melon","cantaloupe","honeydew","durian",
            "jackfruit","lychee","longan","tamarind","starfruit","carambola"
        ]
        FRUIT_IDX = [i for i,n in enumerate(imagenet_labels) if any(k in n.lower() for k in FRUIT_KEYWORDS)]
        FRUIT_GATE_OK = len(FRUIT_IDX) > 0
    except Exception as _:
        FRUIT_GATE_OK = False

def fruit_probability(pil_img):
    """Kembalikan (prob_sum_fruits, top1_prob, top1_label) dari ResNet50 ImageNet."""
    if not FRUIT_GATE_OK:
        return None
    x = imagenet_tf(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = fruit_gate(x)
        probs  = torch.softmax(logits, dim=1)[0]
    prob_sum = float(probs[FRUIT_IDX].sum().item())
    top1_prob, top1_idx = probs.max(dim=0)
    return prob_sum, float(top1_prob.item()), imagenet_labels[int(top1_idx)]

# --- 5.2 Multi-crop inference ---
def infer_multicrop_probs(image_pil, n_side=2):
    W, H = image_pil.size
    crops = [image_pil.crop((int(W*j/n_side), int(H*i/n_side),
                             int(W*(j+1)/n_side), int(H*(i+1)/n_side)))
             for i in range(n_side) for j in range(n_side)]
    crops.append(image_pil)  # full image

    num_tokens = (IMAGE_SIZE // PATCH_SIZE) ** 2
    probs_list = []
    with torch.no_grad():
        for crop in crops:
            x = transform(crop).unsqueeze(0).to(device)
            dummy_text = torch.zeros((1, num_tokens, HIDDEN_DIM), device=device)
            p = torch.sigmoid(model(x, dummy_text)).cpu().numpy()[0]
            probs_list.append(p)
    return np.stack(probs_list, axis=0)  # [num_crops, num_classes]

# --- 6. Streamlit UI ---
st.title("🍉 Klasifikasi Multi-Label Buah")
st.write("Upload gambar buah; sistem akan menolak objek non-buah.")

uploaded_file = st.file_uploader("Unggah gambar", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar Input", use_container_width=True)

    # 6.1 Fruit gate (tolak non-buah sedini mungkin)
    gate_info = fruit_probability(image) if ENABLE_FRUIT_GATE else None
    if gate_info is not None:
        fruit_sum, top1_p, top1_label = gate_info
        if fruit_sum < FRUIT_SUM_MIN:
            st.warning(f"🚫 Ini **bukan buah**.")
            st.stop()

    # 6.2 Inference (multi-crop + votes)
    probs_crops = infer_multicrop_probs(image, n_side=GRID_SIDE) if USE_MULTICROP \
                  else infer_multicrop_probs(image, n_side=1)
    probs_max   = probs_crops.max(axis=0)                     # agregasi nilai
    votes       = (probs_crops >= np.array(THRESHOLDS)).sum(axis=0)  # dukungan per kelas

    # aktifkan label hanya jika melewati threshold & punya votes cukup
    detected = [(lbl, float(p), int(v)) 
                for lbl, p, thr, v in zip(LABELS, probs_max, THRESHOLDS, votes)
                if (p >= thr and v >= MIN_VOTES)]
    detected.sort(key=lambda x: x[1], reverse=True)

    # statistik
    mean_prob = float(probs_max.mean())
    entropy = -float(np.mean([p * math.log(p + 1e-8) for p in probs_max]))

    st.subheader("🔍 Label Terdeteksi:")
    if not detected:
        st.warning("Tidak ada label yang memenuhi kriteria (threshold + votes).")
    else:
        for label, prob, v in detected:
            st.write(f"✅ *{label}* ({prob:.2%}) — votes: {v}")

    with st.expander("📊 Probabilitas & Votes per Kelas"):
        st.write(f"📊 mean_prob: {mean_prob:.3f} | entropy: {entropy:.3f} | crops: {probs_crops.shape[0]}")
        for i, lbl in enumerate(LABELS):
            st.write(f"{lbl}: max={probs_max[i]:.2%} | thr={THRESHOLDS[i]:.2f} | votes={votes[i]}")