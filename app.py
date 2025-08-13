import streamlit as st
import torch
import torch.nn as nn
from safetensors.torch import safe_open
import torchvision.transforms as transforms
from PIL import Image
import gdown
import os
import math
import traceback
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO

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
# pakai threshold per-kelas (dari tuning test retrain-5)
THRESHOLDS = [0.01, 0.01, 0.01, 0.06, 0.02, 0.01]

# --- 2. Download model ---
def download_model():
    with st.spinner('🔄 Mengunduh ulang model dari Google Drive...'):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 50000:
    if os.path.exists(MODEL_PATH):
        st.warning("📦 Ukuran file model terlalu kecil, kemungkinan korup. Mengunduh ulang...")
        os.remove(MODEL_PATH)
    download_model()

# --- 3. Sliding Window Functions ---
def sliding_window_crop(image: Image.Image, 
                       window_sizes: List[int] = [224, 280, 336], 
                       overlap_ratio: float = 0.3,
                       min_area_ratio: float = 0.15) -> List[Dict]:
    """
    Melakukan sliding window cropping pada gambar untuk meningkatkan deteksi.
    """
    crops = []
    orig_w, orig_h = image.size
    
    # Tambahkan gambar asli sebagai crop pertama
    crops.append({
        'image': image,
        'bbox': (0, 0, orig_w, orig_h),
        'scale': 1.0,
        'type': 'original'
    })
    
    for window_size in window_sizes:
        # Skip jika window terlalu besar
        if window_size > min(orig_w, orig_h):
            continue
            
        # Hitung step size berdasarkan overlap
        step_size = int(window_size * (1 - overlap_ratio))
        
        # Sliding window horizontal dan vertikal
        for y in range(0, orig_h - window_size + 1, step_size):
            for x in range(0, orig_w - window_size + 1, step_size):
                # Crop area
                x1, y1 = x, y
                x2, y2 = x + window_size, y + window_size
                
                # Pastikan tidak melebihi batas gambar
                x2 = min(x2, orig_w)
                y2 = min(y2, orig_h)
                
                # Skip jika area terlalu kecil
                area_ratio = (x2 - x1) * (y2 - y1) / (orig_w * orig_h)
                if area_ratio < min_area_ratio:
                    continue
                
                # Crop gambar
                crop_img = image.crop((x1, y1, x2, y2))
                
                crops.append({
                    'image': crop_img,
                    'bbox': (x1, y1, x2, y2),
                    'scale': window_size / min(orig_w, orig_h),
                    'type': 'sliding_window'
                })
    
    return crops

def aggregate_detections(detections: List[Dict], 
                        labels: List[str], 
                        confidence_threshold: float = 0.3) -> List[float]:
    """
    Agregasi deteksi dari multiple crops menggunakan weighted voting.
    """
    num_classes = len(labels)
    vote_scores = [[] for _ in range(num_classes)]
    
    # Kumpulkan votes untuk setiap class
    for det in detections:
        if det['confidence'] >= confidence_threshold:
            class_id = det['class_id']
            confidence = det['confidence']
            scale = det['scale']
            
            # Weight berdasarkan confidence dan scale
            weight = confidence * (1.0 + scale * 0.2)  # Bias sedikit ke scale lebih besar
            vote_scores[class_id].append(weight)
    
    # Hitung skor final untuk setiap class
    aggregated_scores = []
    for class_votes in vote_scores:
        if len(class_votes) == 0:
            aggregated_scores.append(0.0)
        else:
            # Menggunakan weighted average dengan boost untuk multiple votes
            avg_score = sum(class_votes) / len(class_votes)
            vote_boost = min(len(class_votes) * 0.05, 0.2)  # Max boost 0.2
            final_score = min(avg_score + vote_boost, 1.0)
            aggregated_scores.append(final_score)
    
    return aggregated_scores

def multi_scale_detection(image: Image.Image, 
                         model, 
                         transform, 
                         device,
                         labels: List[str],
                         thresholds: List[float],
                         hidden_dim: int = 640,
                         image_size: int = 210,
                         patch_size: int = 14,
                         confidence_threshold: float = 0.3,
                         overlap_ratio: float = 0.25) -> Dict:
    """
    Melakukan deteksi multi-scale menggunakan sliding window.
    """
    
    # Generate crops menggunakan sliding window
    crops = sliding_window_crop(image, 
                               window_sizes=[200, 250, 300, 350],
                               overlap_ratio=overlap_ratio)
    
    all_detections = []
    crop_results = []
    
    # Proses setiap crop
    for i, crop_info in enumerate(crops):
        crop_img = crop_info['image']
        bbox = crop_info['bbox']
        scale = crop_info['scale']
        
        # Transform dan prediksi
        input_tensor = transform(crop_img).unsqueeze(0).to(device)
        
        # Dummy text (sesuai dengan kode asli)
        num_tokens = (image_size // patch_size) ** 2
        dummy_text = torch.zeros((1, num_tokens, hidden_dim), device=device)
        
        with torch.no_grad():
            outputs = model(input_tensor, dummy_text)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]
        
        # Deteksi per crop
        crop_detections = []
        for j, (label, prob, thr) in enumerate(zip(labels, probs, thresholds)):
            if prob >= thr:
                crop_detections.append({
                    'label': label,
                    'confidence': float(prob),
                    'class_id': j,
                    'bbox': bbox,
                    'scale': scale,
                    'crop_id': i
                })
        
        crop_results.append({
            'crop_info': crop_info,
            'probabilities': probs.tolist(),
            'detections': crop_detections
        })
        
        all_detections.extend(crop_detections)
    
    # Agregasi hasil menggunakan weighted voting
    aggregated_scores = aggregate_detections(all_detections, labels, confidence_threshold)
    
    # Filter berdasarkan threshold final
    final_detections = []
    for i, (label, score, thr) in enumerate(zip(labels, aggregated_scores, thresholds)):
        if score >= thr:
            final_detections.append({
                'label': label,
                'confidence': score,
                'class_id': i
            })
    
    # Sort berdasarkan confidence
    final_detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    return {
        'final_detections': final_detections,
        'aggregated_scores': aggregated_scores,
        'crop_results': crop_results,
        'total_crops': len(crops),
        'total_raw_detections': len(all_detections)
    }

def visualize_crops(image: Image.Image, crops: List[Dict]) -> Image.Image:
    """
    Visualisasi crops pada gambar asli untuk debugging.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    # Gambar bounding box untuk setiap crop (skip original)
    colors = plt.cm.Set3(np.linspace(0, 1, len(crops)))
    
    for i, crop_info in enumerate(crops):
        if crop_info['type'] == 'original':
            continue
            
        bbox = crop_info['bbox']
        x1, y1, x2, y2 = bbox
        
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor=colors[i], 
                               facecolor='none', alpha=0.7)
        ax.add_patch(rect)
        
        # Label crop
        ax.text(x1, y1-5, f'Crop {i}', fontsize=8, 
               color=colors[i], fontweight='bold')
    
    ax.set_title(f'Sliding Window Crops ({len(crops)-1} crops)')
    ax.axis('off')
    
    # Convert to PIL Image
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)

# --- 4. Komponen Model ---
# 🔹 Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, emb_size):
        super().__init__()
        self.proj = nn.Conv2d(3, emb_size, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)                          # [B, C, H/ps, W/ps]
        x = x.flatten(2).transpose(1, 2)          # [B, num_patches, emb_size]
        return x

# 🔹 Word Embedding (dummy: pass-through)
class WordEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
    def forward(self, x): 
        return x                                   # x = dummy_text [B, 225, 640]

# 🔹 Gabungan visual + teks
class FeatureFusion(nn.Module):
    def forward(self, v, t):                      # v = [B, N, E], t = [B, N, E]
        return torch.cat([v, t[:, :v.size(1), :]], dim=-1)

# 🔹 Scale Transformation
class ScaleTransformation(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x): 
        return self.linear(x)

# 🔹 Channel Normalization
class ChannelUnification(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
    def forward(self, x): 
        return self.norm(x)

# 🔹 Attention Block (sesuai retrain-5: num_heads=10)
class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=NUM_HEADS):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
    def forward(self, x): 
        return self.attn(x, x, x)[0]

# 🔹 CSA (agregasi skala sederhana)
class CrossScaleAggregation(nn.Module):
    def forward(self, x): 
        return x.mean(dim=1, keepdim=True)  # [B, 1, D]

# 🔹 Linear Head
class HamburgerHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x): 
        return self.linear(x)

# 🔹 Multi-Label Classifier
class MLPClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): 
        return self.mlp(x)

# ==== MODEL ====
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
        image_feat = self.patch_embed(image)         # [B, N, E]
        text_feat  = self.word_embed(text)           # [B, N, E]
        x = self.concat(image_feat, text_feat)
        x = self.scale_transform(x)
        x = self.channel_unification(x)
        x = self.interaction_blocks(x)
        x = self.csa(x)
        x = self.head(x)
        x = x.mean(dim=1)                            # [B, E]
        return self.classifier(x)                    # [B, num_classes]

# --- 5. Load Model ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Safety: cek konsistensi panjang THRESHOLDS
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

# --- 6. Transformasi Gambar ---
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- 7. Streamlit UI ---
st.title("🍉 Klasifikasi Multi-Label Buah")
st.write("Upload gambar buah, sistem akan mendeteksi beberapa label sekaligus. Jika bukan buah, akan ditolak.")

uploaded_file = st.file_uploader("Unggah gambar buah", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar Input", use_container_width=True)
    
    # Pilihan mode deteksi
    detection_mode = st.selectbox(
        "Pilih Mode Deteksi:",
        ["Single Shot (Original)", "Multi-Scale Sliding Window"],
        index=1  # Default ke sliding window
    )
    
    if detection_mode == "Single Shot (Original)":
        # Mode original (kode lama)
        input_tensor = transform(image).unsqueeze(0).to(device)
        num_tokens = (IMAGE_SIZE // PATCH_SIZE) ** 2
        dummy_text = torch.zeros((1, num_tokens, HIDDEN_DIM), device=device)

        with torch.no_grad():
            outputs = model(input_tensor, dummy_text)
            probs = torch.sigmoid(outputs).cpu().numpy()[0].tolist()

        detected_labels = [
            (label, prob) for label, prob, thr in zip(LABELS, probs, THRESHOLDS) if prob >= thr
        ]
        detected_labels.sort(key=lambda x: x[1], reverse=True)

        max_prob = max(probs)
        high_conf_count = sum(int(p >= t) for p, t in zip(probs, THRESHOLDS))
        is_ood = (high_conf_count < 1)

        st.subheader("🔍 Label Terdeteksi (Single Shot):")
        
        if is_ood:
            st.warning("🚫 Gambar tidak mengandung buah yang dikenali.")
        else:
            if detected_labels:
                for label, prob in detected_labels:
                    st.write(f"✅ **{label}** ({prob:.2%})")
            else:
                st.warning("🚫 Tidak ada label yang melewati ambang batas.")

        with st.expander("📊 Lihat Semua Probabilitas"):
            entropy = -sum([p * math.log(p + 1e-8) for p in probs]) / len(probs)
            st.write(f"📊 mean_prob: {sum(probs)/len(probs):.3f} | entropy: {entropy:.3f}")
            for label, prob, thr in zip(LABELS, probs, THRESHOLDS):
                pass_thr = "✓" if prob >= thr else "✗"
                st.write(f"{label}: {prob:.2%} (thr {thr:.2f}) {pass_thr}")
    
    else:
        # Mode Multi-Scale Sliding Window
        st.subheader("🔍 Multi-Scale Sliding Window Detection")
        
        # Parameter kontrol
        col1, col2, col3 = st.columns(3)
        with col1:
            confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.8, 0.3, 0.05)
        with col2:
            overlap_ratio = st.slider("Overlap Ratio", 0.1, 0.6, 0.25, 0.05)
        with col3:
            show_crops = st.checkbox("Tampilkan Visualisasi Crops", False)
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Jalankan multi-scale detection
            status_text.text("🔄 Memproses sliding window crops...")
            progress_bar.progress(0.2)
            
            results = multi_scale_detection(
                image=image,
                model=model,
                transform=transform,
                device=device,
                labels=LABELS,
                thresholds=THRESHOLDS,
                hidden_dim=HIDDEN_DIM,
                image_size=IMAGE_SIZE,
                patch_size=PATCH_SIZE,
                confidence_threshold=confidence_threshold,
                overlap_ratio=overlap_ratio
            )
            
            progress_bar.progress(0.8)
            status_text.text("✅ Selesai memproses!")
            progress_bar.progress(1.0)
            
            # Tampilkan hasil
            final_detections = results['final_detections']
            aggregated_scores = results['aggregated_scores']
            total_crops = results['total_crops']
            total_raw_detections = results['total_raw_detections']
            
            # Info statistik
            st.info(f"📊 **Statistik Deteksi:**\n"
                   f"- Total crops diproses: {total_crops}\n"
                   f"- Total deteksi mentah: {total_raw_detections}\n"
                   f"- Deteksi final: {len(final_detections)}")
            
            # Hasil deteksi final
            if final_detections:
                st.success("✅ **Label Terdeteksi (Multi-Scale):**")
                for detection in final_detections:
                    confidence_bar = "🟩" * int(detection['confidence'] * 10)
                    st.write(f"✅ **{detection['label']}** - {detection['confidence']:.2%} {confidence_bar}")
            else:
                # Check if OOD
                high_conf_count = sum(int(s >= t) for s, t in zip(aggregated_scores, THRESHOLDS))
                if high_conf_count < 1:
                    st.warning("🚫 Gambar tidak mengandung buah yang dikenali.")
                else:
                    st.warning("🚫 Tidak ada buah yang terdeteksi dengan confidence tinggi.")
            
            # Detail per crop (dalam expander)
            with st.expander(f"📋 Detail Hasil per Crop ({total_crops} crops)"):
                for i, crop_result in enumerate(results['crop_results']):
                    crop_info = crop_result['crop_info']
                    crop_detections = crop_result['detections']
                    
                    if crop_detections:  # Hanya tampilkan crop yang ada deteksi
                        st.write(f"**Crop {i}** ({crop_info['type']}, scale: {crop_info['scale']:.2f})")
                        for det in crop_detections:
                            st.write(f"  - {det['label']}: {det['confidence']:.2%}")
                        st.divider()
            
            # Tampilkan semua skor agregasi
            with st.expander("📊 Skor Agregasi Semua Kelas"):
                for label, score, thr in zip(LABELS, aggregated_scores, THRESHOLDS):
                    pass_thr = "✓" if score >= thr else "✗"
                    bar_length = int(score * 20)
                    score_bar = "█" * bar_length + "░" * (20 - bar_length)
                    st.write(f"`{score_bar}` **{label}**: {score:.2%} (thr: {thr:.2f}) {pass_thr}")
            
            # Visualisasi crops jika diminta
            if show_crops:
                with st.expander("🖼️ Visualisasi Sliding Window Crops"):
                    try:
                        # Generate ulang crops untuk visualisasi
                        crops = sliding_window_crop(
                            image, 
                            window_sizes=[200, 250, 300, 350],
                            overlap_ratio=overlap_ratio
                        )
                        
                        viz_image = visualize_crops(image, crops)
                        st.image(viz_image, caption=f"Sliding Window Crops (Total: {len(crops)-1} crops)", 
                                use_container_width=True)
                    except Exception as viz_error:
                        st.error(f"Gagal membuat visualisasi: {viz_error}")
            
            # Clear progress
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"❌ Error dalam multi-scale detection: {e}")
            st.code(traceback.format_exc())

# Footer info
st.markdown("---")
st.markdown("💡 **Tips:** Gunakan mode Multi-Scale untuk akurasi yang lebih baik, terutama untuk buah kecil atau di tepi gambar.")