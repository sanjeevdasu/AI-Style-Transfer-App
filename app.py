import streamlit as st
import tempfile
import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import torch.nn.functional as F

# Initialize Streamlit
st.set_page_config(page_title="🎨 AI Style Transfer", layout="wide", page_icon="🎨")

# Device setup
try:
    import torch_directml
    device = torch_directml.device() if torch_directml.is_available() else torch.device("cpu")
except ImportError:
    device = torch.device("cpu")

# Display device status
if hasattr(device, 'type'):
    if device.type == 'privateuseone':
        st.sidebar.success("✅ AMD GPU detected (DirectML backend)")
    elif device.type == 'cuda':
        st.sidebar.success("✅ NVIDIA GPU detected")
    else:
        st.sidebar.warning("⚠️ Using CPU (no compatible GPU found)")
else:
    st.sidebar.warning("⚠️ Using CPU (device type unknown)")

# Custom CSS
st.markdown("""
<style>
     .header {
        font-size: 2.5em;
        color: #4a4a4a;
        text-align: center;
        margin-bottom: 0.5em;
    }
    .subheader {
        font-size: 1.1em;
        color: #6a6a6a;
        text-align: center;
        margin-bottom: 2em;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5em 1em;
        width: 100%;
    }
    .stImage>img {
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    .progress-text {
        text-align: center;
        margin-top: 10px;
        font-size: 0.9em;
        color: #6a6a6a;
    }
</style>
""", unsafe_allow_html=True)

# Header with banner
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown('<p class="header">AI Style Transfer</p>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Transform your photos into artwork using AI</p>', unsafe_allow_html=True)
    try:
        banner_img = Image.open(r"C:\Users\R.SAI SANJEEV DASU\Downloads\art.jpg").resize((800, 200))
        st.image(banner_img, use_column_width=True)
    except:
        st.warning("Banner image not found. Using placeholder.")
        banner_img = Image.new('RGB', (800, 200), color='#4CAF50')
        st.image(banner_img, use_column_width=True)

# Style Transfer Functions
def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

def load_and_resize_image(image_file, target_size):
    """Load and resize image while maintaining aspect ratio"""
    img = Image.open(image_file).convert('RGB')
    
    # Calculate new dimensions maintaining aspect ratio
    width, height = img.size
    if width > height:
        new_width = target_size
        new_height = int(height * (target_size / width))
    else:
        new_height = target_size
        new_width = int(width * (target_size / height))
    
    img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Center crop to exact target size
    left = (new_width - target_size)/2
    top = (new_height - target_size)/2
    right = (new_width + target_size)/2
    bottom = (new_height + target_size)/2
    img = img.crop((left, top, right, bottom))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0).to(device)

def run_style_transfer(content_path, style_path, steps=300, 
                     style_weight=1e6, content_weight=1,
                     target_size=512, progress_callback=None):
    
    # Load and resize images to ensure same dimensions
    content = load_and_resize_image(content_path, target_size)
    style = load_and_resize_image(style_path, target_size)
    
    # Verify dimensions match
    if content.shape != style.shape:
        content = F.interpolate(content, size=(target_size, target_size), mode='bilinear')
        style = F.interpolate(style, size=(target_size, target_size), mode='bilinear')
    
    generated = content.clone().requires_grad_(True)
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
    
    # Layer configuration using numerical indices
    content_layer = '4'  # ReLU4_2
    style_layers = ['0', '5', '10', '19', '28']  # ReLU1_1, ReLU2_1, etc.
    
    # Freeze VGG parameters
    for param in vgg.parameters():
        param.requires_grad_(False)
    
    # Feature extraction
    def get_features(x, model, layers):
        features = {}
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[name] = x
        return features
    
    # Get target features
    content_features = get_features(content, vgg, [content_layer])
    style_features = get_features(style, vgg, style_layers)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    
    # Use Adam optimizer instead of LBFGS for more stable training
    optimizer = optim.Adam([generated], lr=0.05)
    
    for step in range(steps):
        def closure():
            # Clamp the image data to maintain valid range
            generated.data.clamp_(0, 1)
            
            optimizer.zero_grad()
            gen_features = get_features(generated, vgg, set(style_layers + [content_layer]))
            
            # Content loss
            content_loss = F.mse_loss(gen_features[content_layer], content_features[content_layer])
            
            # Style loss
            style_loss = 0
            for layer in style_layers:
                gen_gram = gram_matrix(gen_features[layer])
                style_loss += F.mse_loss(gen_gram, style_grams[layer])
            style_loss /= len(style_layers)
            
            total_loss = content_weight * content_loss + style_weight * style_loss
            total_loss.backward()
            
            if progress_callback:
                progress_callback(step + 1, steps)
            
            return total_loss
        
        optimizer.step(closure)
    
    # Final clamping
    with torch.no_grad():
        generated.data.clamp_(0, 1)
        output = generated.squeeze(0).cpu().permute(1, 2, 0).numpy()
        output = output * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        output = np.clip(output, 0, 1)
        return Image.fromarray((output * 255).astype(np.uint8))
    
  

# Settings sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    steps = st.slider("Processing Iterations", 10, 500, 100, 10)
    style_weight = st.slider("Style Strength", 1e4, 1e6, 1e5, 1e4)
    content_weight = st.slider("Content Preservation", 1, 100, 10, 1)
    target_size = st.selectbox("Output Size", [256, 512, 768], index=1)

# File upload section
st.markdown("## 📄 Upload Your Images")
col1, col2 = st.columns(2)

with col1:
    content_file = st.file_uploader("Content Image", type=["jpg", "jpeg", "png"])

with col2:
    style_file = st.file_uploader("Style Image", type=["jpg", "jpeg", "png"])

if content_file and style_file:
    try:
        # Validate images
        content_img = Image.open(content_file)
        style_img = Image.open(style_file)
        if min(content_img.size + style_img.size) < 64:
            st.error("❌ Images should be at least 64x64 pixels")
            st.stop()
            
        # Display inputs
        st.subheader("Input Images")
        col1, col2 = st.columns(2)
        with col1:
            st.image(content_file, caption="Content", use_column_width=True)
        with col2:
            st.image(style_file, caption="Style", use_column_width=True)

        if st.button("✨ Generate Artwork", type="primary"):
            with st.spinner("Creating artwork..."):
                # Create temp files
                temp_dir = tempfile.mkdtemp()
                content_path = os.path.join(temp_dir, "content.jpg")
                style_path = os.path.join(temp_dir, "style.jpg")
                output_path = os.path.join(temp_dir, "output.jpg")
                
                Image.open(content_file).save(content_path)
                Image.open(style_file).save(style_path)
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_callback(step, total):
                    progress = min(step / total, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: {step}/{total} iterations")
                
                # Run transfer
                output = run_style_transfer(
                    content_path,
                    style_path,
                    steps=steps,
                    style_weight=style_weight,
                    content_weight=content_weight,
                    target_size=target_size,
                    progress_callback=progress_callback
                )
                
                # Display result
                output.save(output_path)
                st.subheader("🎨 Your Styled Image")
                st.image(output, use_column_width=True)
                
                # Download button
                with open(output_path, "rb") as f:
                    st.download_button(
                        "💾 Download Result",
                        f,
                        file_name="styled_image.jpg",
                        mime="image/jpeg"
                    )
    except Exception as e:
        st.error(f"""
        ❌ Processing failed: {str(e)}
        
        Common solutions:
        1. Try smaller image sizes
        2. Reduce iteration count
        3. Restart the app
        """)
    finally:
        # Cleanup
        if 'temp_dir' in locals():
            for f in [content_path, style_path, output_path]:
                try:
                    if os.path.exists(f):
                        os.remove(f)
                except:
                    pass
            try:
                os.rmdir(temp_dir)
            except:
                pass

# Footer
st.markdown("---")
st.markdown('<p class="footer">AI Style Transfer App • Created with Streamlit and PyTorch</p>', unsafe_allow_html=True)