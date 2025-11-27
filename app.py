import streamlit as st
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image, ImageDraw, ImageFont
import os
import time
import json
import io
import re

# Model configuration
MODEL_ID = 'stabilityai/stable-diffusion-xl-base-1.0'

# Set page configuration
st.set_page_config(
    page_title="AI-Powered Image Generator (SDXL)",
    layout="wide"
)


def check_content_filter(prompt, negative_prompt):
    """
    Check if prompt or negative prompt contains inappropriate content.
    
    Args:
        prompt: Main prompt text
        negative_prompt: Negative prompt text
    
    Returns:
        tuple: (is_blocked: bool, blocked_keyword: str or None)
    """
    # List of inappropriate keywords (case-insensitive)
    blocked_keywords = [
        'pornography', 'porn', 'explicit', 'nude', 'nudity',
        'violence', 'violent', 'gore', 'blood',
        'hate', 'hateful', 'discrimination',
        'child abuse', 'abuse', 'illegal'
    ]
    
    # Combine prompts for checking
    combined_text = f"{prompt} {negative_prompt}".lower()
    
    for keyword in blocked_keywords:
        if keyword.lower() in combined_text:
            return True, keyword
    
    return False, None


def save_image_with_metadata(image: Image.Image, prompt: str, negative_prompt: str, style: str, index: int, device: str):
    """
    Save image with watermark and create corresponding metadata JSON file.
    
    Args:
        image: PIL Image object to save
        prompt: Original prompt text
        negative_prompt: Negative prompt text
        style: Selected style guidance
        index: Image index number
        device: Device used for generation (cuda/cpu)
    
    Returns:
        tuple: (image_path: str, metadata_path: str)
    """
    # Generate timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create folder structure
    output_dir = "generated_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create clean filename from prompt (truncate and sanitize)
    # Remove special characters and limit length
    clean_prompt = re.sub(r'[^\w\s-]', '', prompt)[:30].strip().replace(' ', '_')
    if not clean_prompt:
        clean_prompt = "image"
    
    filename_base = f"{timestamp}_{clean_prompt}_{index:02d}"
    image_path = os.path.join(output_dir, f"{filename_base}.png")
    metadata_path = os.path.join(output_dir, f"{filename_base}.json")
    
    # Add watermark to image
    watermarked_image = image.copy()
    
    # Convert to RGBA for overlay operations
    if watermarked_image.mode != 'RGBA':
        watermarked_image = watermarked_image.convert('RGBA')
    
    # Create overlay for watermark
    overlay = Image.new('RGBA', watermarked_image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Try to use a default font, fallback to basic if not available
    try:
        # Try to use a system font
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
    
    watermark_text = "AI Generated - Talrn Task"
    
    # Get text dimensions
    bbox = draw.textbbox((0, 0), watermark_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Position watermark in bottom-right corner with padding
    img_width, img_height = watermarked_image.size
    x = img_width - text_width - 10
    y = img_height - text_height - 10
    
    # Draw semi-transparent background for watermark
    padding = 5
    draw.rectangle(
        [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
        fill=(0, 0, 0, 180)  # Semi-transparent black (RGBA)
    )
    
    # Draw watermark text
    draw.text((x, y), watermark_text, fill=(255, 255, 255, 255), font=font)
    
    # Composite overlay onto image
    watermarked_image = Image.alpha_composite(watermarked_image, overlay)
    
    # Convert back to RGB for saving (PNG supports RGB)
    watermarked_image = watermarked_image.convert('RGB')
    
    # Save watermarked image
    watermarked_image.save(image_path, "PNG")
    
    # Create and save metadata JSON
    metadata = {
        "original_prompt": prompt,
        "negative_prompt": negative_prompt,
        "style": style,
        "timestamp": timestamp,
        "generation_device": device,
        "model_id": MODEL_ID,
        "image_index": index,
        "image_path": image_path
    }
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return image_path, metadata_path


@st.cache_resource
def load_model():
    """
    Load the Stable Diffusion XL model with caching.
    Downloads the model on first run.
    """
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        # Load with FP16 optimization for CUDA
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        pipeline = pipeline.to(device)
    else:
        # Standard model for CPU
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID,
            use_safetensors=True
        )
        pipeline = pipeline.to(device)
    
    return pipeline


def generate_image(pipeline, prompt, negative_prompt, style, num_images):
    """
    Generate images using the Stable Diffusion XL pipeline.
    
    Args:
        pipeline: The loaded Stable Diffusion XL pipeline
        prompt: Main prompt text
        negative_prompt: Negative prompt text
        style: Selected style guidance
        num_images: Number of images to generate
    
    Returns:
        List of PIL Image objects
    """
    # Prompt engineering: Add descriptive tags
    enhanced_prompt = prompt + ", 4K, cinematic lighting, highly detailed, professional photography"
    
    # Adjust prompt based on style
    if style == "Photorealistic":
        enhanced_prompt = f"a hyper-realistic photograph of {prompt}, 4K, cinematic lighting, highly detailed, professional photography"
    elif style == "Artistic (Digital Painting)":
        enhanced_prompt = f"a hyper-detailed digital painting of {prompt}, 4K, cinematic lighting, highly detailed, artistic, masterpiece"
    elif style == "Cartoon (Cell-Shaded)":
        enhanced_prompt = f"a cell-shaded cartoon style illustration of {prompt}, vibrant colors, clean lines, 4K, highly detailed"
    
    # Generate images
    images = pipeline(
        prompt=enhanced_prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        num_images_per_prompt=num_images,
        num_inference_steps=25,
        guidance_scale=7.5
    ).images
    
    return images


# Page title
st.title("AI-Powered Image Generator (SDXL)")

# Sidebar for user inputs
with st.sidebar:
    st.header("Image Generation Settings")
    
    # Large text area for main prompt
    prompt = st.text_area(
        "Prompt",
        height=150,
        placeholder="Enter your image generation prompt here...",
        help="Describe the image you want to generate"
    )
    
    # Smaller text area for negative prompt
    negative_prompt = st.text_area(
        "Negative Prompt",
        height=100,
        placeholder="Enter what you don't want in the image...",
        help="Specify elements to avoid in the generated image"
    )
    
    # Style guidance selectbox
    style_guidance = st.selectbox(
        "Style Guidance",
        options=["Photorealistic", "Artistic (Digital Painting)", "Cartoon (Cell-Shaded)"],
        help="Select the style for image generation"
    )
    
    # Number of images input
    num_images = st.number_input(
        "Number of Images",
        min_value=1,
        max_value=4,
        value=1,
        step=1,
        help="Select how many images to generate (1-4)"
    )

# Main area
st.header("Generate Your Image")

# Generate button
generate_button = st.button("Generate Image(s)", type="primary", use_container_width=True)

# Placeholder for progress and results
if generate_button:
    if prompt:
        # Content filtering (Ethical AI)
        is_blocked, blocked_keyword = check_content_filter(prompt, negative_prompt)
        if is_blocked:
            st.error(f"Content filter blocked: The prompt contains inappropriate content related to '{blocked_keyword}'. Please modify your prompt and try again.")
            st.stop()
        
        try:
            # Start timing
            start_time = time.time()
            
            # Load model (cached, downloads on first run)
            with st.spinner("Loading model... This may take a while on first run."):
                pipeline = load_model()
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Generate images
            with st.spinner("Generating image(s)... Please wait."):
                images = generate_image(pipeline, prompt, negative_prompt, style_guidance, num_images)
            
            # Calculate generation time
            generation_time = time.time() - start_time
            
            # Save images with metadata
            saved_paths = []
            for idx, image in enumerate(images):
                image_path, metadata_path = save_image_with_metadata(
                    image, prompt, negative_prompt, style_guidance, idx + 1, device
                )
                saved_paths.append((image_path, metadata_path))
            
            # Display success message with time
            st.success(f"Successfully generated {len(images)} image(s)!")
            st.info(f"⏱️ Total generation time: {generation_time:.2f} seconds")
            
            # Display images in columns with download buttons
            cols = st.columns(min(num_images, 2))  # Max 2 columns
            for idx, image in enumerate(images):
                with cols[idx % 2]:
                    st.image(image, caption=f"Generated Image {idx + 1}", use_container_width=True)
                    
                    # Download buttons
                    col1, col2 = st.columns(2)
                    
                    # Download PNG button
                    with col1:
                        # Read the saved PNG file
                        with open(saved_paths[idx][0], 'rb') as f:
                            png_bytes = f.read()
                        
                        st.download_button(
                            label="📥 Download PNG",
                            data=png_bytes,
                            file_name=os.path.basename(saved_paths[idx][0]),
                            mime="image/png",
                            use_container_width=True
                        )
                    
                    # Download JPEG button
                    with col2:
                        # Convert image to JPEG bytes
                        jpeg_buffer = io.BytesIO()
                        image.save(jpeg_buffer, format="JPEG", quality=95)
                        jpeg_bytes = jpeg_buffer.getvalue()
                        
                        jpeg_filename = os.path.basename(saved_paths[idx][0]).replace('.png', '.jpg')
                        st.download_button(
                            label="📥 Download JPEG",
                            data=jpeg_bytes,
                            file_name=jpeg_filename,
                            mime="image/jpeg",
                            use_container_width=True
                        )
        
        except Exception as e:
            st.error(f"An error occurred during image generation: {str(e)}")
            st.info("If this is your first run, the model is being downloaded. Please wait and try again.")
    else:
        st.warning("Please enter a prompt to generate an image.")

# Placeholder area for displaying images (when not generating)
if not generate_button:
    st.info("Configure your settings in the sidebar and click 'Generate Image(s)' to create your images.")

