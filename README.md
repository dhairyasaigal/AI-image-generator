# 🎨 AI-Powered Image Generator (SDXL)
### **ML Internship Task Assessment | Talrn.com**

This repository contains the complete solution for the **Talrn.com ML Internship Selection Task**: developing an **AI-Powered Text-to-Image Generation System**.

The system is built using **Streamlit** and the open-source **Stable Diffusion XL (SDXL)** model to convert descriptive text prompts into high-quality images—fulfilling all requirements for advanced features, performance optimization, and ethical use.

---

## 🚀 Key Features and Requirements Met

| Feature Area | Implementation Detail | Status |
|--------------|------------------------|--------|
| **Model & Setup** | Stable Diffusion XL 1.0 on PyTorch; GPU Optimization (FP16, CPU Offloading) with CPU fallback | ✅ Complete |
| **User Interface** | Streamlit UI with adjustable parameters | ✅ Complete |
| **Image Quality** | Prompt Engineering + Adjustable Negative Prompts | ✅ Complete |
| **Storage & Export** | Saves outputs as PNG + Exports metadata as JSON; In-app PNG/JPEG downloads | ✅ Complete |
| **Ethical AI** | Content Filtering + Persistent Watermark | ✅ Complete |

---

## ⚙️ Technology Stack and Architecture

### **Technology Stack**
- **Deep Learning Framework:** PyTorch  
- **Generative Model:** Stable Diffusion XL (SDXL) Base 1.0  
- **Model Pipeline:** Hugging Face *diffusers*  
- **Web Interface:** Streamlit  
- **Utilities:** Pillow (Watermarking), JSON, OS  

### **Architecture Overview**
The system runs as a **cached web service**:
- The Streamlit UI handles inputs.
- `load_model()` (cached using `@st.cache_resource`) loads the SDXL model once, optimizing:
  - VRAM usage  
  - CPU offloading  
- `generate_image()` handles prompt processing + inference.

---

## 💻 Hardware Requirements and Documentation

### 🥇 **GPU Implementation Path (Recommended)**
Utilizes VRAM optimizations for fast generation.

- **Hardware:** NVIDIA GPU (8 GB VRAM+) — e.g., RTX 3060/4060  
- **Optimizations:**  
  - `torch_dtype=torch.float16`  
  - `pipeline.enable_model_cpu_offload()`  
- **Speed:** ~30 sec–2 min per image (based on prompt complexity)

### 🥈 **CPU Fallback Path**
Activated automatically when `torch.cuda.is_available() == False`.

- **Hardware:** 16 GB RAM recommended (min 8 GB)  
- **Speed:** 5–20+ minutes per image  

---

## 🛠️ Setup and Installation

### **1. Clone the Repository**
```bash
git clone https://github.com/dhairyasaigal/AI-image-generator.git
cd AI-image-generator
```
## **2.Create and Activate Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
```
# OR:
```bash
.venv\Scripts\activate         # Windows
```
## **3. Install Dependencies**
```bash
pip install -r requirements.txt
```
# Ensure correct PyTorch version based on CUDA setup.

## **4. Run the Application**
```bash
python -m streamlit run app.py
```

The application will launch at:
👉 http://localhost:8501

## 🖼️ Usage Instructions and Prompt Engineering

### **Usage Instructions**
- Enter your description inside the **Prompt** field.  
- *(Optional)* Add text in the **Negative Prompt** to exclude unwanted elements.  
- Choose a **Style Guidance** option (e.g., Artistic, Photorealistic, Cartoon).  
- Click **Generate Image** to produce the output.

---

### **Prompt Engineering Tips**
Improve the quality of generated images by focusing on:

- **Detail:**  
  Use descriptive nouns and adjectives.  
  *Example:* `"vintage, neon-lit motorcycle"` instead of `"a bike"`  

- **Art Styles:**  
  Add stylistic elements such as `"oil painting"`, `"cyberpunk"`, `"cinematic lighting"`  

- **Style Guidance Options:**  
  - **Photorealistic:** *a hyper-realistic photograph of...*  
  - **Artistic:** *a hyper-detailed digital painting of...*  
  - **Cartoon:** *a cell-shaded cartoon illustration of...*

---

## 🔒 Ethical AI Use and Data Persistence

### **Content Filtering**
The `check_content_filter()` function blocks prompts containing:
- Violence  
- Explicit content  
- Hate speech  

This ensures safe, ethical, and compliant AI usage.

---

### **Watermarking & Metadata**
- All images are watermarked with: **"AI Generated - Talrn Task"**  
- A `.json` metadata file is generated for each image, containing:
  - Prompt  
  - Negative prompt  
  - Parameter settings  
  - Timestamp  

All generated assets (PNG + JSON) are saved in the **`/generated_images/`** directory.

---
## 🎥 Demo Video

Watch the full demo of the project here:  
👉 **YouTube Link:** https://youtu.be/LMu8JGIUVkc

---

## 📄 License
This project was developed as part of the **Talrn ML Internship Assessment** and is free to review and reproduce for evaluation purposes.

---

## ⭐ Acknowledgments
Special thanks to:
- **Talrn.com** for the assignment  
- **Hugging Face** for the `diffusers` library  
- **Stability AI** for the SDXL model  

---
