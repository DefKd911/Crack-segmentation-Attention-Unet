# 🧠 Attention UNet for Crack Segmentation

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-blue)](https://huggingface.co/spaces/kokSD/attentionUnet_crack_segmentation)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DefKd911/Crack-segmentation-Attention-Unet/blob/main/notebooks/training_unet.ipynb)

🚧 **Automatic crack detection in concrete structures** using **Attention UNet** for semantic segmentation.  
This project applies **deep learning in computer vision** to enhance **infrastructure safety monitoring** and preventive maintenance.

🌐 **Live Demo** → [Try it on Hugging Face Spaces](https://huggingface.co/spaces/kokSD/attentionUnet_crack_segmentation)  

---

## ✨ Key Highlights

- 🏗️ Implements **Attention UNet**—a step up from the classical UNet model.
- 🎯 Segments **cracks in concrete surfaces** with high accuracy.
- 🌐 Deployed as an interactive web app using **Gradio + Hugging Face Spaces**.
- 📓 Includes a Colab training notebook for reproducibility.
- ⚡ Modular, easy-to-understand codebase for training, inference, and deployment.

---

## 🏗️ Architecture

### 🔹 Why UNet?

UNet is a robust encoder-decoder architecture designed for biomedical image segmentation, with skip connections that help preserve spatial information—making it ideal for pixel-wise segmentation tasks.

### 🔹 Why Attention UNet?

While standard UNet treats all encoder features equally, **Attention UNet** introduces **Attention Gates (AGs)** that help the model focus on the most relevant features (such as cracks), filtering out background noise and irrelevant regions.

**Architecture Overview:**
1. **Encoder (downsampling):** Extracts hierarchical feature maps from the input image.
2. **Attention Gates:** Apply spatial attention to highlight crack-relevant features.
3. **Decoder (upsampling):** Reconstructs the segmentation mask, fusing features from the encoder.
4. **Output Layer:** Produces a binary mask (crack vs. no crack).

![Attention UNet](https://i.ibb.co/3N7Nn8z/attention-unet-arch.png)  
*Visual: Attention UNet with attention gates in skip connections*

---

## 🛠️ Use Cases

- 🛣️ **Road Infrastructure:** Detect cracks in highways, bridges, and flyovers.
- 🏢 **Civil Engineering:** Monitor structural health of buildings.
- 🏗️ **Smart Cities:** Automated drone inspection of concrete surfaces.
- 🚆 **Railways & Runways:** Early crack detection for public safety.

---

## 💡 What Makes This Project Stand Out?

1. **Attention-driven segmentation:** Outperforms vanilla UNet by focusing on fine cracks and filtering noise.
2. **End-to-end pipeline:** From training (Colab) to deployment (Hugging Face Spaces via Gradio).
3. **Real-time, lightweight deployment:** The web demo runs in your browser.
4. **Modular design:** Clean separation between `app.py`, `model.py`, and `unet.py`.

---

## 📂 Repository Structure

```
Crack-segmentation-Attention-Unet/
├── app.py                # Gradio web app
├── model.py              # Model loader + prediction logic
├── unet.py               # Attention UNet architecture
├── requirements.txt      # Dependencies
├── weights/
│   └── model.pth         # Pretrained weights
├── notebooks/
│   └── training_unet.ipynb # Training notebook
├── examples/
│   ├── crack1.jpg
│   ├── crack2.jpg
│   ├── output1.png
│   └── output2.png
└── README.md
```

---

## 🚀 Quick Start

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/DefKd911/Crack-segmentation-Attention-Unet.git
cd Crack-segmentation-Attention-Unet
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Locally

```bash
python app.py
```
This launches a local Gradio app. Open the provided URL in your browser to use the crack segmentation demo.

---

## 🧪 Example Results

| Input Image | Predicted Crack Mask |
| ----------- | ------------------- |
| ![crack1.jpg](examples/crack1.jpg) | ![output1.png](examples/output1.png) |
| ![crack2.jpg](examples/crack2.jpg) | ![output2.png](examples/output2.png) |

---

## 📓 Training Details

- **Framework:** PyTorch
- **Dataset:** Public Concrete Crack Dataset (e.g., [Kaggle SDNET2018](https://www.kaggle.com/datasets/snehilsanyal/concrete-crack-images-for-classification))
- **Image Size:** 256 x 256
- **Loss Function:** Binary Cross Entropy (BCE)
- **Optimizer:** Adam (lr=1e-4)
- **Hardware:** Trained on Google Colab GPU (Tesla T4)
- **Epochs:** ~50

Full training notebook: [`notebooks/training_unet.ipynb`](notebooks/training_unet.ipynb)  
[Open in Colab](https://colab.research.google.com/github/DefKd911/Crack-segmentation-Attention-Unet/blob/main/notebooks/training_unet.ipynb)

---

## ⚙️ Tech Stack

- **PyTorch** – deep learning framework
- **Gradio** – web demo UI
- **Hugging Face Spaces** – demo deployment
- **Google Colab** – training environment

---

## 📚 References

- [UNet: Convolutional Networks for Biomedical Image Segmentation (2015)](https://arxiv.org/abs/1505.04597)
- [Attention UNet: Learning Where to Look for the Pancreas (2018)](https://arxiv.org/abs/1804.03999)
- [Concrete Crack Dataset - SDNET2018](https://www.kaggle.com/datasets/snehilsanyal/concrete-crack-images-for-classification)

---

## 🌍 Impact & Future Work

Detecting cracks in concrete infrastructure is vital for public safety and cost-effective maintenance.  
**Future directions:**
- 🔬 Multi-class segmentation (detect multiple defect types)
- 📊 Integration with drone-based inspection systems
- 🌐 Real-time edge deployment (Raspberry Pi, Jetson Nano, etc.)

---

**Star this repo if you find it useful!**  
Feel free to open issues or pull requests to improve or extend the project.
