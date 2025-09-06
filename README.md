#  Crack Segmentation - Attention U-Net

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-blue)](https://huggingface.co/spaces/kokSD/attentionUnet_crack_segmentation)

🚧 **Automatic crack detection in concrete structures** using **Attention UNet** for semantic segmentation.  
This project applies **deep learning in computer vision** to enhance **infrastructure safety monitoring** and preventive maintenance.

🌐 **Live Demo** → [Try it on Hugging Face Spaces](https://huggingface.co/spaces/kokSD/attentionUnet_crack_segmentation)  

<img width="1846" height="700" alt="image" src="https://github.com/user-attachments/assets/ef693f6c-6a4f-48e5-a521-0794ebfb7b47" />


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

<img width="850" height="487" alt="image" src="https://github.com/user-attachments/assets/7386dc8a-e635-4267-8f21-cc74d66a2e22" />

*Visual: Attention UNet with attention gates in skip connections*
---

## 🛠️ Use Cases

- 🛣️ **Road Infrastructure:** Detect cracks in highways, bridges, and flyovers.
- 🏢 **Civil Engineering:** Monitor structural health of buildings.
- 🏗️ **Smart Cities:** Automated drone inspection of concrete surfaces.
- 🚆 **Railways & Runways:** Early crack detection for public safety.

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



## 🧪 Example Results

<img width="1816" height="809" alt="image" src="https://github.com/user-attachments/assets/19ad54d4-fc41-48b9-afde-26399085ddc9" />


## 📓 Training Details

- **Framework:** PyTorch
- **Dataset:** Public Concrete Crack Dataset
- **Image Size:** 256 x 256
- **Loss Function:** Binary Cross Entropy (BCE)
- **Optimizer:** Adam (lr=1e-4)
- **Hardware:** Trained on Google Colab GPU (Tesla T4)
- **Epochs:** ~50


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
- 🌐 Real-time edge deployment (Raspberry Pi, Jetson Nano, etc.)

---

**Star this repo if you find it useful!**  
Feel free to open issues or pull requests to improve or extend the project.
