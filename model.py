import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from unet import UNet

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = UNet(n_channels=1, n_classes=1, bilinear=True)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # same as training
])

def predict(image):
    image = image.convert("L")  # grayscale
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        pred = torch.sigmoid(output).cpu().numpy()[0, 0]
        mask = (pred > 0.5).astype(np.uint8) * 255
    return Image.fromarray(mask)
