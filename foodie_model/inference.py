import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from model import ResNet50
from dataloader import FoodDataset

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

best_model_path = "foodie_model/resnetfoodie.pth"
food_classes = FoodDataset(root_dir="foodie_model/data/train", transform=None).classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet50(num_classes=len(food_classes)).to(device)
model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
model.eval()

def classify(img: Image.Image) -> str:
    if isinstance(img, np.ndarray): img = Image.fromarray(img)
    img = transform(img).unsqueeze(0).to(device)
    
    with torch.inference_mode():
        prediction = F.softmax(model(img), dim=1)
        index = torch.argmax(prediction, dim=1)
        return food_classes[index.item()].capitalize()