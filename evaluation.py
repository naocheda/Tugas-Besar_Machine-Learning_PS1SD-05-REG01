import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from data_loader import get_dataloaders
from models import get_mobilenet_v2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, test_loader = get_dataloaders('Alzheimer_Dataset_Details.csv', 'Alzheimer_Split')

model = get_mobilenet_v2(num_classes=3).to(device)
model.load_state_dict(torch.load('mobilenet_v2_model.pth'))
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Non_Demented', 'Very_Mild_Demented', 'Mild_Demented']))
print(f"Overall Accuracy: {accuracy_score(y_true, y_pred)*100:.2f}%")
