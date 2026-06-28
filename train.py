import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloaders
from models import get_baseline_cnn, get_mobilenet_v2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, test_loader = get_dataloaders('Alzheimer_Dataset_Details.csv', 'Alzheimer_Split/')

model = get_mobilenet_v2(num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

print("Memulai Training Model...")
for epoch in range(15): 
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

torch.save(model.state_dict(), 'mobilenet_v2_model.pth')
