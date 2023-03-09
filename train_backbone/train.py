import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from data import HeroImage

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.backends import cudnn

# Quick settings
cudnn.benchmark = True
torch.seed = 42

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
])
train_dataset = HeroImage(transform=transform)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet18(pretrained=True)

# make model only train on last 2 layers
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512, 64)
for param in model.fc.parameters():
    param.requires_grad = True

# AdmS_criterion, CE_criterion = define_loss()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

def calculate_accuracy(outputs_list, targets_list):
    outputs = torch.cat(outputs_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    _, predictions = outputs.max(1)
    num_correct = (predictions == targets).sum()
    num_samples = predictions.size(0)
    acc = float(num_correct) / num_samples
    return acc

best_acc = 0.0

for epoch in range(300):
    losses = []
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        outputs = model(data)
        loss = criterion(outputs, targets)
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch}/{300}] Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")
            
    print(f"Epoch [{epoch}/{300}] Average Loss: {sum(losses)/len(losses):.4f}")
            
    scheduler.step()
    
    # calculate accuracy of model on training data
    model.eval()
    with torch.no_grad():
        outputs_list, targets_list = [], []
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device=device)
            targets = targets.to(device=device)
            outputs = model(data)
            
            outputs_list.append(outputs)
            targets_list.append(targets)
            
        train_acc = calculate_accuracy(outputs_list, targets_list)
        print(f"Train accuracy: {train_acc:.4f}")
            
        # save model if it has the best accuracy
        if train_acc > best_acc:
            best_acc = train_acc
            torch.save(model.state_dict(), 'best_model_state.pth')
            print(f"Saved model with accuracy: {train_acc:.4f}")
            
print("Best accuracy: ", best_acc)

