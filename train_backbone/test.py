import torch
import torchvision.models as models

model = models.resnet18(pretrained=False)
model.load_state_dict(torch.load('best_model_state.pth'))

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