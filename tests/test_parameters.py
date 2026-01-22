import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

model = SimpleModel()

print("Model Parameters:")
for name, param in model.named_parameters():
    print(f"Name: {name}, Shape: {param.shape}")

print(f"\nTotal number of parameters: {sum(p.numel() for p in model.parameters())}")

