import torch
from torch import nn


class StrategyNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(StrategyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.softmax = nn.Softmax(dim=-1)
        self.initialize_weights()  # Custom initialization

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

    def initialize_weights(self):
        """Custom weight initialization to set initial output probabilities."""
        with torch.no_grad():
            # Set weights to small random values
            nn.init.normal_(self.fc1.weight, mean=0, std=0.01)
            nn.init.normal_(self.fc2.weight, mean=0, std=0.01)
            nn.init.normal_(self.fc3.weight, mean=0, std=0.01)

            # Adjust bias of the final layer to achieve desired probabilities
            target_probs = torch.tensor([0.80, 0.05, 0.15])  # Desired initial probabilities
            log_probs = torch.log(target_probs)  # Logits that would produce these probabilities
            self.fc3.bias[:] = log_probs
