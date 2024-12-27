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


class KANLayer(nn.Module):
    def __init__(self, input_dim, num_inner_funcs=10):
        super().__init__()
        self.input_dim = input_dim
        self.num_inner_funcs = num_inner_funcs

        # Inner functions g_i for each input dimension
        self.inner_funcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, num_inner_funcs),
                nn.Tanh(),
                nn.Linear(num_inner_funcs, num_inner_funcs)
            )
            for _ in range(input_dim)
        ])

        # Outer function Φ
        self.outer_func = nn.Sequential(
            nn.Linear(input_dim * num_inner_funcs, num_inner_funcs),
            nn.Tanh(),
            nn.Linear(num_inner_funcs, 1)
        )

    def forward(self, x):
        # Apply inner functions to each input dimension
        inner_results = []
        for i in range(self.input_dim):
            x_i = x[:, i:i + 1]
            g_i = self.inner_funcs[i](x_i)
            inner_results.append(g_i)

        combined = torch.cat(inner_results, dim=1)
        return self.outer_func(combined)


class KAN(nn.Module):
    def __init__(self, input_size, output_size, num_inner_funcs=10):
        super().__init__()
        self.kan_layers = nn.ModuleList([
            KANLayer(input_size, num_inner_funcs)
            for _ in range(output_size)
        ])

    def forward(self, x):
        outputs = []
        for kan in self.kan_layers:
            out_i = kan(x)
            outputs.append(out_i)
        logits = torch.cat(outputs, dim=1)
        return logits
