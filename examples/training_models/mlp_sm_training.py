import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchebm.core import BaseEnergyFunction
from torchebm.losses import ScoreMatching
from torchebm.datasets import GaussianMixtureDataset


# A trainable EBM
class MLPEnergy(BaseEnergyFunction):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # a scalar value


# Setup model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
energy_fn = MLPEnergy(input_dim=2).to(device)

# Choose score matching variant
sm_loss_fn = ScoreMatching(
    energy_function=energy_fn,
    hessian_method="hutchinson",  # More efficient for higher dimensions
    hutchinson_samples=5,
    device=device,
)

optimizer = optim.Adam(energy_fn.parameters(), lr=0.001)

# Setup data
mixture_dataset = GaussianMixtureDataset(
    n_samples=500, n_components=4, std=0.1, seed=123
).get_data()
dataloader = DataLoader(mixture_dataset, batch_size=32, shuffle=True)

# Training Loop
for epoch in range(10):
    epoch_loss = 0.0
    for batch_data in dataloader:
        batch_data = batch_data.to(device)

        optimizer.zero_grad()
        loss = sm_loss_fn(batch_data)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1}/10, Loss: {avg_loss:.6f}")
