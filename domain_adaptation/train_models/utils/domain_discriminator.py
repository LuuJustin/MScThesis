import torch.nn as nn

from train_models.utils.gradient_reversal_layer import grad_reverse


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=1)
        )

    def forward(self, x, lambda_):
        x = grad_reverse(x, lambda_)
        return self.model(x)
