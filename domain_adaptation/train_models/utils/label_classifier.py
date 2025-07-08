from torch import nn


class LabelClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        # self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        return self.classifier(x)
