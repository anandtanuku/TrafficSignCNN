import torch
import torch.nn as nn

class TrafficSignCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels = 3,
                out_channels = 32,
                kernel_size = 3,
                padding = 1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels = 32,
                out_channels = 64,
                kernel_size = 3,
                padding = 1
            ),
            
            nn.ReLU(),

            nn.MaxPool2d(2)

        )

        self.classifier = nn.Sequential(


            nn.Linear(64 * 8 * 8, 128),

            nn.ReLU(),

            nn.Linear(128, 43)

        )

    def forward(self, x):

        x = self.features(x)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x