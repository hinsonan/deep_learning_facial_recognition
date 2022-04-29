from torch import nn
import torch.nn.functional as F

class EmotionalModel(nn.Module):

    def __init__(self, model_type) -> None:
        super(EmotionalModel,self).__init__()

        self.model_type = model_type

        if model_type == "vgg":
            self.model = nn.Sequential(
                nn.Conv2d(1,64,kernel_size=3,padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64,64,kernel_size=3,padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Conv2d(64,128,kernel_size=3,padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Conv2d(128,256,kernel_size=3,padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Conv2d(256,512,kernel_size=3,padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512,512,kernel_size=3,padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Conv2d(512,512,kernel_size=3,padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2)
            )
            self.flatten = nn.Flatten()
            self.classify = nn.Linear(8192,8)
        else:
            # Layer 0
            self.layer0 = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

            # Layer 1
            self.layer1 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )

            # Layer 2
            self.layer2a = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )

            self.layer2b = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128)
            )

            self.layer2_shortcut = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=1, stride=2),
                nn.BatchNorm2d(128)
            )

            # Layer 3
            self.layer3a = nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)
            )

            self.layer3b = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256)
            )

            self.layer3_shortcut = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=1, stride=2),
                nn.BatchNorm2d(256)
            )

            # Layer 4
            self.layer4a = nn.Sequential(
                    nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True)
            )

            self.layer4b = nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512)
            )

            self.layer4_shortcut = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=1, stride=2),
                nn.BatchNorm2d(512)
            )

            self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            self.flatten = nn.Flatten()
            self.classify = nn.Linear(in_features=512, out_features=8)

    def forward(self,x):
        if self.model_type == "vgg":
            out = self.model(x)
        else:
            out = self.layer0(x)
            out = self.layer1(out)
            d_sample = self.layer2_shortcut(out)
            out = self.layer2a(out) + d_sample
            out = self.layer2b(out)
            d_sample = self.layer3_shortcut(out)
            out = self.layer3a(out) + d_sample
            out = self.layer3b(out)
            d_sample = self.layer4_shortcut(out)
            out = self.layer4a(out) + d_sample
            out = self.layer4b(out)
            out = self.pool(out)

        out = self.flatten(out)
        out = self.classify(out)

        return out
