import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['CustomAlex']

class CustomAlex(nn.Module):

    def __init__(self,model_id, num_classes=1000):
        super(CustomAlex,self).__init__()
        self.model_id = model_id

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        if model_id == "A":
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 1024),
            )

        if model_id != "A":
            # self.classifier = nn.Sequential(
            #     nn.Dropout(),
            #     nn.Linear(256 * 6 * 6, 4096),
            #     nn.ReLU(inplace=True),
            #     nn.Dropout(),
            #     nn.Linear(4096,4096),
            # )
            # Define each layer and then define loss
            # The layers to pass for B and C, D are the same
            # With an extra FC-skip layer for D 
            self.fc6 = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(256 * 6 * 6, 4096),
                )
            self.fc7 = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                )
            self.fc8 = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                )


        if model_id == "D":
            # self.classifier = nn.Sequential(
            #     nn.Dropout(),
            #     nn.Linear(256 * 6 * 6, 4096),
            # )
            self.fc_skip = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(256 * 6 * 6, 4096),
                )
        

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)

        # The Linear layers will be different for different losses
        if self.model_id == "A":
            x = self.classifier(x)
            return x

        elif self.model_id == "B":
            x = self.fc6(x)
            penultimate = self.fc7(x)
            output = self.fc8(x)
            return penultimate, output

        elif self.model_id == "C":
            fc6 = self.fc6(x)
            fc7 = self.fc7(fc6)
            fc8 = self.fc8(fc7)
            return fc6, fc7, fc8

        else:
            fc6 = self.fc6(x)
            skip_fc6 = self.skip_fc6(x)
            fc7 = self.fc7(fc6 + skip_fc6)
            fc8 = self.fc8(fc7)
            return fc_skip, fc7, fc8
        return temp
