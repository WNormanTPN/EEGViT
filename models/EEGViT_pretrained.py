import torch
from torch import nn
import transformers


class EEGViT_pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.preprocess = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        self.conv1 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(1, 36),
            stride=(1, 36),
            padding=(0, 2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        model_name = "google/vit-base-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (136, 14)})
        config.update({'patch_size': (8, 1)})

        model = transformers.ViTForImageClassification.from_pretrained(
            model_name, config=config, ignore_mismatched_sizes=True
        )
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(
            256, 768, kernel_size=(8, 1), stride=(8, 1), padding=(0, 0), groups=256
        )
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 1000, bias=True),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(1000, 2, bias=True)
        )
        self.ViT = model

    def forward(self, x):
        x = self.preprocess(x)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.ViT.forward(x).logits
        return x