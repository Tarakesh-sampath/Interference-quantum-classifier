import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class QuantumFeatureExtractor(nn.Module):
    """
    Lightweight CNN to extract features for the Quantum Classifier.
    Maps 96x96x3 images to a small feature vector (e.g., 16D).
    """
    def __init__(self, feature_dim=16):
        super(QuantumFeatureExtractor, self).__init__()
        # Use a pre-trained ResNet18 as a robust backbone
        # We replace the final fully-connected layer to match our feature_dim
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # PCam images are 96x96, ResNet usually likes 224x224, but it works on 96x96
        # The output of ResNet18's average pool is 512
        num_filters = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_filters, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
            nn.Sigmoid() # Sigmoid helps scale features to [0, 1] for Amplitude Encoding
        )

    def forward(self, x):
        return self.backbone(x)

if __name__ == "__main__":
    # Fast test
    model = QuantumFeatureExtractor(feature_dim=16)
    dummy_input = torch.randn(1, 3, 96, 96)
    output = model(dummy_input)
    print(f"Feature Extractor Test:\nInput Shape: {dummy_input.shape}\nOutput Feature Shape: {output.shape}")
    print(f"Sample Features: {output[0][:5]}...")
