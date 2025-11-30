"""
Model architecture for ear teacher using pretrained ConvNeXt backbone.
"""
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny


class EarTeacherModel(nn.Module):
    """
    Self-supervised ear teacher model using ConvNeXt-Tiny backbone.

    This model learns intricate ear details through reconstruction
    for later distillation to detector and landmarker models.
    """

    def __init__(
        self,
        pretrained_path: str = "models/convnext_tiny_22k_224.pth",
        embedding_dim: int = 768,
        projection_dim: int = 256,
    ):
        """
        Args:
            pretrained_path: Path to pretrained ConvNeXt weights
            embedding_dim: Dimension of ConvNeXt-Tiny output (768 for tiny)
            projection_dim: Dimension of projection head output
        """
        super().__init__()

        # Load ConvNeXt-Tiny backbone
        self.backbone = convnext_tiny(weights=None)

        # Load pretrained weights
        state_dict = torch.load(pretrained_path, map_location='cpu')

        # Handle different state dict formats
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # Load weights (may have partial match)
        self.backbone.load_state_dict(state_dict, strict=False)

        # Replace classifier head with identity to get embeddings
        # ConvNeXt classifier is a Sequential with LayerNorm + Flatten + Linear
        # We want to keep the normalization and flattening but remove the final linear layer
        self.backbone.classifier[-1] = nn.Identity()

        # Projection head for self-supervised learning
        # This maps embeddings to a lower dimensional space for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, projection_dim),
        )

        # Prediction head (for methods like SimSiam/BYOL)
        self.prediction_head = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.GELU(),
            nn.Linear(projection_dim // 2, projection_dim),
        )

        # Reconstruction decoder
        # Upsamples embeddings back to image space
        self.decoder = nn.Sequential(
            # From embedding_dim to 256 spatial features (7x7)
            nn.Linear(embedding_dim, 256 * 7 * 7),
            nn.GELU(),
            nn.Unflatten(1, (256, 7, 7)),

            # Upsample: 7x7 -> 14x14
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            # Upsample: 14x14 -> 28x28
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            # Upsample: 28x28 -> 56x56
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),

            # Upsample: 56x56 -> 112x112
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),

            # Upsample: 112x112 -> 224x224
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # Output in range [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            Dictionary containing:
                - embeddings: Backbone features [B, embedding_dim]
                - projections: Projected features [B, projection_dim]
                - predictions: Predicted features [B, projection_dim]
                - reconstruction: Reconstructed images [B, 3, H, W]
        """
        # Extract features from backbone
        embeddings = self.backbone(x)  # [B, 768]

        # Project to lower dimensional space
        projections = self.projection_head(embeddings)  # [B, projection_dim]

        # Predict (for SimSiam-style learning)
        predictions = self.prediction_head(projections)  # [B, projection_dim]

        # Reconstruct image
        reconstruction = self.decoder(embeddings)  # [B, 3, 224, 224]

        return {
            'embeddings': embeddings,
            'projections': projections,
            'predictions': predictions,
            'reconstruction': reconstruction,
        }

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get backbone embeddings only (for inference/feature extraction)."""
        return self.backbone(x)


def create_ear_teacher_model(
    pretrained_path: str = "models/convnext_tiny_22k_224.pth",
    embedding_dim: int = 768,
    projection_dim: int = 256,
) -> EarTeacherModel:
    """
    Factory function to create ear teacher model.

    Args:
        pretrained_path: Path to pretrained ConvNeXt weights
        embedding_dim: Dimension of ConvNeXt-Tiny output
        projection_dim: Dimension of projection head output

    Returns:
        EarTeacherModel instance
    """
    return EarTeacherModel(
        pretrained_path=pretrained_path,
        embedding_dim=embedding_dim,
        projection_dim=projection_dim,
    )
