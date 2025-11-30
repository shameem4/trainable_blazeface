"""
Model architecture for ear teacher using pretrained ConvNeXt backbone.
"""
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny


class EarTeacherModel(nn.Module):
    """
    Self-supervised ear teacher model using ConvNeXt-Tiny backbone.

    This model learns intricate ear details for later distillation to
    detector and landmarker models.
    """

    def __init__(
        self,
        pretrained_path: str = "models/convnext_tiny_22k_224.pth",
        embedding_dim: int = 768,
        projection_dim: int = 256,
        freeze_backbone: bool = False,
    ):
        """
        Args:
            pretrained_path: Path to pretrained ConvNeXt weights
            embedding_dim: Dimension of ConvNeXt-Tiny output (768 for tiny)
            projection_dim: Dimension of projection head output
            freeze_backbone: Whether to freeze backbone weights initially
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
        self.backbone.classifier = nn.Identity()

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

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
        """
        # Extract features from backbone
        embeddings = self.backbone(x)  # [B, 768]

        # Project to lower dimensional space
        projections = self.projection_head(embeddings)  # [B, projection_dim]

        # Predict (for SimSiam-style learning)
        predictions = self.prediction_head(projections)  # [B, projection_dim]

        return {
            'embeddings': embeddings,
            'projections': projections,
            'predictions': predictions,
        }

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get backbone embeddings only (for inference/feature extraction)."""
        return self.backbone(x)

    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def create_ear_teacher_model(
    pretrained_path: str = "models/convnext_tiny_22k_224.pth",
    embedding_dim: int = 768,
    projection_dim: int = 256,
    freeze_backbone: bool = False,
) -> EarTeacherModel:
    """
    Factory function to create ear teacher model.

    Args:
        pretrained_path: Path to pretrained ConvNeXt weights
        embedding_dim: Dimension of ConvNeXt-Tiny output
        projection_dim: Dimension of projection head output
        freeze_backbone: Whether to freeze backbone initially

    Returns:
        EarTeacherModel instance
    """
    return EarTeacherModel(
        pretrained_path=pretrained_path,
        embedding_dim=embedding_dim,
        projection_dim=projection_dim,
        freeze_backbone=freeze_backbone,
    )
