"""
Loss functions for ear teacher model.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """
    ArcFace loss for learning discriminative embeddings.

    Reference: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    https://arxiv.org/abs/1801.07698

    This is particularly useful for learning embeddings that can distinguish
    between different individuals' ears.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        num_classes: int = 1000,
        margin: float = 0.5,
        scale: float = 64.0,
    ):
        """
        Args:
            embedding_dim: Dimension of input embeddings
            num_classes: Number of identity classes (pseudo-classes for self-supervised)
            margin: Angular margin penalty (m in paper, default: 0.5)
            scale: Feature scale (s in paper, default: 64.0)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # Weight matrix for classification
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            embeddings: Input embeddings [B, embedding_dim]
            labels: Target labels [B]

        Returns:
            Loss value
        """
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarity
        cosine = F.linear(embeddings, weight)  # [B, num_classes]

        # Compute sine from cosine
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Easy margin: if cos(theta) > cos(pi - m), use phi, else use cos(theta) - mm
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encoding
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Apply margin only to target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        # Cross-entropy loss
        return F.cross_entropy(output, labels)


class CosFaceLoss(nn.Module):
    """
    CosFace loss (also known as Large Margin Cosine Loss).

    Reference: "CosFace: Large Margin Cosine Loss for Deep Face Recognition"
    https://arxiv.org/abs/1801.09414

    Simpler than ArcFace but also effective for learning discriminative embeddings.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        num_classes: int = 1000,
        margin: float = 0.35,
        scale: float = 64.0,
    ):
        """
        Args:
            embedding_dim: Dimension of input embeddings
            num_classes: Number of identity classes
            margin: Cosine margin (m in paper, default: 0.35)
            scale: Feature scale (s in paper, default: 64.0)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # Weight matrix for classification
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            embeddings: Input embeddings [B, embedding_dim]
            labels: Target labels [B]

        Returns:
            Loss value
        """
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarity
        cosine = F.linear(embeddings, weight)  # [B, num_classes]

        # One-hot encoding
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Apply margin to target class: cos(theta) - m
        output = cosine - (one_hot * self.margin)
        output *= self.scale

        # Cross-entropy loss
        return F.cross_entropy(output, labels)
