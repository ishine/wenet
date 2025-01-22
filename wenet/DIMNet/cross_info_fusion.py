import torch
from torch import nn
import torch.nn.functional as F

from wenet.utils.class_utils import WENET_NORM_CLASSES


class CrossInformationFusionModule(nn.Module):
    def __init__(
        self,
        acoustic_dim,
        accent_dim,
        normalize: bool = True,
        layer_norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
    ):
        """Cross Information Fusion Module

        Args:
            acoustic_dim: The feature dimension of the acoustic feature
            accent_dim: The feature dimension of the accent feature
        """
        super().__init__()
        # Acoustic features are directly mapped to the same dimension as accent features
        self.acoustic_to_k = nn.Linear(acoustic_dim, accent_dim)
        self.acoustic_to_v = nn.Linear(acoustic_dim, accent_dim)

        # Linear projection of accent features to Q
        self.accent_to_q = nn.Linear(accent_dim, accent_dim)

        # output from the first stage -> Q for the second stage
        self.intermediate_q = nn.Linear(accent_dim, accent_dim)

        # Accent features -> KV for the second stage
        self.accent_to_k2 = nn.Linear(accent_dim, accent_dim)
        self.accent_to_v2 = nn.Linear(accent_dim, accent_dim)

        self.relu = nn.ReLU()

        self.normalize = normalize
        self.norm = WENET_NORM_CLASSES[layer_norm_type](accent_dim, eps=norm_eps)

    def forward(
        self,
        acoustic: torch.Tensor,
        accent: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            acoustic (torch.Tensor): acoustic features (B, T, D)
            accent (torch.Tensor): accent features (B, T, C)
            mask (torch.Tensor): mask tensor (B, T)

        Returns:
            torch.Tensor: fused features (B, T, C)
        """
        # Acoustic features as KV, accent features as Q
        K1 = self.acoustic_to_k(acoustic)  # (B, T, C)
        V1 = self.acoustic_to_v(acoustic)  # (B, T, C)
        Q1 = self.accent_to_q(accent)  # (B, T, C)

        # Compute attention scores
        attention_scores1 = torch.matmul(Q1, K1.transpose(-2, -1)) / (
            K1.size(-1) ** 0.5
        )  # (B, T, T)
        if mask is not None:
            # Expand mask to (B, T, T)
            mask_expanded = mask.expand(-1, -1, attention_scores1.size(-1))  # (B, T, T)
            attention_scores1 = attention_scores1.masked_fill(
                ~mask_expanded, float("-inf")
            )  # Apply mask
        attention_weights1 = F.softmax(attention_scores1, dim=-1)  # (B, T, T)

        # Compute the output of the first stage
        output1 = torch.matmul(attention_weights1, V1)  # (B, T, C)
        output1_activated = self.relu(output1)

        # The output from the previous step as Q, accent features as KV
        Q2 = self.intermediate_q(output1_activated)  # (B, T, C)
        K2 = self.accent_to_k2(accent)  # (B, T, C)
        V2 = self.accent_to_v2(accent)  # (B, T, C)

        # Compute attention scores
        attention_scores2 = torch.matmul(Q2, K2.transpose(-2, -1)) / (
            K2.size(-1) ** 0.5
        )  # (B, T, T)
        if mask is not None:
            # Expand mask to (B, T, T)
            mask_expanded = mask.expand(-1, -1, attention_scores2.size(-1))  # (B, T, T)
            attention_scores2 = attention_scores2.masked_fill(
                ~mask_expanded, float("-inf")
            )  # Apply mask
        attention_weights2 = F.softmax(attention_scores2, dim=-1)  # (B, T, T)

        # Compute the output of the second stage
        output2 = torch.matmul(attention_weights2, V2)  # (B, T, C)
        output2_activated = self.relu(output2)

        if self.normalize:
            output2_activated = self.norm(output2_activated)

        return output2_activated
