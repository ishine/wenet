from typing import Tuple

import torch

from wenet.utils.class_utils import WENET_NORM_CLASSES


class LASASBlock(torch.nn.Module):
    def __init__(
        self,
        acoustic_dim: int,
        text_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
    ):
        """
        LASAS Module
        Args:
            acoustic_dim (int): Dimension of the input acoustic embedding.
            text_dim (int): Dimension of the input text vector.
            hidden_dim (int): Dimension of the mapped space.
            num_heads (int): Number of mapping spaces (heads).
        """
        super().__init__()
        self.num_heads = num_heads

        # Acoustic mapping layers
        self.acoustic_mapping_layers = torch.nn.ModuleList(
            [torch.nn.Linear(acoustic_dim, hidden_dim) for _ in range(num_heads)]
        )

        # Text mapping layers
        self.text_mapping_layers = torch.nn.ModuleList(
            [torch.nn.Linear(text_dim, hidden_dim) for _ in range(num_heads)]
        )

        # Dimension reduction for text
        self.text_dim_reduction = torch.nn.Linear(text_dim, hidden_dim - num_heads)

    def forward(
        self,
        acoustic_embeddings: torch.Tensor,
        text_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for LASAS.

        Args:
            acoustic_embeddings (torch.Tensor): Acoustic embeddings of shape (B, T, D).
            text_vectors (torch.Tensor): Aligned text vectors of shape (B, T, vocab).

        Returns:
            torch.Tensor: Linguistic-acoustic bimodal representation of shape (B, T, C).
        """
        similarities = []

        # Compute similarities for each head
        for text_map, acoustic_map in zip(
            self.text_mapping_layers, self.acoustic_mapping_layers
        ):
            acoustic_projected = acoustic_map(acoustic_embeddings)  # (B, T, C)
            text_anchor = text_map(text_vectors)  # (B, T, C)

            # Scale dot-product similarity
            similarity = torch.sum(text_anchor * acoustic_projected, dim=-1) / (
                (text_anchor.size(-1) / self.num_heads) ** 0.5
            )  # (B, T)
            similarities.append(similarity.unsqueeze(-1))

        # Concatenate similarities along the last dimension
        accent_shift = torch.cat(similarities, dim=-1)  # (B, T, N)

        # Reduce text dimension
        reduced_text = self.text_dim_reduction(text_vectors)  # (B, T, C - N)

        # Concatenate accent shift and reduced text
        bimodal_representation = torch.cat(
            [accent_shift, reduced_text], dim=-1
        )  # (B, T, C)

        return bimodal_representation


class LASASARModel(torch.nn.Module):
    def __init__(
        self,
        acoustic_dim: int,
        text_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_classes: int,
        normalize: bool = True,
        layer_norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
    ):
        """
        LASAS Accent Recognition Model

        Args:
            acoustic_dim (int): Dimension of the input acoustic embedding.
            text_dim (int): Dimension of the input text vector.
            hidden_dim (int): Hidden dimension for LASAS block.
            num_heads (int): Number of heads in the LASAS block.
            num_classes (int): Number of accent classes.
        """
        super().__init__()

        self.lasas_block = LASASBlock(acoustic_dim, text_dim, hidden_dim, num_heads)

        # Lightweight Transformer encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=3
        )

        # Linears
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, hidden_dim),
        )

        self.normalize = normalize
        self.norm = WENET_NORM_CLASSES[layer_norm_type](hidden_dim, eps=norm_eps)

        # Classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, num_classes),
        )

        # Loss function
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        acoustic_embeddings: torch.Tensor,
        text_vectors: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for LASAS AR Model

        Args:
            acoustic_embeddings (torch.Tensor): Acoustic embeddings of shape (B, T, D).
            text_vectors (torch.Tensor): Aligned text vectors of shape (B, T, vocab).
            labels (torch.Tensor, optional): Ground truth labels of shape (B, num_classes).

        Returns:
            bimodal_feats (torch.Tensor): The bimodal features of dialect and text (B, T, C).
            loss (torch.Tensor, optional): Computed loss if labels are provided.
        """
        # Generate bimodal representation
        bimodal_representation = self.lasas_block(
            acoustic_embeddings, text_vectors
        )  # (B, T, C)

        # Transformer encoder
        bimodal_representation = bimodal_representation.permute(1, 0, 2)  # (T, B, C)
        context_sensitive_representation = self.transformer_encoder(
            bimodal_representation
        )  # (T, B, C)
        context_sensitive_representation = context_sensitive_representation.permute(
            1, 0, 2
        )  # (B, T, C)

        # Linears
        bimodal_feats = self.linear(context_sensitive_representation)  # (B, T, C)
        if self.normalize:
            bimodal_feats = self.norm(bimodal_feats)

        # Global pooling (mean pooling over time)
        pooled_representation = context_sensitive_representation.mean(dim=1)  # (B, C)

        # Classification
        logits = self.classifier(pooled_representation)  # (B, num_classes)

        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return bimodal_feats, loss

    def forward_lasas(
        self,
        acoustic_embeddings: torch.Tensor,
        text_vectors: torch.Tensor,
    ):
        # Generate bimodal representation
        bimodal_representation = self.lasas_block(
            acoustic_embeddings, text_vectors
        )  # (B, T, C)

        # Transformer encoder
        bimodal_representation = bimodal_representation.permute(1, 0, 2)  # (T, B, C)
        context_sensitive_representation = self.transformer_encoder(
            bimodal_representation
        )  # (T, B, C)
        context_sensitive_representation = context_sensitive_representation.permute(
            1, 0, 2
        )  # (B, T, C)

        # Linears
        bimodal_feats = self.linear(context_sensitive_representation)  # (B, T, C)
        if self.normalize:
            bimodal_feats = self.norm(bimodal_feats)

        return bimodal_feats
