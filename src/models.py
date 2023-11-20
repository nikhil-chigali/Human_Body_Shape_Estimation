import numpy as np

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

# Torchvision
import torchvision

# Pytorch Lightning
import pytorch_lightning as pl
import lightning as L

from .utils import img_to_strips


class AttentionBlock(nn.Module):
    """
    A module that implements a self-attention mechanism with feed-forward neural network layers.

    Args:
        embed_size (int): The dimensionality of the input embeddings.
        hidden_size (int): The dimensionality of the hidden layer in the feed-forward neural network. (Usually 2-4x larger than embed_size)
        num_heads (int): The number of attention heads in the multi-head attention mechanism.
        dropout (float, optional): The dropout probability. Default is 0.0.
        batch_first (bool, optional): batch_size in dimension 0. Default is False. ie., [T, B, _]

    Example Usage:
        # Create an instance of AttentionBlock
        attention_block = AttentionBlock(embed_size=512, hidden_size=2048, num_heads=8, dropout=0.1)

        # Pass input tensor through the attention block
        output = attention_block(input_tensor)
    """

    def __init__(
        self, embed_size, hidden_size, num_heads, dropout=0.0, batch_first=False
    ):
        super().__init__(self)
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.self_attn = nn.MultiheadAttention(
            embed_size, num_heads, dropout=dropout, batch_first=batch_first
        )
        self.layer_norm2 = nn.LayerNorm(embed_size)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embed_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Perform the forward pass of the attention block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying layer normalization, multi-head attention, and feed-forward neural network layers.
        """
        ln1_x = self.layer_norm1(x)
        attn_outputs = self.self_attn(ln1_x, ln1_x, ln1_x)[0]
        add1_x = x + attn_outputs
        ln2_x = self.layer_norm2(add1_x)
        ffn_outputs = self.ffn(ln2_x)
        result = add1_x + ffn_outputs
        return result


class VisionTransformer(nn.Module):
    def __init__(
        self,
        strip_thickness,
        num_strips,
        num_layers,
        embed_size,
        hidden_size,
        num_heads,
        img_size,
        num_channels,
        output_size,
        dropout=0.0,
        batch_first=False,
    ):
        super().__init__(self)
        self.strip_thickness = strip_thickness
        self.num_strips = num_strips
        self.batch_first = batch_first

        self.linear_projection = nn.Linear(
            img_size * strip_thickness * num_channels, embed_size
        )
        self.transformer = [
            AttentionBlock(embed_size, hidden_size, num_heads, dropout, batch_first)
            for _ in range(num_layers)
        ]
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, output_size),
        )
        self.dropout = nn.Dropout(dropout)
        # self.pos_encoding [TODO]

    def forward(self, x):
        x = img_to_strips(
            x, self.strip_thickness, True, self.batch_first
        )  # [T, B, strip_thickness*img_size*num_channels]
        embeds = self.linear_projection(x)
        # embeds += self.pos_encoding [TODO]

        x = self.dropout(embeds)
        x = self.transformer(x)

        result = self.mlp_head(x)
        return result


class Img2SMPLx(nn.Module):
    def __init__(self):
        super().__init__(self)

    def forward(self):
        pass
