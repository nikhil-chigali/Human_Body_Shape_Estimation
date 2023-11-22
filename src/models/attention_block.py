import torch.nn as nn


class AttentionBlock(nn.Module):
    """
    A module that implements a self-attention mechanism with feed-forward neural network layers.

    Args:
        embed_size (int): The dimensionality of the input embeddings.
        hidden_size (int): The dimensionality of the hidden layer in the feed-forward neural network. (Usually 2-4x larger than embed_size)
        num_heads (int): The number of attention heads in the multi-head attention mechanism.
        dropout (float, optional): The dropout probability. Default is 0.0.

    Example Usage:
        # Create an instance of AttentionBlock
        attention_block = AttentionBlock(embed_size=512, hidden_size=2048, num_heads=8, dropout=0.1)

        # Pass input tensor through the attention block
        output = attention_block(input_tensor)
    """

    def __init__(self, embed_size, hidden_size, num_heads, dropout=0.0):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.self_attn = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)
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
