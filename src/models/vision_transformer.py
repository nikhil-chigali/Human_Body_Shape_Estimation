import torch
import torch.nn as nn

from utils.img2segments import img_to_segments
from models import AttentionBlock
from utils.consts import device


class VisionTransformer(nn.Module):
    def __init__(
        self,
        segment_size,
        num_layers,
        embed_size,
        hidden_size,
        num_heads,
        img_size,
        num_channels,
        output_size,
        segment_type="strips",
        dropout=0.0,
    ):
        super().__init__()
        self.segment_size = segment_size
        self.num_segments = 2 * (img_size // segment_size)
        self.embed_size = embed_size
        self.segment_type = segment_type
        # self.linear_projection = nn.Linear(
        #     img_size * segment_size * num_channels, embed_size
        # )
        self.linear_projection = nn.LazyLinear(embed_size)
        self.attn_blocks = nn.Sequential(
            *[
                AttentionBlock(embed_size, hidden_size, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm((2 + self.num_segments) * embed_size),
            nn.Linear((2 + self.num_segments) * embed_size, output_size),
        )
        self.dropout = nn.Dropout(dropout)

        self.pos_enc = nn.Parameter(torch.randn(2 + self.num_segments, 1, embed_size))

    def make_gender_tokens(self, genders: list):
        encoding = {"male": [0, 1], "female": [1, 0], "neutral": [1, 1]}
        self.gender_tokens = torch.Tensor(
            [encoding[gender] for gender in genders]
        )  # [B, 2]
        self.gender_tokens = self.gender_tokens.T.unsqueeze(2)  # [2, B, 1]
        self.gender_tokens = self.gender_tokens.repeat(
            (1, 1, self.embed_size)
        )  # [2, B, embed_size]
        self.gender_tokens = self.gender_tokens.to(device)

    def forward(self, x, genders):
        self.make_gender_tokens(genders)
        x = img_to_segments(
            {
                "img": x,
                "strip_thickness": self.segment_size,
                "flatten_channels": True,
                "batch_first": True,
            },
            segment_type=self.segment_type,
        )  # [T, B, segment_size*img_size*num_channels]
        embeds = self.linear_projection(x).transpose(0, 1)  # [T, B, embed_size]
        # Concatenate Gender Tokens: Male(0,1), Female(1,0), Neutral(1,1)
        embeds = torch.cat((self.gender_tokens, embeds), dim=0)  # [2+T, B, embed_size]
        # Adding positional encoding
        embeds += self.pos_enc  # [2+T, B, embed_size]

        x = self.dropout(embeds)
        x = self.attn_blocks(x)  # [2+T, B, embed_size]
        x = x.transpose(0, 1).flatten(1, 2)  # [B, (2+T)*embed_size]
        result = self.mlp_head(x)  # [B, output_size]
        return result
