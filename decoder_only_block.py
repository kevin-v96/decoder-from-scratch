"""
Source: https://github.com/karpathy/nanoGPT/blob/master/model.py
"""
from transformer_ffnn import FFNN
from causal_self_attention import CausalSelfAttention
from torch import nn

class DecoderBlock(nn.Module):
    def __init__(self, d, H, T, bias = False, dropout = 0.2) -> None:
        """
        Arguments:
        d: size of embedding dimension
        H: number of attention heads
        T: maximum length of input sequences (in tokens)
        bias: whether or not to use bias in linear layers
        dropout: probability of dropout
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, H, T, bias, dropout)
        self.ln_2 = nn.LayerNorm(d)
        self.ffnn = FFNN(d, bias, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffnn(self.ln_2(x))
        return x