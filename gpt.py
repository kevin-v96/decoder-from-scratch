"""
Source: https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import torch 
from torch import nn 
import torch.nn.functional as F
from decoder_only_block import DecoderBlock

class GPT(nn.Module):
    def __init__(self, d, H, T, V, layers, bias = False, dropout = 0.2) -> None:
        """
        Arguments:
        d: size of embedding dimension
        H: number of attention heads
        T: maximum length of input sequences (in tokens)
        V: size of the token vocabulary
        layers: number of decoder-only blocks
        bias: whether or not to use bias in linear layers
        dropout: probability of dropout
        """
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(V, d), #token embeddings
            wpe = nn.Embedding(T, d), #positional embeddings
            drop = nn.Dropout(dropout),
            blocks = nn.ModuleList([DecoderBlock(d, H, T, bias, dropout) for _ in range(layers)]),
            ln_f = nn.LayerNorm(d),
            head = nn.Linear(d, V, bias = bias)
        ))

    def forward(self, idx, targets = None):
        # idx is a [B, T] matrix of token indices
        # targets is a [B, T] matrix of target (next) token indices
        device = idx.device
        _, T = idx.size() #(B, T)
        pos = torch.arange(0, T, dtype = torch.long, device = device)

        #generate position and token embeddings
        tok_emb = self.transformer.wte(idx) #[B,T,d]
        pos_emb = self.transformer.wpe(pos) #[T, d]
        x = self.transformer.drop(tok_emb + pos_emb)

        #pass through all decoder-only blocks
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.ln_f(x) #final layer norm

        if targets is not None:
            # compute the loss if we are given targets
            logits = self.transformer.head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            #only look at the last token if performing inference
            logits = self.transformer.head(x[:, [-1], :])
            loss = None

        return logits, loss

