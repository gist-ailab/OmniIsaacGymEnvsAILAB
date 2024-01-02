import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
copy from 
https://github.com/haosulab/ManiSkill-Learn/blob/6ccff87b6e6def8d0c3938cd20d3136a964be160/mani_skill_learn/networks/modules/attention.py#L103
'''

def compute_attention(q, k, v, dropout=None, mask=None):
    """
    :param q: Query [B, NH, NQ, EL] or [NH, 1, EL] (in this case NQ=1)
    :param k: Key [B, NH, NK, EL]
    :param v: Value [B, NH, NK, EL]
    :param mask: [B, NQ, NK]
    :param dropout:
    :return:
    """
    if q.ndim + 1 == k.ndim:
        score = torch.einsum('nij,bnkj->bnik', q, k) # [B, NH, 1, NK]
    elif q.ndim == k.ndim:
        score = torch.einsum('bnij,bnkj->bnik', q, k) # [B, NH, NQ, NK]
    score = score / np.sqrt(q.shape[-1])  
    if mask is not None:
        mask = mask[:, None]
        score = score * mask + (-1e8) * (1 - mask)
    score = F.softmax(score, dim=-1)   # [B, NH, NQ, NK]
    if dropout is not None:
        score = dropout(score)
    return torch.einsum('bnij,bnjk->bnik', score, v)    # [B, NH, NQ, EL]


class MultiHeadedAttentionBase(nn.Module):
    def __init__(self, embed_dim, num_heads, latent_dim, dropout=None):
        """
        :param embed_dim: The dimension of feature in each entity.
        :param num_heads: The number of attention heads.
        :param latent_dim:
        :param dropout:
        """
        super().__init__()
        self.w_k = nn.Parameter(torch.empty(num_heads, embed_dim, latent_dim))
        self.w_v = nn.Parameter(torch.empty(num_heads, embed_dim, latent_dim))
        self.w_o = nn.Parameter(torch.empty(num_heads, latent_dim, embed_dim))
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.w_k)
        nn.init.xavier_normal_(self.w_v)
        nn.init.xavier_normal_(self.w_o)
        if hasattr(self, 'q'):
            nn.init.xavier_normal_(self.q)
        if hasattr(self, 'w_q'):
            nn.init.xavier_normal_(self.w_q)


class AttentionPooling(MultiHeadedAttentionBase):
    def __init__(self, embed_dim, num_heads, latent_dim, dropout=None):
        super().__init__(embed_dim, num_heads, latent_dim, dropout)
        self.q = nn.Parameter(torch.empty(num_heads, 1, latent_dim))
        self._reset_parameters()

    def forward(self, x, mask=None):
        """
        :param x: [B, N, E] [batch size, length, embed_dim] the input to the layer, a tensor of shape
        :param mask: [B, 1, N] [batch size, 1, length]
        :return: [B, E] [batch_size, embed_dim] one feature with size
        """
        k = torch.einsum('blj,njd->bnld', x, self.w_k)  # [B, NH, N, EL]
        v = torch.einsum('blj,njd->bnld', x, self.w_v)  # [B, NH, N, EL]

        out = compute_attention(self.q, k, v, self.dropout, mask)
        out = torch.einsum('bnlj,njk->blk', out, self.w_o) # [B, 1, E]
        out = out[:, 0]
        return out