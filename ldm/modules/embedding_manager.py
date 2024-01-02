import torch
from torch import nn

from ldm.data.personalized import per_img_token_list
from transformers import CLIPTokenizer
from functools import partial
import numpy as np
from ldm.modules.attention import CrossAttention
import PIL
from PIL import Image

DEFAULT_PLACEHOLDER_TOKEN = ["*"]

PROGRESSIVE_SCALE = 2000


def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(string, truncation=True, max_length=77, return_length=True,
                               return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"]
    # assert torch.count_nonzero(tokens - 49407) == 2, f"String '{string}' maps to more than a single token. Please use another string"

    return tokens[0, 1]


def get_bert_token_for_string(tokenizer, string):
    token = tokenizer(string)
    assert torch.count_nonzero(
        token) == 3, f"String '{string}' maps to more than a single token. Please use another string"

    token = token[0, 1]

    return token


def get_embedding_for_clip_token(embedder, token):
    return embedder(token.unsqueeze(0))[0, 0]


class EmbeddingManager(nn.Module):
    def __init__(
            self,
            embedder,
            placeholder_strings=None,
            initializer_per_img=None,
            per_image_tokens=False,
            num_vectors_per_token=1,
            progressive_words=False,
            **kwargs
    ):
        super().__init__()

        self.string_to_token_dict = {}

        self.init = True

        self.cond_stage_model = embedder

        # self.learnable_vector = nn.Parameter(torch.rand((1, 1, 768)), requires_grad=True)
        # self.learnable_vector.requires_grad_(True)
        # self.mapper = Mapper(input_dim=1024, output_dim=768)

        self.progressive_words = progressive_words
        self.progressive_counter = 0

        self.max_vectors_per_token = num_vectors_per_token

        if hasattr(embedder, 'tokenizer'):  # using Stable Diffusion's CLIP encoder
            self.is_clip = True
            get_token_for_string = partial(get_clip_token_for_string, embedder.tokenizer)
            get_embedding_for_tkn = partial(get_embedding_for_clip_token, embedder.transformer.text_model.embeddings)
            token_dim = 768
        else:  # using LDM's BERT encoder
            self.is_clip = False
            get_token_for_string = partial(get_bert_token_for_string, embedder.tknz_fn)
            get_embedding_for_tkn = embedder.transformer.token_emb
            token_dim = 1280

        self.attention = Attentions(dim=token_dim, n_heads=8, d_head=64, dropout=0.05)

        if per_image_tokens:
            placeholder_strings.extend(per_img_token_list)

        for idx, placeholder_string in enumerate(placeholder_strings):
            token = get_token_for_string(placeholder_string)

            self.string_to_token_dict[placeholder_string] = token
        self.MLP = MLP1()

    def forward(
            self,
            tokenized_text,
            embedded_text,
            image_embeds,
    ):
        b, n, device = *tokenized_text.shape, tokenized_text.device
        for placeholder_string, placeholder_token in self.string_to_token_dict.items():
        
            # placeholder_embedding = self.attention(image_embeds.view(b, 5, 768).to(device),
            #                                        image_embeds.view(b, 5, 768).to(device))[0,4].view(1, 768)
            placeholder_embedding = self.attention(self.MLP.learnable_vector.view(b, 1, 768).to(device)).view(1, 768)
            # ZZZ  = self.MLP(self.MLP.learnable_vector).view(1, 768)
        
            if self.max_vectors_per_token == 1:  # If there's only one vector per token, we can do a simple replacement
                placeholder_idx = torch.where(tokenized_text == placeholder_token.to(device))
                embedded_text[placeholder_idx] = placeholder_embedding.float()
            else:  # otherwise, need to insert and keep track of changing indices
                if self.progressive_words:
                    self.progressive_counter += 1
                    max_step_tokens = 1 + self.progressive_counter // PROGRESSIVE_SCALE
                else:
                    max_step_tokens = self.max_vectors_per_token
        
                num_vectors_for_token = min(placeholder_embedding.shape[0], max_step_tokens)
        
                placeholder_rows, placeholder_cols = torch.where(tokenized_text == placeholder_token.to(device))
        
                if placeholder_rows.nelement() == 0:
                    continue
        
                sorted_cols, sort_idx = torch.sort(placeholder_cols, descending=True)
                sorted_rows = placeholder_rows[sort_idx]
        
                for idx in range(len(sorted_rows)):
                    row = sorted_rows[idx]
                    col = sorted_cols[idx]
        
                    new_token_row = torch.cat(
                        [tokenized_text[row][:col], placeholder_token.repeat(num_vectors_for_token).to(device),
                         tokenized_text[row][col + 1:]], axis=0)[:n]
                    new_embed_row = torch.cat([embedded_text[row][:col], placeholder_embedding[:num_vectors_for_token],
                                               embedded_text[row][col + 1:]], axis=0)[:n]
        
                    embedded_text[row] = new_embed_row
                    tokenized_text[row] = new_token_row

        return embedded_text
        # print(self.MLP.learnable_vector)
        # ZZZ = self.MLP(self.MLP.learnable_vector)

    def save(self, ckpt_path):
        torch.save({
            "string_to_token": self.string_to_token_dict,
            "attention": self.attention,
            "learnable_vector": self.MLP.learnable_vector
        }, ckpt_path)

    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        print('find keys:', ckpt.keys())

        self.string_to_token_dict = ckpt["string_to_token"]

        if 'attention' in ckpt.keys():
            self.attention = ckpt["attention"]
        else:
            self.attention = None
        if 'learnable_vector' in ckpt.keys():
            self.MLP.learnable_vector = ckpt["learnable_vector"]
        else:
            self.MLP.learnable_vector = None

    def embedding_parameters(self):
        # print(11111)
        parms = []
        parms = list(self.attention.parameters())
        # parms += list(self.learnable_vector)
        return parms

    def embedding_to_coarse_loss(self):
        loss = 0.
        num_embeddings = len(self.initial_embeddings)

        for key in self.initial_embeddings:
            optimized = self.string_to_param_dict[key]
            coarse = self.initial_embeddings[key].clone().to(optimized.device)

            loss = loss + (optimized - coarse) @ (optimized - coarse).T / num_embeddings

        return loss


class Attentions(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head,
                                    dropout=dropout)  # is a self-attention

        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, dim))

    def forward(self, x, context=None):
        x_1 = self.attn1(x)
        x_2 = self.attn2(x_1, x)
        x_3 = self.net(x_2)
        return x_3


import math

import torch as th
import torch.nn as nn


def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


class LayerNorm(nn.LayerNorm):
    """
    Implementation that supports fp16 inputs but fp32 gains/biases.
    """

    def forward(self, x: th.Tensor):
        return super().forward(x.float()).to(x.dtype)


class MultiheadAttention(nn.Module):
    def __init__(self, n_ctx, width, heads):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(heads, n_ctx)

    def forward(self, x):
        x = self.c_qkv(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class QKVMultiheadAttention(nn.Module):
    def __init__(self, n_heads: int, n_ctx: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_ctx = n_ctx

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.n_heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.n_heads, -1)
        q, k, v = th.split(qkv, attn_ch, dim=-1)
        weight = th.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        wdtype = weight.dtype
        weight = th.softmax(weight.float(), dim=-1).type(wdtype)
        return th.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            n_ctx: int,
            width: int,
            heads: int,
    ):
        super().__init__()

        self.attn = MultiheadAttention(
            n_ctx,
            width,
            heads,
        )
        self.ln_1 = LayerNorm(width)
        self.mlp = MLP(width)
        self.ln_2 = LayerNorm(width)

    def forward(self, x: th.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
            self,  # 1 1024 5 1
            n_ctx: int,
            width: int,
            layers: int,
            heads: int,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    n_ctx,
                    width,
                    heads,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: th.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x


# from ldm.modules.encoders.xf import LayerNorm, Transformer

class MLP1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.learnable_vector = nn.Parameter(torch.randn((1, 1, 768)), requires_grad=True)
        self.proj = nn.Linear(768, 1024)
        self.final_ln = LayerNorm(1024)
        self.mapper = Transformer(1,1024,5,1,)
        self.proj1 = nn.Linear(1024, 768)

        self.freeze()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.mapper.parameters():
            param.requires_grad = True
        for param in self.final_ln.parameters():
            param.requires_grad = True
        for param in self.proj.parameters():
            param.requires_grad = True
        for param in self.proj1.parameters():
            param.requires_grad = True
        self.learnable_vector.requires_grad_(True)

    def forward(self, x):
        x = self.proj(x)
        x = self.final_ln(x)
        z = self.mapper(x)
        z = self.proj1(z)
        return z

class Mapper(nn.Module):
    def __init__(self,
        input_dim: int,
        output_dim: int,
    ):
        super(Mapper, self).__init__()

        for i in range(5):
            setattr(self, f'mapping_{i}', nn.Sequential(nn.Linear(input_dim, 1024),
                                         nn.LayerNorm(1024),
                                         nn.LeakyReLU(),
                                         nn.Linear(1024, 1024),
                                         nn.LayerNorm(1024),
                                         nn.LeakyReLU(),
                                         nn.Linear(1024, output_dim)))

            setattr(self, f'mapping_patch_{i}', nn.Sequential(nn.Linear(input_dim, 1024),
                                                        nn.LayerNorm(1024),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1024, 1024),
                                                        nn.LayerNorm(1024),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1024, output_dim)))

    def forward(self, embs):
        hidden_states = ()
        for i, emb in enumerate(embs):
            hidden_state = getattr(self, f'mapping_{i}')(emb[:, :1]) + getattr(self, f'mapping_patch_{i}')(emb[:, 1:]).mean(dim=1, keepdim=True)
            hidden_states += (hidden_state, )
        hidden_states = torch.cat(hidden_states, dim=1)
        return hidden_states
    

if __name__ == '__main__':
    model = MLP1()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    model.train()
    optimizer.zero_grad()
    # wo_weight = model()
    print(model.learnable_vector)
    # for parms in model.named_parameters():
    #     print(parms)
    # linear.weight.data = init_weight * (1 + wo_weight)
    # out = linear(x)
    y = torch.randn(1, 77, 768)
    x = torch.randn(1, 77, 768)
    out = model(model.learnable_vector)
    loss = nn.functional.mse_loss(y, out)
    # loss = wo_weight.sum()
    print("loss:", loss)
    loss.backward()
    # grad = linear.weight.grad * init_weight
    # wo_weight.backward(grad)
    optimizer.step()
    # for parms in model.named_parameters():
    #     print(parms)
    print(model.learnable_vector)
