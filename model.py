import torch
import torch.nn as nn
from torch.nn import functional as F
from config import LargeLanguageModelConfig

import tiktoken

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        assert config.embed % config.nb_head == 0 # if the config is impossible return an assertion error

        self.nb_head  = config.nb_head
        self.embed    = config.embed
        self.dropout  = config.dropout
        self.fast_att = config.fast_att

        # attention map boolean mask
        mask = torch.ones(config.seq_size, config.seq_size, dtype=torch.bool).tril(diagonal=0)
        self.register_buffer("mask", mask)

        if not config.fast_att:
            norm_val = torch.rsqrt(torch.Tensor([config.embed // config.nb_head]))
            self.register_buffer("norm_val", norm_val)

        # Make the calculation of Query, Key & Value in the same FullyConnected layer to simplify code but could be bad for quantization
        self.fully_connected_0 = nn.Linear(config.embed, 3 * config.embed, bias=config.bias)
        # output projection
        self.fully_connected_1 = nn.Linear(config.embed, config.embed, bias=config.bias)

        # regularization
        self.dropout_0 = nn.Dropout(config.dropout)
        self.dropout_1 = nn.Dropout(config.dropout)

    def forward(self, x):

        B, S, E = x.size() # x -> shape : (batch, seq, embed)

        # qkv shape : (batch, seq, embed*3)
        qkv = self.fully_connected_0(x)

        # q,k,v shape : (batch, seq, embed)
        q, k, v = qkv.split(self.embed, dim=2)
        
        # q,k,v shapes : (batch, nb_head, seq, embed//nb_head)
        q = q.view(B, S, self.nb_head, E // self.nb_head).transpose(1, 2)
        k = k.view(B, S, self.nb_head, E // self.nb_head).transpose(1, 2)
        v = v.view(B, S, self.nb_head, E // self.nb_head).transpose(1, 2)

        if self.fast_att:
            # fast-attentionv2 mechanism
            r = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                        dropout_p=self.dropout,
                                                        is_causal=True,
                                                        scale=None,
                                                        enable_gqa=False)
        else:
            # attention mechanism
            k_t = k.transpose(-2, -1)
            a = (q @ k_t) # shape : (batch, nb_head, seq, seq)
            a *= self.norm_val # Normalisation by the sqrt of embed//nb_head
            a = a.masked_fill(self.mask.logical_not(), float('-inf')) # masking the attention map to avoid a token to see the next tokens
            a = F.softmax(a, dim=-1)
            a = self.dropout_0(a)
            r = a @ v # shape : (batch, nb_head, seq, embed//nb_head)

        r = r.transpose(1, 2).reshape(B,S,E) # shape : (batch, seq, embed)

        # output projection
        r = self.fully_connected_1(r)
        r = self.dropout_1(r)

        return r

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.fully_connected_0 = nn.Linear(config.embed, 4*config.embed, bias=config.bias)
        self.fully_connected_1 = nn.Linear(4*config.embed, config.embed, bias=config.bias)
        self.dropout           = nn.Dropout(config.dropout)
        self.gelu              = nn.GELU()

    def forward(self, x):
        x = self.fully_connected_0(x)
        x = self.gelu(x)
        x = self.fully_connected_1(x)
        x = self.dropout(x)
        return x

class TransformerLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embed)
        self.attn = MultiHeadSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.embed)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class LargeLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.config = LargeLanguageModelConfig()

        self.token_embed = nn.Embedding(self.config.vocab_size, self.config.embed)
        self.pos_embed   = nn.Embedding(self.config.seq_size, self.config.embed)
        self.translayers = nn.Sequential(*[TransformerLayer(self.config) for _ in range(self.config.nb_trans)])
        self.ln          = nn.LayerNorm(self.config.embed) # final layer norm
        self.lm_head     = nn.Linear(self.config.embed, self.config.vocab_size, bias=False)

        arpos = torch.arange(self.config.seq_size)
        self.register_buffer("arpos", arpos)

        # Share the same weights between the first and last layer
        #  so that the dot product of output embeddings with last-layer 
        #   is revealing the closest tokens to the input vocab embedding
        self.token_embed.weight = self.lm_head.weight
        # self.lm_head.weight = nn.Parameter(torch.transpose(torch.linalg.pinv(self.token_embed.weight.float()).bfloat16(),1,0))

        # initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):

        B, S = idx.shape

        # idx shape : (B,S) tensor of integer
        tok_emb = self.token_embed(idx) # shape : (B,S,E)
        pos_emb = self.pos_embed(torch.arange(S).to(tok_emb.device)) # shape : (S,E)
        x = tok_emb + pos_emb # shape : (B,S,E)
        x = self.translayers(x) # shape : (B,S,E)
        x = self.ln(x) # shape : (B,S,E)

        if self.training:
            # During training
            logits = self.lm_head(x) # shape : (B,S,vocab_size)
            return logits
        else:

            # During inference
            logits = self.lm_head(x[:, -1, :]) # shape : (B,vocab_size)
            return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        enc = tiktoken.get_encoding("gpt2")

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at seq_size
            idx_cond = idx if idx.size(1) <= self.config.seq_size else idx[:, -self.config.seq_size:]
            # pad = torch.zeros((1,self.config.seq_size-idx_cond.size(1)),dtype=idx_cond.dtype).to(idx_cond.device) # [0] = !   [1] = "
            # idx_cond = torch.cat([pad,idx_cond],-1)
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits /= temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                if self.training:
                    logits[logits < v[:, [-1]]] = -float('Inf')
                else:
                    logits[logits < v] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

            decoded_generated_token = enc.decode([idx_next])

            print(decoded_generated_token, end="")

        print()

        return idx