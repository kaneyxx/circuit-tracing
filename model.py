import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, List, Dict, Any, Union

class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        self.d_model = d_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False) + self.eps
        y = (x - mean) / std
        return self.weight * y + self.bias

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: Optional[int] = None):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_head if d_head is not None else d_model // n_heads
        
        # Create weight matrices for Q, K, V and output projection
        self.W_Q = nn.Parameter(torch.empty(n_heads, self.d_head, d_model))
        self.W_K = nn.Parameter(torch.empty(n_heads, self.d_head, d_model))
        self.W_V = nn.Parameter(torch.empty(n_heads, self.d_head, d_model))
        self.W_O = nn.Parameter(torch.empty(n_heads, d_model, self.d_head))
        
        # Initialize weights
        nn.init.normal_(self.W_Q, mean=0.0, std=0.02)
        nn.init.normal_(self.W_K, mean=0.0, std=0.02)
        nn.init.normal_(self.W_V, mean=0.0, std=0.02)
        nn.init.normal_(self.W_O, mean=0.0, std=0.02)
        
    def forward(self, x: torch.Tensor, cache: Optional[Dict[str, torch.Tensor]] = None, 
               layer_idx: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x: [batch_size, seq_len, d_model]
        Returns: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # [batch_size, seq_len, n_heads, d_head]
        q = torch.einsum('bsd,hds->bhsd', x, self.W_Q)
        k = torch.einsum('bsd,hds->bhsd', x, self.W_K)
        v = torch.einsum('bsd,hds->bhsd', x, self.W_V)
        
        # Store intermediate activations if caching
        if cache is not None and layer_idx is not None:
            cache[f'q_{layer_idx}'] = q
            cache[f'k_{layer_idx}'] = k
            cache[f'v_{layer_idx}'] = v
        
        # Calculate attention scores: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.einsum('bhsd,bhmd->bhsm', q, k) / math.sqrt(self.d_head)
        
        # Apply causal mask to enforce autoregressive property
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores.masked_fill_(mask, -1e9)
        
        # Calculate attention pattern
        pattern = torch.softmax(scores, dim=-1)
        
        if cache is not None and layer_idx is not None:
            cache[f'pattern_{layer_idx}'] = pattern
        
        # Apply attention to values: [batch_size, n_heads, seq_len, d_head]
        z = torch.einsum('bhsm,bhmd->bhsd', pattern, v)
        
        # Apply output projection: [batch_size, seq_len, d_model]
        output = torch.einsum('bhsd,hds->bsd', z, self.W_O)
        
        return output, cache

class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, activation: nn.Module = nn.GELU()):
        super().__init__()
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.activation = activation
        
        # Weight matrices for MLP
        self.W_in = nn.Parameter(torch.empty(d_mlp, d_model))
        self.W_out = nn.Parameter(torch.empty(d_model, d_mlp))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.b_out = nn.Parameter(torch.zeros(d_model))
        
        # Initialize weights
        nn.init.normal_(self.W_in, mean=0.0, std=0.02)
        nn.init.normal_(self.W_out, mean=0.0, std=0.02)
        
    def forward(self, x: torch.Tensor, cache: Optional[Dict[str, torch.Tensor]] = None, 
               layer_idx: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x: [batch_size, seq_len, d_model]
        Returns: [batch_size, seq_len, d_model]
        """
        # First linear layer + activation
        pre_act = torch.einsum('bsd,md->bsm', x, self.W_in) + self.b_in
        
        if cache is not None and layer_idx is not None:
            cache[f'pre_act_{layer_idx}'] = pre_act
        
        act = self.activation(pre_act)
        
        if cache is not None and layer_idx is not None:
            cache[f'act_{layer_idx}'] = act
        
        # Second linear layer
        output = torch.einsum('bsm,dm->bsd', act, self.W_out) + self.b_out
        
        return output, cache

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mlp: int, d_head: Optional[int] = None):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads, d_head)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_mlp)
        
    def forward(self, x: torch.Tensor, cache: Optional[Dict[str, torch.Tensor]] = None, 
               layer_idx: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Store residual connections for circuit tracing
        if cache is not None:
            cache[f'resid_pre_{layer_idx}'] = x
        
        # Attention sublayer
        ln1_out = self.ln1(x)
        if cache is not None:
            cache[f'normalized_{layer_idx}_ln1'] = ln1_out
            
        attn_out, cache = self.attention(ln1_out, cache, layer_idx)
        x = x + attn_out
        
        if cache is not None:
            cache[f'resid_mid_{layer_idx}'] = x
        
        # MLP sublayer
        ln2_out = self.ln2(x)
        if cache is not None:
            cache[f'normalized_{layer_idx}_ln2'] = ln2_out
            
        mlp_out, cache = self.mlp(ln2_out, cache, layer_idx)
        x = x + mlp_out
        
        if cache is not None:
            cache[f'resid_post_{layer_idx}'] = x
        
        return x, cache

class Transformer(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        n_layers: int, 
        n_heads: int, 
        d_mlp: int, 
        vocab_size: int, 
        d_head: Optional[int] = None,
        max_seq_len: int = 1024
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_mlp = d_mlp
        self.vocab_size = vocab_size
        
        # Token embedding and positional encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.empty(max_seq_len, d_model))
        
        # Initialize position embeddings
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_mlp, d_head) for _ in range(n_layers)]
        )
        
        # Output normalization and projection
        self.ln_final = LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        # Tie embedding and unembedding weights
        self.unembed.weight = self.token_embedding.weight
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [batch_size, seq_len]
        Returns: [batch_size, seq_len, vocab_size] logits
        """
        batch_size, seq_len = input_ids.shape
        
        # Combine token embeddings and positional embeddings
        x = self.token_embedding(input_ids) + self.pos_embedding[:seq_len]
        
        # Pass through transformer blocks
        for i, block in enumerate(self.blocks):
            x, _ = block(x, layer_idx=i)
        
        # Final layernorm and unembedding
        x = self.ln_final(x)
        logits = self.unembed(x)
        
        return logits
    
    def run_with_cache(
        self, 
        input_ids: torch.Tensor, 
        stop_at_layer: Optional[int] = None,
        names_filter: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Run the model and cache activations. Useful for circuit analysis.
        input_ids: [batch_size, seq_len]
        stop_at_layer: Optional int, stop computation at this layer
        names_filter: Optional list of strings, only cache activations whose names contain these strings
        
        Returns: (logits, cache) tuple where cache contains activation tensors
        """
        batch_size, seq_len = input_ids.shape
        cache = {}
        
        # Combine token embeddings and positional embeddings
        x = self.token_embedding(input_ids) + self.pos_embedding[:seq_len]
        cache['token_embed'] = self.token_embedding(input_ids)
        cache['pos_embed'] = self.pos_embedding[:seq_len].expand(batch_size, seq_len, -1)
        
        # Pass through transformer blocks
        for i, block in enumerate(self.blocks):
            if stop_at_layer is not None and i >= stop_at_layer:
                break
                
            x, cache = block(x, cache=cache, layer_idx=i)
        
        # Apply filter to cache if needed
        if names_filter is not None:
            cache = {k: v for k, v in cache.items() 
                    if any(filter_str in k for filter_str in names_filter)}
        
        # Final layernorm and unembedding if needed
        if stop_at_layer is None or stop_at_layer >= len(self.blocks):
            x = self.ln_final(x)
            cache['ln_final_out'] = x
            logits = self.unembed(x)
        else:
            logits = None
            
        return logits, cache 