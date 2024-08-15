import math
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import ModelMixin
from diffusers.models.embeddings import ImagePositionalEmbeddings
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available

@dataclass
class Transformer2DModelOutput(BaseOutput):
    sample: torch.FloatTensor

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

class Transformer2DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        activation_fn: str = "silu",
        num_embeds_ada_norm: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = kwargs.get('attention_head_dim', [5, 10, 20])
        self.transformer_layers_per_block = kwargs.get('transformer_layers_per_block', [1, 2, 10])
        inner_dim = num_attention_heads * attention_head_dim

        self.is_input_continuous = in_channels is not None
        self.is_input_vectorized = num_vector_embeds is not None

        if self.is_input_continuous and self.is_input_vectorized:
            raise ValueError(
                f"Cannot define both `in_channels`: {in_channels} and `num_vector_embeds`: {num_vector_embeds}. Make"
                " sure that either `in_channels` or `num_vector_embeds` is None."
            )
        elif not self.is_input_continuous and not self.is_input_vectorized:
            raise ValueError(
                f"Has to define either `in_channels`: {in_channels} or `num_vector_embeds`: {num_vector_embeds}. Make"
                " sure that either `in_channels` or `num_vector_embeds` is not None."
            )

        if self.is_input_continuous:
            self.in_channels = in_channels

            self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        elif self.is_input_vectorized:
            assert sample_size is not None, "Transformer2DModel over discrete input must provide sample_size"
            assert num_vector_embeds is not None, "Transformer2DModel over discrete input must provide num_embed"

            self.height = sample_size
            self.width = sample_size
            self.num_vector_embeds = num_vector_embeds
            self.num_latent_pixels = self.height * self.width

            self.latent_image_embedding = ImagePositionalEmbeddings(
                num_embed=num_vector_embeds, embed_dim=inner_dim, height=self.height, width=self.width
            )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                )
                for d in range(num_layers)
            ]
        )

        if self.is_input_continuous:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
        elif self.is_input_vectorized:
            self.norm_out = nn.LayerNorm(inner_dim)
            self.out = nn.Linear(inner_dim, self.num_vector_embeds - 1)

    def _set_attention_slice(self, slice_size):
        for block in self.transformer_blocks:
            block._set_attention_slice(slice_size)

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, attn_map=None, attn_shift=False, obj_ids=None, relationship=None, return_dict: bool = True):
        if self.is_input_continuous:
            batch, channel, height, weight = hidden_states.shape
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        elif self.is_input_vectorized:
            hidden_states = self.latent_image_embedding(hidden_states)

        for block in self.transformer_blocks:
            hidden_states, cross_attn_prob = block(hidden_states, context=encoder_hidden_states, timestep=timestep)

        if self.is_input_continuous:
            hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2)
            hidden_states = self.proj_out(hidden_states)
            output = hidden_states + residual
        elif self.is_input_vectorized:
            hidden_states = self.norm_out(hidden_states)
            logits = self.out(hidden_states)
            logits = logits.permute(0, 2, 1)
            output = F.log_softmax(logits.double(), dim=1).float()

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output), cross_attn_prob

    def _set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        for block in self.transformer_blocks:
            block._set_use_memory_efficient_attention_xformers(use_memory_efficient_attention_xformers)

class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_head_channels: Optional[int] = None,
        norm_num_groups: int = 32,
        rescale_output_factor: float = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.channels = channels

        self.num_heads = channels // num_head_channels if num_head_channels is not None else 1
        self.num_head_size = num_head_channels
        self.group_norm = nn.GroupNorm(num_channels=channels, num_groups=norm_num_groups, eps=eps, affine=True)

        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)

        self.rescale_output_factor = rescale_output_factor
        self.proj_attn = nn.Linear(channels, channels, 1)

    def transpose_for_scores(self, projection: torch.Tensor) -> torch.Tensor:
        new_projection_shape = projection.size()[:-1] + (self.num_heads, -1)
        new_projection = projection.view(new_projection_shape).permute(0, 2, 1, 3)
        return new_projection

    def forward(self, hidden_states):
        residual = hidden_states
        batch, channel, height, width = hidden_states.shape

        hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

        query_proj = self.query(hidden_states)
        key_proj = self.key(hidden_states)
        value_proj = self.value(hidden_states)

        query_states = self.transpose_for_scores(query_proj)
        key_states = self.transpose_for_scores(key_proj)
        value_states = self.transpose_for_scores(value_proj)

        scale = 1 / math.sqrt(math.sqrt(self.channels / self.num_heads))
        attention_scores = torch.matmul(query_states * scale, key_states.transpose(-1, -2) * scale)
        attention_probs = torch.softmax(attention_scores.float(), dim=-1).type(attention_scores.dtype)

        hidden_states = torch.matmul(attention_probs, value_states)

        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        new_hidden_states_shape = hidden_states.size()[:-2] + (self.channels,)
        hidden_states = hidden_states.view(new_hidden_states_shape)

        hidden_states = self.proj_attn(hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)

        hidden_states = (hidden_states + residual) / self.rescale_output_factor
        return hidden_states

class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "silu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
    ):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
        )
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.attn2 = CrossAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
        )

        self.use_ada_layer_norm = num_embeds_ada_norm is not None
        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
            self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm)
        else:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def _set_attention_slice(self, slice_size):
        self.attn1._slice_size = slice_size
        self.attn2._slice_size = slice_size

    def _set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    def forward(self, hidden_states, context=None, timestep=None):
        # print(f"BasicTransformerBlock - context shape: {context.shape if context is not None else 'None'}")
        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        )
        tmp_hidden_states, cross_attn_prob = self.attn1(norm_hidden_states)
        hidden_states = tmp_hidden_states + hidden_states

        norm_hidden_states = (
            self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
        )
        tmp_hidden_states, cross_attn_prob = self.attn2(norm_hidden_states, context=context)
        hidden_states = tmp_hidden_states + hidden_states

        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        return hidden_states, cross_attn_prob

class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        **kwargs,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.scale = dim_head**-0.5
        self.heads = heads
        self._slice_size = None
        self._use_memory_efficient_attention_xformers = False

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def forward(self, hidden_states, context=None, mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)

        # 调试信息
        # print(f"context shape: {context.shape}")
        # print(f"self.to_k weight shape: {self.to_k.weight.shape}")

        dim = query.shape[-1]

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states, attention_probs = self._attention(query, key, value)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim)

        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states, attention_probs

    def _attention(self, query, key, value):
        if query.device.type == "mps":
            attention_scores = torch.einsum("b i d, b j d -> b i j", query, key) * self.scale
        else:
            attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attention_probs = attention_scores.softmax(dim=-1)
        if query.device.type == "mps":
            hidden_states = torch.einsum("b i j, b j d -> b i d", attention_probs, value)
        else:
            hidden_states = torch.matmul(attention_probs, value)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states, attention_probs

    def _sliced_attention(self, query, key, value, sequence_length, dim):
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            if query.device.type == "mps":
                attn_slice = (
                    torch.einsum("b i d, b j d -> b i j", query[start_idx:end_idx], key[start_idx:end_idx])
                    * self.scale
                )
            else:
                attn_slice = (
                    torch.matmul(query[start_idx:end_idx], key[start_idx:end_idx].transpose(1, 2)) * self.scale
                )
            attn_slice = attn_slice.softmax(dim=-1)
            if query.device.type == "mps":
                attn_slice = torch.einsum("b i j, b j d -> b i d", attn_slice, value[start_idx:end_idx])
            else:
                attn_slice = torch.matmul(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _memory_efficient_attention_xformers(self, query, key, value):
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "silu",
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "geglu":
            self.activation = GEGLU(dim, inner_dim)
        elif activation_fn == "geglu-approximate":
            self.activation = ApproximateGELU(dim, inner_dim)
        elif activation_fn == "silu":
            self.activation = nn.SiLU()

        self.net = nn.ModuleList([])
        self.net.append(nn.Linear(dim, inner_dim))
        self.net.append(self.activation)
        self.net.append(nn.Dropout(dropout))
        self.net.append(nn.Linear(inner_dim, dim_out))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states

class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def gelu(self, gate):
        if gate.device.type != "mps":
            return F.gelu(gate)
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)

class ApproximateGELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)

class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim, num_embeddings):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep)))
        scale, shift = torch.chunk(emb, 2)
        x = self.norm(x) * (1 + scale) + shift
        return x
