"""
Model Patching Utilities for MemoryMAG.

This module provides utilities to patch pre-trained models (specifically Qwen3)
with MAG components. The patching strategy:

1. Load pre-trained model
2. Identify decoder layers
3. Wrap each layer with MAGDecoderLayer
4. Freeze original weights
5. Initialize trainable MAG components
6. Modify forward pass for ltm_out propagation
"""

import torch
import torch.nn as nn
from typing import Optional, List, Any
from transformers import AutoModelForCausalLM
import logging

from .mag_layer import MAGGate
from .neural_memory import NeuralMemory
from .query_projector import QueryProjector

logger = logging.getLogger(__name__)


class Qwen3MAGConfig:
    """Configuration for Qwen3 MAG patching."""
    
    def __init__(
        self,
        d_memory: Optional[int] = None,
        memory_layers: int = 2,
        chunk_size: int = 64,
        gate_init_bias: float = -2.0,
        n_persistent_tokens: int = 16,
        layers_to_patch: Optional[List[int]] = None,
        query_combine_mode: str = 'gate',
        freeze_embeddings: bool = True,
        freeze_lm_head: bool = True,
    ):
        self.d_memory = d_memory
        self.memory_layers = memory_layers
        self.chunk_size = chunk_size
        self.gate_init_bias = gate_init_bias
        self.n_persistent_tokens = n_persistent_tokens
        self.layers_to_patch = layers_to_patch
        self.query_combine_mode = query_combine_mode
        self.freeze_embeddings = freeze_embeddings
        self.freeze_lm_head = freeze_lm_head


class Qwen3MAGDecoderLayer(nn.Module):
    """MAG-augmented decoder layer for Qwen3."""
    
    def __init__(
        self,
        original_layer: nn.Module,
        d_model: int,
        layer_idx: int,
        config: Qwen3MAGConfig,
        n_layers_total: int,
        parent_model: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        self.layer_idx = layer_idx
        self.d_model = d_model
        self.parent_model = parent_model  # Reference to Qwen3MAGModel for LTM buffer
        
        # Copy attributes from original layer that Qwen3 expects
        if hasattr(original_layer, 'attention_type'):
            self.attention_type = original_layer.attention_type
        if hasattr(original_layer, 'is_sliding'):
            self.is_sliding = original_layer.is_sliding
        
        # Store original components
        self.self_attn = original_layer.self_attn
        self.mlp = original_layer.mlp
        self.input_layernorm = original_layer.input_layernorm
        self.post_attention_layernorm = original_layer.post_attention_layernorm
        
        # Freeze original components
        for param in self.self_attn.parameters():
            param.requires_grad = False
        for param in self.mlp.parameters():
            param.requires_grad = False
        for param in self.input_layernorm.parameters():
            param.requires_grad = False
        for param in self.post_attention_layernorm.parameters():
            param.requires_grad = False
        
        # MAG Components (trainable)
        d_memory = config.d_memory if config.d_memory else d_model * 2
        
        self.neural_memory = NeuralMemory(
            d_model=d_model,
            d_memory=d_memory,
            n_layers=config.memory_layers,
            chunk_size=config.chunk_size,
        )
        
        self.query_projector = QueryProjector(
            d_model=d_model,
            use_prev_memory=(layer_idx > 0),
            combine_mode=config.query_combine_mode,
        )
        
        self.gate = MAGGate(d_model=d_model, init_bias=config.gate_init_bias)
        self.memory_norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # Get previous layer's LTM output from shared buffer (if available)
        prev_ltm_out = None
        if self.parent_model is not None and self.layer_idx > 0:
            # Find the most recent MAG layer before this one
            for prev_idx in range(self.layer_idx - 1, -1, -1):
                if prev_idx in self.parent_model._ltm_buffer:
                    prev_ltm_out = self.parent_model._ltm_buffer[prev_idx]
                    break
        
        dtype = hidden_states.dtype
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Attention path (frozen)
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        
        if isinstance(attn_outputs, tuple):
            attn_out = attn_outputs[0]
            present_key_value = attn_outputs[1] if len(attn_outputs) > 1 else None
            attn_weights = attn_outputs[2] if output_attentions and len(attn_outputs) > 2 else None
        else:
            attn_out = attn_outputs
            present_key_value = None
            attn_weights = None
        
        # Memory path (trainable) - ensure dtype consistency
        query = self.query_projector(hidden_states, prev_ltm_out)
        ltm_out, _ = self.neural_memory(query, update_memory=True, return_surprise=True)
        ltm_out = self.memory_norm(ltm_out).to(dtype=dtype)
        
        # Store LTM output in shared buffer for next MAG layer
        if self.parent_model is not None:
            self.parent_model._ltm_buffer[self.layer_idx] = ltm_out
        
        # Gated combination
        combined, gate_values = self.gate(hidden_states, attn_out, ltm_out)
        hidden_states = residual + combined.to(dtype=dtype)
        
        # Store gate values for diagnostics
        if self.parent_model is not None:
            self.parent_model._gate_values_buffer[self.layer_idx] = gate_values
        
        # MLP (frozen)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        # Return in standard HuggingFace format (same as original Qwen3DecoderLayer)
        # This ensures compatibility with non-patched layers and gradient checkpointing
        outputs = (hidden_states,)
        if use_cache:
            outputs = outputs + (present_key_value,)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        
        return outputs
    
    def reset_memory(self):
        self.neural_memory.reset_memory()


class Qwen3MAGModel(nn.Module):
    """Complete Qwen3 model with MAG augmentation."""
    
    def __init__(self, base_model: nn.Module, config: Qwen3MAGConfig, gradient_checkpointing: bool = False):
        super().__init__()
        
        self.base_model = base_model
        self.mag_config = config
        self.model_config = base_model.config
        self.d_model = self.model_config.hidden_size
        self.n_layers = self.model_config.num_hidden_layers
        self.vocab_size = self.model_config.vocab_size
        
        if config.layers_to_patch is None:
            self.layers_to_patch = list(range(self.n_layers))
        else:
            self.layers_to_patch = config.layers_to_patch
        
        if config.n_persistent_tokens > 0:
            self.persistent_memory = nn.Parameter(
                torch.randn(1, config.n_persistent_tokens, self.d_model) * 0.02
            )
        else:
            self.persistent_memory = None
        
        # Shared buffer for cross-layer LTM communication
        # MAG layers write their ltm_out here, and read previous layer's ltm_out
        # This avoids changing the return format of decoder layers
        self._ltm_buffer = {}
        self._gate_values_buffer = {}
        
        self._patch_layers()
        self._freeze_base_components()
        
        # Enable gradient checkpointing for memory efficiency
        if gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()
    
    def _patch_layers(self):
        if hasattr(self.base_model, 'model'):
            decoder_layers = self.base_model.model.layers
        else:
            decoder_layers = self.base_model.layers
        
        # Get dtype from base model
        dtype = next(self.base_model.parameters()).dtype
        device = next(self.base_model.parameters()).device
        
        self.mag_layers = nn.ModuleList()
        
        for idx in range(self.n_layers):
            if idx in self.layers_to_patch:
                original_layer = decoder_layers[idx]
                mag_layer = Qwen3MAGDecoderLayer(
                    original_layer=original_layer,
                    d_model=self.d_model,
                    layer_idx=idx,
                    config=self.mag_config,
                    n_layers_total=self.n_layers,
                    parent_model=self,  # Pass reference for LTM buffer access
                )
                # Match dtype and device of base model
                mag_layer = mag_layer.to(dtype=dtype, device=device)
                self.mag_layers.append(mag_layer)
                decoder_layers[idx] = mag_layer
            else:
                self.mag_layers.append(None)
    
    def _freeze_base_components(self):
        if self.mag_config.freeze_embeddings:
            embed = self.base_model.model.embed_tokens if hasattr(self.base_model, 'model') else self.base_model.embed_tokens
            for param in embed.parameters():
                param.requires_grad = False
        
        if self.mag_config.freeze_lm_head:
            for param in self.base_model.lm_head.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Any]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        if inputs_embeds is None:
            embed = self.base_model.model.embed_tokens if hasattr(self.base_model, 'model') else self.base_model.embed_tokens
            inputs_embeds = embed(input_ids)
        
        batch_size, seq_len, _ = inputs_embeds.shape
        dtype = inputs_embeds.dtype
        
        if self.persistent_memory is not None:
            # Ensure persistent memory matches input dtype
            persistent = self.persistent_memory.to(dtype=dtype).expand(batch_size, -1, -1)
            inputs_embeds = torch.cat([persistent, inputs_embeds], dim=1)
            
            if attention_mask is not None:
                persistent_mask = torch.ones(
                    batch_size, self.mag_config.n_persistent_tokens,
                    device=attention_mask.device, dtype=attention_mask.dtype
                )
                attention_mask = torch.cat([persistent_mask, attention_mask], dim=1)
        
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        
        if self.persistent_memory is not None and hasattr(outputs, 'logits'):
            outputs.logits = outputs.logits[:, self.mag_config.n_persistent_tokens:, :]
        
        if labels is not None:
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1)
            )
            if hasattr(outputs, 'loss'):
                outputs.loss = loss
        
        return outputs
    
    def reset_all_memory(self):
        # Clear the LTM communication buffers
        self._ltm_buffer.clear()
        self._gate_values_buffer.clear()
        
        # Reset memory in each MAG layer
        for layer in self.mag_layers:
            if layer is not None:
                layer.reset_memory()
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        params = []
        if self.persistent_memory is not None:
            params.append(self.persistent_memory)
        for layer in self.mag_layers:
            if layer is not None:
                params.extend(layer.neural_memory.parameters())
                params.extend(layer.query_projector.parameters())
                params.extend(layer.gate.parameters())
                params.extend(layer.memory_norm.parameters())
        return params
    
    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.get_trainable_parameters())


def patch_qwen3_with_mag(
    model_name_or_path: str = "Qwen/Qwen3-1.7B",
    config: Optional[Qwen3MAGConfig] = None,
    device: str = "auto",
    dtype: torch.dtype = torch.bfloat16,
    gradient_checkpointing: bool = False,
    attn_implementation: str = "eager",
) -> Qwen3MAGModel:
    """Load Qwen3 and patch with MAG components.
    
    Args:
        model_name_or_path: HuggingFace model identifier or local path
        config: MAG configuration
        device: Device to load model on ("auto", "cuda", "cpu")
        dtype: Model dtype (torch.bfloat16, torch.float16, etc.)
        gradient_checkpointing: Enable gradient checkpointing to reduce memory usage
        attn_implementation: Attention implementation ("eager", "sdpa", "flash_attention_2")
            - "eager": Standard attention (high memory, O(n²))
            - "sdpa": PyTorch scaled_dot_product_attention (memory efficient when available)  
            - "flash_attention_2": Flash Attention 2 (requires flash-attn package)
    """
    if config is None:
        config = Qwen3MAGConfig()
    
    logger.info(f"Loading base model: {model_name_or_path}")
    logger.info(f"Using attention implementation: {attn_implementation}")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation=attn_implementation,
    )
    
    logger.info(f"Base model params: {sum(p.numel() for p in base_model.parameters()):,}")
    
    mag_model = Qwen3MAGModel(base_model, config, gradient_checkpointing=gradient_checkpointing)
    
    trainable = mag_model.count_trainable_parameters()
    total = sum(p.numel() for p in mag_model.parameters())
    
    logger.info(f"MAG model created. Trainable: {trainable:,} / Total: {total:,}")
    if gradient_checkpointing:
        logger.info("Gradient checkpointing enabled for memory efficiency")
    
    return mag_model
