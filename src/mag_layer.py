"""
MAG (Memory as Gate) Layer for MemoryMAG.

This implements the augmented decoder layer that combines:
1. Original attention (frozen) - short-term memory
2. Neural Memory Module - long-term memory  
3. Learned gate - decides when to use memory vs attention

The key insight is that attention excels at precise local dependencies,
while neural memory excels at compressing and retrieving long-range context.
The gate learns when each is most useful.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any

from .neural_memory import NeuralMemory
from .query_projector import QueryProjector


class MAGGate(nn.Module):
    """
    Learned gate for mixing attention and memory outputs.
    
    g = sigmoid(W_gate @ h)
    output = (1 - g) * attn_out + g * ltm_out
    
    The gate learns to:
    - Stay low (use attention) for local dependencies
    - Go high (use memory) for long-range retrieval
    
    Args:
        d_model: Hidden dimension
        init_bias: Initial bias (negative = prefer attention initially)
    """
    
    def __init__(self, d_model: int, init_bias: float = -2.0):
        super().__init__()
        
        self.gate_proj = nn.Linear(d_model, d_model)
        
        # Initialize to prefer attention (memory gate starts low)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, init_bias)
        
        # Learnable scale for gate output
        self.gate_scale = nn.Parameter(torch.ones(d_model))
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attn_out: torch.Tensor,
        ltm_out: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gated mixture of attention and memory outputs.
        
        Args:
            hidden_states: Input to compute gate from [batch, seq, d_model]
            attn_out: Attention output [batch, seq, d_model]
            ltm_out: Memory output [batch, seq, d_model]
            
        Returns:
            output: Gated mixture [batch, seq, d_model]
            gate_values: Gate activations [batch, seq, d_model] (for diagnostics)
        """
        dtype = hidden_states.dtype
        
        # Ensure all inputs have same dtype
        attn_out = attn_out.to(dtype=dtype)
        ltm_out = ltm_out.to(dtype=dtype)
        
        # Compute gate values
        g = torch.sigmoid(self.gate_proj(hidden_states) * self.gate_scale)
        
        # Gated mixture
        output = (1 - g) * attn_out + g * ltm_out
        
        return output, g


class MAGDecoderLayer(nn.Module):
    """
    Memory-Augmented Gate Decoder Layer.
    
    Wraps an original transformer decoder layer with MAG components:
    - Neural Memory for long-term storage/retrieval
    - Query Projector for generating memory queries
    - Gate for mixing attention and memory outputs
    
    Architecture:
    ```
    Input h_in
        |
        +---> Original Attention (frozen) ---> attn_out
        |
        +---> QueryProjector(h_in, prev_ltm_out) ---> query
                    |
                    v
              NeuralMemory(query) ---> ltm_out
        |
        +---> Gate(h_in) ---> g
        |
        v
    h_out = residual + (1-g)*attn_out + g*ltm_out
    
    ltm_out is passed to next layer's QueryProjector
    ```
    
    Args:
        original_layer: The frozen decoder layer to wrap
        d_model: Hidden dimension
        d_memory: Memory MLP hidden dimension
        memory_layers: Number of layers in memory MLP
        layer_idx: Index of this layer (for query refinement logic)
        n_layers_total: Total number of layers in model
    """
    
    def __init__(
        self,
        original_layer: nn.Module,
        d_model: int,
        d_memory: Optional[int] = None,
        memory_layers: int = 2,
        layer_idx: int = 0,
        n_layers_total: int = 28,
        chunk_size: int = 64,
        memory_lr: float = 0.01,
        memory_momentum: float = 0.9,
        memory_weight_decay: float = 0.01,
        memory_learnable_params: bool = True,
        memory_surprise_threshold: float = 0.0,
        memory_max_update_norm: Optional[float] = 1.0,
    ):
        super().__init__()
        
        self.original_layer = original_layer
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.n_layers_total = n_layers_total
        
        # Freeze original layer parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # Neural Memory Module
        self.neural_memory = NeuralMemory(
            d_model=d_model,
            d_memory=d_memory,
            n_layers=memory_layers,
            momentum=memory_momentum,
            lr=memory_lr,
            weight_decay=memory_weight_decay,
            learnable_params=memory_learnable_params,
            chunk_size=chunk_size,
            surprise_threshold=memory_surprise_threshold,
            max_update_norm=memory_max_update_norm,
        )
        
        # Query Projector (uses prev layer's ltm_out for refinement)
        self.query_projector = QueryProjector(
            d_model=d_model,
            use_prev_memory=(layer_idx > 0),  # First layer has no prev memory
            combine_mode='gate',
        )
        
        # Gate for mixing attention and memory
        self.gate = MAGGate(d_model=d_model)
        
        # Normalization layers
        self.memory_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        prev_ltm_out: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[Any]]:
        """
        Forward pass through MAG decoder layer.
        
        Args:
            hidden_states: Input tensor [batch, seq, d_model]
            attention_mask: Attention mask for original layer
            position_ids: Position IDs for original layer
            prev_ltm_out: Previous layer's memory output for query refinement
            past_key_value: KV cache for original layer
            output_attentions: Whether to output attention weights
            use_cache: Whether to use/return KV cache
            
        Returns:
            hidden_states: Output tensor [batch, seq, d_model]
            ltm_out: Memory output for next layer [batch, seq, d_model]
            gate_values: Gate activations [batch, seq, d_model]
            past_key_value: Updated KV cache if use_cache=True
        """
        residual = hidden_states
        
        # 1. Pass through original (frozen) attention layer
        # Handle different return formats from different model architectures
        original_outputs = self.original_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        
        # Extract attention output (handle tuple vs single tensor)
        if isinstance(original_outputs, tuple):
            attn_out = original_outputs[0]
            past_key_value = original_outputs[1] if use_cache and len(original_outputs) > 1 else None
        else:
            attn_out = original_outputs
            past_key_value = None
        
        # 2. Generate query for memory lookup
        query = self.query_projector(hidden_states, prev_ltm_out)
        
        # 3. Memory retrieval and update
        ltm_out, surprise = self.neural_memory(query, update_memory=True, return_surprise=True)
        ltm_out = self.memory_norm(ltm_out)
        ltm_out = torch.nan_to_num(ltm_out, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 4. Gate mixing
        combined, gate_values = self.gate(hidden_states, attn_out, ltm_out)
        
        # 5. Residual connection
        output = residual + combined
        output = self.output_norm(output)
        
        return output, ltm_out, gate_values, past_key_value
    
    def reset_memory(self):
        """Reset memory state (call between sequences)."""
        self.neural_memory.reset_memory()


class MAGModelWrapper(nn.Module):
    """
    Wrapper to convert a full transformer model to MAG architecture.
    
    This handles:
    - Wrapping each decoder layer with MAG components
    - Propagating ltm_out between layers for query refinement
    - Managing persistent memory (prepended to sequence)
    
    Args:
        model: The original transformer model
        d_model: Hidden dimension
        d_memory: Memory MLP hidden dimension
        memory_layers: Number of layers in memory MLP
        n_persistent_tokens: Number of persistent memory tokens
    """
    
    def __init__(
        self,
        model: nn.Module,
        d_model: int,
        d_memory: Optional[int] = None,
        memory_layers: int = 2,
        n_persistent_tokens: int = 16,
        chunk_size: int = 64,
        memory_lr: float = 0.01,
        memory_momentum: float = 0.9,
        memory_weight_decay: float = 0.01,
        memory_learnable_params: bool = True,
        memory_surprise_threshold: float = 0.0,
        memory_max_update_norm: Optional[float] = 1.0,
        layers_to_wrap: Optional[list] = None,  # Which layers to wrap (None = all)
    ):
        super().__init__()
        
        self.model = model
        self.d_model = d_model
        self.n_persistent_tokens = n_persistent_tokens
        
        # Persistent memory tokens (learned, input-independent)
        if n_persistent_tokens > 0:
            self.persistent_memory = nn.Parameter(
                torch.randn(1, n_persistent_tokens, d_model) * 0.02
            )
        else:
            self.persistent_memory = None
        
        # Find and wrap decoder layers
        self.mag_layers = nn.ModuleList()
        self._wrap_decoder_layers(
            d_memory=d_memory,
            memory_layers=memory_layers,
            chunk_size=chunk_size,
            memory_lr=memory_lr,
            memory_momentum=memory_momentum,
            memory_weight_decay=memory_weight_decay,
            memory_learnable_params=memory_learnable_params,
            memory_surprise_threshold=memory_surprise_threshold,
            memory_max_update_norm=memory_max_update_norm,
            layers_to_wrap=layers_to_wrap,
        )
        
    def _wrap_decoder_layers(
        self, 
        d_memory: Optional[int],
        memory_layers: int,
        chunk_size: int,
        memory_lr: float,
        memory_momentum: float,
        memory_weight_decay: float,
        memory_learnable_params: bool,
        memory_surprise_threshold: float,
        memory_max_update_norm: Optional[float],
        layers_to_wrap: Optional[list],
    ):
        """Find decoder layers in model and wrap with MAG components."""
        # This needs to be adapted for specific model architecture
        # For Qwen3, decoder layers are in model.model.layers
        
        decoder_layers = self._find_decoder_layers()
        n_layers = len(decoder_layers)
        
        if layers_to_wrap is None:
            layers_to_wrap = list(range(n_layers))
        
        for idx, layer in enumerate(decoder_layers):
            if idx in layers_to_wrap:
                mag_layer = MAGDecoderLayer(
                    original_layer=layer,
                    d_model=self.d_model,
                    d_memory=d_memory,
                    memory_layers=memory_layers,
                    layer_idx=idx,
                    n_layers_total=n_layers,
                    chunk_size=chunk_size,
                    memory_lr=memory_lr,
                    memory_momentum=memory_momentum,
                    memory_weight_decay=memory_weight_decay,
                    memory_learnable_params=memory_learnable_params,
                    memory_surprise_threshold=memory_surprise_threshold,
                    memory_max_update_norm=memory_max_update_norm,
                )
                self.mag_layers.append(mag_layer)
                # Replace in original model
                self._replace_layer(idx, mag_layer)
            else:
                self.mag_layers.append(None)  # Keep original
    
    def _find_decoder_layers(self) -> list:
        """Find decoder layers in model. Override for specific architectures."""
        # Try common paths
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return list(self.model.model.layers)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return list(self.model.transformer.h)
        elif hasattr(self.model, 'layers'):
            return list(self.model.layers)
        else:
            raise ValueError("Could not find decoder layers in model")
    
    def _replace_layer(self, idx: int, new_layer: nn.Module):
        """Replace a decoder layer with MAG version."""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            self.model.model.layers[idx] = new_layer
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            self.model.transformer.h[idx] = new_layer
        elif hasattr(self.model, 'layers'):
            self.model.layers[idx] = new_layer
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_gate_values: bool = False,
        **kwargs,
    ):
        """
        Forward pass through MAG-augmented model.
        
        Handles persistent memory prepending and ltm_out propagation.
        """
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        batch_size, seq_len, _ = inputs_embeds.shape
        
        # Prepend persistent memory
        if self.persistent_memory is not None:
            persistent = self.persistent_memory.expand(batch_size, -1, -1)
            inputs_embeds = torch.cat([persistent, inputs_embeds], dim=1)
            
            # Adjust attention mask
            if attention_mask is not None:
                persistent_mask = torch.ones(
                    batch_size, self.n_persistent_tokens, 
                    device=attention_mask.device, dtype=attention_mask.dtype
                )
                attention_mask = torch.cat([persistent_mask, attention_mask], dim=1)
        
        # Forward through model (MAG layers will handle ltm propagation)
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )
        
        # Remove persistent memory from output if present
        if self.persistent_memory is not None and hasattr(outputs, 'logits'):
            outputs.logits = outputs.logits[:, self.n_persistent_tokens:, :]
        
        return outputs
    
    def reset_all_memory(self):
        """Reset memory state in all MAG layers."""
        for layer in self.mag_layers:
            if layer is not None:
                layer.reset_memory()
    
    def get_trainable_parameters(self):
        """Get only the trainable (MAG) parameters."""
        params = []
        
        # Persistent memory
        if self.persistent_memory is not None:
            params.append(self.persistent_memory)
        
        # MAG layer parameters
        for layer in self.mag_layers:
            if layer is not None:
                # Neural memory
                params.extend(layer.neural_memory.parameters())
                # Query projector
                params.extend(layer.query_projector.parameters())
                # Gate
                params.extend(layer.gate.parameters())
                # Norms
                params.extend(layer.memory_norm.parameters())
                params.extend(layer.output_norm.parameters())
        
        return params
    
    def freeze_base_model(self):
        """Ensure base model parameters are frozen."""
        for name, param in self.model.named_parameters():
            # Only freeze if not part of MAG components
            if 'neural_memory' not in name and 'query_projector' not in name and 'gate' not in name:
                param.requires_grad = False
