"""
Neural Long-Term Memory Module for MemoryMAG.

This implements the core Neural Memory Module (NMM) from the Titans paper.
The memory is a deep MLP that learns to memorize at test time through
gradient-based updates driven by a "surprise" metric.

Key concepts:
- Memory is stored in the weights of a 2+ layer MLP
- Surprise = reconstruction error (how unexpected is the input)
- Updates use gradient descent with momentum and weight decay
- Momentum captures context around surprising events
- Weight decay provides adaptive forgetting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class NeuralMemory(nn.Module):
    """
    Deep Neural Long-Term Memory Module.
    
    The memory is parameterized as a multi-layer MLP. Information is:
    - Written via gradient descent on reconstruction loss (surprise-based)
    - Read via forward pass without weight updates
    
    This implements the MAG (Memory as Gate) variant's memory component.
    
    Args:
        d_model: Hidden dimension of the base model
        d_memory: Hidden dimension inside the memory MLP (typically 2-4x d_model)
        n_layers: Number of layers in the memory MLP (default: 2, paper shows deeper is better)
        momentum: Momentum coefficient for surprise accumulation (η in paper)
        lr: Learning rate for memory updates (θ in paper)
        weight_decay: Forgetting factor (α in paper, applied as 1-α)
        learnable_params: Whether η, θ, α are learnable or fixed
        surprise_threshold: Skip updates when surprise is below this threshold (0 disables)
        max_update_norm: Clip update gradients by this norm (None disables)
        update_dtype: Dtype for fast updates (defaults to float32 for stability)
    """
    
    def __init__(
        self,
        d_model: int,
        d_memory: Optional[int] = None,
        n_layers: int = 2,
        momentum: float = 0.9,
        lr: float = 0.1,
        weight_decay: float = 0.01,
        learnable_params: bool = True,
        chunk_size: int = 64,
        surprise_threshold: float = 0.0,
        max_update_norm: Optional[float] = None,
        update_dtype: Optional[torch.dtype] = torch.float32,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_memory = d_memory if d_memory is not None else d_model * 2
        self.n_layers = n_layers
        self.chunk_size = chunk_size
        self.surprise_threshold = surprise_threshold
        self.max_update_norm = max_update_norm
        self.update_dtype = update_dtype
        
        # Build the memory MLP layers
        # Structure: d_model -> d_memory -> ... -> d_model
        layers = []
        in_dim = d_model
        for i in range(n_layers):
            out_dim = d_model if i == n_layers - 1 else self.d_memory
            layers.append(nn.Linear(in_dim, out_dim, bias=False))
            in_dim = out_dim
        self.memory_layers = nn.ModuleList(layers)
        
        # Initialize memory weights to approximate identity mapping
        # This ensures initial retrieval returns something close to the query
        self._init_identity_like()
        
        # Learnable or fixed hyperparameters for memory updates
        if learnable_params:
            # These will be learned during outer-loop training
            # Using sigmoid to constrain to [0, 1]
            self.log_momentum = nn.Parameter(torch.tensor(math.log(momentum / (1 - momentum + 1e-8))))
            self.log_lr = nn.Parameter(torch.tensor(math.log(lr)))
            self.log_weight_decay = nn.Parameter(torch.tensor(math.log(weight_decay / (1 - weight_decay + 1e-8))))
        else:
            self.register_buffer('momentum', torch.tensor(momentum))
            self.register_buffer('lr', torch.tensor(lr))
            self.register_buffer('weight_decay', torch.tensor(weight_decay))
        
        self.learnable_params = learnable_params
        
        # Key and Value projections for associative memory objective
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Convolution for local context (following modern linear RNNs)
        self.conv_k = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)
        self.conv_v = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)
        self.conv_q = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)
        
        # For tracking momentum state during forward pass
        self._momentum_state = None
        
    def _init_identity_like(self):
        """Initialize memory to approximate identity mapping with small noise."""
        for i, layer in enumerate(self.memory_layers):
            with torch.no_grad():
                if layer.weight.shape[0] == layer.weight.shape[1]:
                    # Square matrix: initialize close to identity
                    nn.init.eye_(layer.weight)
                    layer.weight.add_(torch.randn_like(layer.weight) * 0.01)
                else:
                    # Non-square: use xavier instead of orthogonal (avoids LAPACK requirement)
                    nn.init.xavier_uniform_(layer.weight)
                    layer.weight.mul_(0.1)
    
    @property
    def eta(self) -> torch.Tensor:
        """Momentum coefficient (past surprise decay)."""
        if self.learnable_params:
            return torch.sigmoid(self.log_momentum)
        return self.momentum
    
    @property 
    def theta(self) -> torch.Tensor:
        """Learning rate for momentary surprise."""
        if self.learnable_params:
            return torch.exp(self.log_lr).clamp(max=1.0)
        return self.lr
    
    @property
    def alpha(self) -> torch.Tensor:
        """Weight decay / forgetting factor."""
        if self.learnable_params:
            return torch.sigmoid(self.log_weight_decay)
        return self.weight_decay
    
    def memory_forward(self, x: torch.Tensor, weights: Optional[list] = None) -> torch.Tensor:
        """
        Forward pass through memory MLP.
        
        Args:
            x: Input tensor [batch, seq, d_model] or [batch, d_model]
            weights: Optional list of weight tensors to use instead of self.memory_layers
                     (used during test-time training with modified weights)
        
        Returns:
            Output tensor same shape as input
        """
        if weights is None:
            weights = [layer.weight for layer in self.memory_layers]
        
        h = x
        for i, w in enumerate(weights):
            h = F.linear(h, w)
            # Apply activation for all layers except the last
            if i < len(weights) - 1:
                h = F.silu(h)
        return h
    
    def compute_surprise(
        self, 
        hidden_states: torch.Tensor,
        weights: Optional[list] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute surprise metric as reconstruction error.
        
        Surprise = ||Memory(key) - value||²
        
        This measures how unexpected the input is given the current memory state.
        High surprise → the memory hasn't seen similar patterns → should memorize.
        
        Args:
            hidden_states: Input hidden states [batch, seq, d_model]
            weights: Optional memory weights (for gradient computation)
            
        Returns:
            surprise: Scalar surprise value
            grad: Gradient of surprise w.r.t. memory weights
        """
        # Project to keys and values
        # Apply conv for local context
        h_t = hidden_states.transpose(1, 2)  # [batch, d_model, seq]
        keys = self.conv_k(self.key_proj(hidden_states).transpose(1, 2))[:, :, :hidden_states.shape[1]].transpose(1, 2)
        values = self.conv_v(self.value_proj(hidden_states).transpose(1, 2))[:, :, :hidden_states.shape[1]].transpose(1, 2)
        
        # Memory reconstruction
        reconstructed = self.memory_forward(keys, weights)
        
        # Surprise = reconstruction error
        surprise = F.mse_loss(reconstructed, values, reduction='none')
        
        return surprise, keys, values
    
    def compute_weight_gradients(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        weights: list
    ) -> list:
        """
        Compute gradients of reconstruction loss w.r.t. memory weights.
        
        For a 2-layer MLP: Memory(k) = W2 @ SiLU(W1 @ k)
        Loss = ||Memory(k) - v||²
        
        We compute gradients analytically for efficiency.
        """
        batch, seq, d = keys.shape
        
        # Forward through memory with intermediate activations
        h = keys
        activations = [keys]
        for i, w in enumerate(weights[:-1]):
            h = F.linear(h, w)
            pre_act = h
            h = F.silu(h)
            activations.append(h)
        
        # Final layer
        output = F.linear(h, weights[-1])
        
        # Error signal
        error = 2 * (output - values)  # [batch, seq, d_model]
        
        # Backprop through layers (reverse order)
        grads = []
        delta = error
        
        for i in range(len(weights) - 1, -1, -1):
            # Gradient for this layer's weight
            # dL/dW = delta^T @ activation
            act = activations[i]  # [batch, seq, in_dim]
            # Sum over batch and seq
            grad_w = torch.einsum('bso,bsi->oi', delta, act) / (batch * seq)
            grads.insert(0, grad_w)
            
            if i > 0:
                # Backprop through this layer
                delta = torch.einsum('bso,oi->bsi', delta, weights[i])
                # Backprop through SiLU: d/dx[x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                pre_silu = F.linear(activations[i-1], weights[i-1]) if i > 1 else F.linear(keys, weights[0])
                sigmoid_x = torch.sigmoid(pre_silu)
                silu_grad = sigmoid_x * (1 + pre_silu * (1 - sigmoid_x))
                delta = delta * silu_grad
        
        return grads
    
    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """
        Retrieve information from memory given a query.
        
        This is a simple forward pass without weight updates.
        
        Args:
            query: Query tensor [batch, seq, d_model]
            
        Returns:
            Retrieved values [batch, seq, d_model]
        """
        # Project query and apply conv
        q = self.conv_q(self.query_proj(query).transpose(1, 2))[:, :, :query.shape[1]].transpose(1, 2)
        
        # Forward through memory (inference mode, no weight update)
        return self.memory_forward(q)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        update_memory: bool = True,
        return_surprise: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process sequence through neural memory.
        
        For each token:
        1. Compute surprise (how unexpected is this input)
        2. Update memory weights if surprise is high (with momentum)
        3. Retrieve relevant information via query
        
        In training mode, we accumulate gradients for the entire sequence
        and update using chunked parallel processing.
        
        Args:
            hidden_states: Input [batch, seq, d_model]
            update_memory: Whether to update memory weights (True during forward, False for pure retrieval)
            return_surprise: Whether to return surprise values for diagnostics
            
        Returns:
            output: Retrieved memory values [batch, seq, d_model]
            surprise: Optional surprise values [batch, seq] if return_surprise=True
        """
        batch, seq_len, d = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        # Get current memory weights (we'll modify copies for TTT)
        update_dtype = self.update_dtype if update_memory else hidden_states.dtype
        if update_dtype is None:
            update_dtype = hidden_states.dtype
        current_weights = [layer.weight.to(update_dtype).clone() for layer in self.memory_layers]
        
        # Initialize momentum state if needed
        if self._momentum_state is None or self._momentum_state[0].shape != current_weights[0].shape:
            self._momentum_state = [torch.zeros_like(w) for w in current_weights]
        
        outputs = []
        surprises = []
        
        # Process in chunks for efficiency
        for chunk_start in range(0, seq_len, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, seq_len)
            chunk = hidden_states[:, chunk_start:chunk_end, :]
            if update_dtype != chunk.dtype:
                chunk = chunk.to(dtype=update_dtype)
            
            if update_memory:
                # Compute surprise and gradients for this chunk
                surprise, keys, values = self.compute_surprise(chunk, current_weights)
                
                if return_surprise:
                    surprises.append(torch.nan_to_num(surprise.mean(dim=-1), nan=0.0, posinf=0.0, neginf=0.0))

                do_update = True
                if self.surprise_threshold and surprise.mean().item() < self.surprise_threshold:
                    do_update = False

                if do_update:
                    # Compute weight gradients
                    grads = self.compute_weight_gradients(keys, values, current_weights)
                
                # Update weights with momentum and weight decay
                # S_t = η * S_{t-1} - θ * ∇L (momentum update)
                # M_t = (1 - α) * M_{t-1} + S_t (weight decay + gradient)
                if do_update:
                    eta = self.eta
                    theta = self.theta
                    alpha = self.alpha
                    
                    for i, (w, g, s) in enumerate(zip(current_weights, grads, self._momentum_state)):
                        g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                        if self.max_update_norm is not None:
                            grad_norm = torch.linalg.vector_norm(g)
                            if torch.isfinite(grad_norm) and grad_norm > self.max_update_norm:
                                g = g * (self.max_update_norm / (grad_norm + 1e-6))
                        # Update momentum
                        s_new = eta * s - theta * g
                        s_new = torch.nan_to_num(s_new, nan=0.0, posinf=0.0, neginf=0.0)
                        self._momentum_state[i] = s_new
                        
                        # Update weights with decay
                        current_weights[i] = (1 - alpha) * w + s_new
            
            # Retrieve using updated weights
            q = self.conv_q(self.query_proj(chunk).transpose(1, 2))[:, :, :chunk.shape[1]].transpose(1, 2)
            output = self.memory_forward(q, current_weights)
            output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
            outputs.append(output)
        
        output = torch.cat(outputs, dim=1).to(dtype=dtype)
        
        if return_surprise:
            surprise = torch.cat(surprises, dim=1) if surprises else None
            return output, surprise
        
        return output, None
    
    def reset_memory(self, reset_weights: bool = False):
        """Reset fast memory state (call between sequences)."""
        self._momentum_state = None
        if reset_weights:
            self._init_identity_like()

    def reinit_memory(self):
        """Reinitialize memory weights to the identity-like default."""
        self._init_identity_like()


class NeuralMemoryWithDataDependentParams(NeuralMemory):
    """
    Neural Memory with input-dependent learning rate, momentum, and decay.
    
    This is the full version where η_t, θ_t, α_t are functions of the input x_t,
    allowing the memory to adaptively control its update behavior based on context.
    """
    
    def __init__(
        self,
        d_model: int,
        d_memory: Optional[int] = None,
        n_layers: int = 2,
        chunk_size: int = 64,
    ):
        # Initialize parent with learnable_params=False since we'll compute them differently
        super().__init__(
            d_model=d_model,
            d_memory=d_memory,
            n_layers=n_layers,
            learnable_params=False,
            chunk_size=chunk_size,
        )
        
        # Networks to produce data-dependent parameters
        # Each takes hidden state and produces a scalar per position
        self.eta_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()  # η ∈ [0, 1]
        )
        
        self.theta_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Softplus()  # θ > 0, but allow larger values
        )
        
        self.alpha_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()  # α ∈ [0, 1]
        )
    
    def get_data_dependent_params(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute input-dependent hyperparameters.
        
        Args:
            x: Input tensor [batch, seq, d_model]
            
        Returns:
            eta: Momentum decay [batch, seq, 1]
            theta: Learning rate [batch, seq, 1] 
            alpha: Weight decay [batch, seq, 1]
        """
        eta = self.eta_net(x)
        theta = self.theta_net(x).clamp(max=1.0)  # Cap learning rate
        alpha = self.alpha_net(x)
        return eta, theta, alpha
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        update_memory: bool = True,
        return_surprise: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process with data-dependent update parameters.
        """
        batch, seq_len, d = hidden_states.shape
        
        # Get data-dependent parameters for the whole sequence
        eta_all, theta_all, alpha_all = self.get_data_dependent_params(hidden_states)
        
        current_weights = [layer.weight.clone() for layer in self.memory_layers]
        
        if self._momentum_state is None:
            self._momentum_state = [torch.zeros_like(w) for w in current_weights]
        
        outputs = []
        surprises = []
        
        # Process token by token for data-dependent params
        # (Could be optimized with parallel scan for production)
        for t in range(seq_len):
            x_t = hidden_states[:, t:t+1, :]
            eta_t = eta_all[:, t:t+1, :]
            theta_t = theta_all[:, t:t+1, :]
            alpha_t = alpha_all[:, t:t+1, :]
            
            if update_memory:
                surprise, keys, values = self.compute_surprise(x_t, current_weights)
                if return_surprise:
                    surprises.append(surprise.mean(dim=-1))
                
                grads = self.compute_weight_gradients(keys, values, current_weights)
                
                # Use position-specific parameters (average over batch for weight update)
                eta = eta_t.mean()
                theta = theta_t.mean()
                alpha = alpha_t.mean()
                
                for i, (w, g, s) in enumerate(zip(current_weights, grads, self._momentum_state)):
                    s_new = eta * s - theta * g
                    self._momentum_state[i] = s_new
                    current_weights[i] = (1 - alpha) * w + s_new
            
            q = self.conv_q(self.query_proj(x_t).transpose(1, 2))[:, :, :1].transpose(1, 2)
            output = self.memory_forward(q, current_weights)
            outputs.append(output)
        
        output = torch.cat(outputs, dim=1)
        
        if return_surprise:
            surprise = torch.cat(surprises, dim=1) if surprises else None
            return output, surprise
        
        return output, None
