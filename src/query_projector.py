"""
Query Projector for MemoryMAG.

The Query Projector generates query vectors for memory retrieval.
A key innovation is "query refinement" across layers - each layer's
memory output (ltm_out) is passed to the next layer's query projector,
enabling progressive refinement of the search.

Early layers: Broad, syntax-focused queries
Middle layers: Relationship-focused queries  
Late layers: Intent-focused, precise queries
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class QueryProjector(nn.Module):
    """
    Projects hidden states (+ optional previous memory output) into query vectors.
    
    The query projector enables cross-layer query refinement:
    - Takes current hidden state h_t
    - Optionally takes previous layer's memory output (ltm_out_prev)
    - Produces query vector for memory lookup
    
    Args:
        d_model: Model hidden dimension
        d_query: Query dimension (defaults to d_model)
        use_prev_memory: Whether to incorporate previous layer's memory output
        combine_mode: How to combine h and prev_ltm ('concat', 'add', 'gate')
    """
    
    def __init__(
        self,
        d_model: int,
        d_query: Optional[int] = None,
        use_prev_memory: bool = True,
        combine_mode: str = 'gate',
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_query = d_query if d_query is not None else d_model
        self.use_prev_memory = use_prev_memory
        self.combine_mode = combine_mode
        
        if combine_mode == 'concat' and use_prev_memory:
            # Concatenate h and prev_ltm_out, project to query dim
            self.proj = nn.Linear(d_model * 2, self.d_query, bias=False)
        elif combine_mode == 'gate' and use_prev_memory:
            # Separate projections + learned gate
            self.proj_h = nn.Linear(d_model, self.d_query, bias=False)
            self.proj_prev = nn.Linear(d_model, self.d_query, bias=False)
            self.gate = nn.Linear(d_model * 2, self.d_query)
        elif combine_mode == 'add' and use_prev_memory:
            # Separate projections, then add
            self.proj_h = nn.Linear(d_model, self.d_query, bias=False)
            self.proj_prev = nn.Linear(d_model, self.d_query, bias=False)
        else:
            # No previous memory, simple projection
            self.proj = nn.Linear(d_model, self.d_query, bias=False)
        
        # Layer norm for query stability
        self.norm = nn.LayerNorm(self.d_query)
        
        # 1D conv for local context (following Titans architecture)
        self.conv = nn.Conv1d(
            self.d_query, self.d_query, 
            kernel_size=4, padding=3, groups=self.d_query
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_ltm_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate query vectors for memory retrieval.
        
        Args:
            hidden_states: Current layer hidden states [batch, seq, d_model]
            prev_ltm_out: Previous layer's memory output [batch, seq, d_model]
                         (None for first layer or if use_prev_memory=False)
        
        Returns:
            Query vectors [batch, seq, d_query]
        """
        batch, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype
        
        if self.use_prev_memory and prev_ltm_out is not None:
            # Ensure prev_ltm_out matches dtype
            prev_ltm_out = prev_ltm_out.to(dtype=dtype)
            
            if self.combine_mode == 'concat':
                combined = torch.cat([hidden_states, prev_ltm_out], dim=-1)
                q = self.proj(combined)
                
            elif self.combine_mode == 'gate':
                q_h = self.proj_h(hidden_states)
                q_prev = self.proj_prev(prev_ltm_out)
                
                # Compute gate from both inputs
                gate_input = torch.cat([hidden_states, prev_ltm_out], dim=-1)
                g = torch.sigmoid(self.gate(gate_input))
                
                # Gated combination
                q = (1 - g) * q_h + g * q_prev
                
            elif self.combine_mode == 'add':
                q_h = self.proj_h(hidden_states)
                q_prev = self.proj_prev(prev_ltm_out)
                q = q_h + q_prev
        else:
            if hasattr(self, 'proj'):
                q = self.proj(hidden_states)
            else:
                # Fallback if use_prev_memory but no prev provided
                q = self.proj_h(hidden_states)
        
        # Apply conv for local context
        q = q.transpose(1, 2)  # [batch, d_query, seq]
        q = self.conv(q)[:, :, :seq_len]  # Trim to original length
        q = q.transpose(1, 2)  # [batch, seq, d_query]
        
        # Normalize and ensure dtype
        q = self.norm(q).to(dtype=dtype)
        
        return q


class MultiHeadQueryProjector(nn.Module):
    """
    Multi-head version of QueryProjector for attending to different aspects.
    
    Different heads can specialize in different types of queries:
    - Some heads for exact match (syntax)
    - Some heads for semantic similarity
    - Some heads for structural patterns
    
    Args:
        d_model: Model hidden dimension
        n_heads: Number of query heads
        d_head: Dimension per head (defaults to d_model // n_heads)
        use_prev_memory: Whether to use previous layer's memory output
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        d_head: Optional[int] = None,
        use_prev_memory: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head if d_head is not None else d_model // n_heads
        self.use_prev_memory = use_prev_memory
        
        # Each head has its own projector
        self.head_projectors = nn.ModuleList([
            QueryProjector(
                d_model=d_model,
                d_query=self.d_head,
                use_prev_memory=use_prev_memory,
                combine_mode='gate'
            )
            for _ in range(n_heads)
        ])
        
        # Output projection to combine heads
        self.out_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_ltm_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate multi-head query vectors.
        
        Args:
            hidden_states: [batch, seq, d_model]
            prev_ltm_out: [batch, seq, d_model] or None
            
        Returns:
            Combined query vectors [batch, seq, d_model]
        """
        # Generate queries from each head
        head_queries = [
            proj(hidden_states, prev_ltm_out) 
            for proj in self.head_projectors
        ]
        
        # Concatenate heads
        combined = torch.cat(head_queries, dim=-1)  # [batch, seq, n_heads * d_head]
        
        # Project to output dimension
        output = self.out_proj(combined)
        
        return output
