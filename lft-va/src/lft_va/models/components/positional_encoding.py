"""
Positional encoding modules for Transformer models.
 
Transformer 模型的位置编码模块。
"""
 
from __future__ import annotations
 
import math
from typing import Optional
 
import torch
from torch import nn
 
 
class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al., 2017).
 
    正弦位置编码（Attention is All You Need）。
 
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
 
    Parameters
    ----------
    d_model : int
        Model dimension.
        / 模型维度。
    max_len : int
        Maximum sequence length.
        / 最大序列长度。
    dropout : float
        Dropout probability.
        / Dropout 概率。
 
    Examples
    --------
    >>> pe = SinusoidalPositionalEncoding(d_model=256, max_len=1000)
    >>> x = torch.randn(8, 100, 256)  # [B, T, D]
    >>> out = pe(x)  # [B, T, D]
    """
 
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
 
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
 
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
 
        # Register as buffer (not a parameter, but saved with model)
        # Shape: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
 
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, T, D).
            / 输入张量。
 
        Returns
        -------
        torch.Tensor
            Output with positional encoding added, shape (B, T, D).
            / 加上位置编码后的输出。
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)
 
    def get_encoding(self, seq_len: int) -> torch.Tensor:
        """
        Get positional encoding for a given sequence length.
 
        获取指定长度的位置编码。
 
        Parameters
        ----------
        seq_len : int
            Sequence length.
            / 序列长度。
 
        Returns
        -------
        torch.Tensor
            Positional encoding of shape (1, seq_len, d_model).
            / 位置编码。
        """
        return self.pe[:, :seq_len, :]
 
 
class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding.
 
    可学习的位置编码。
 
    Parameters
    ----------
    d_model : int
        Model dimension.
        / 模型维度。
    max_len : int
        Maximum sequence length.
        / 最大序列长度。
    dropout : float
        Dropout probability.
        / Dropout 概率。
 
    Examples
    --------
    >>> pe = LearnablePositionalEncoding(d_model=256, max_len=1000)
    >>> x = torch.randn(8, 100, 256)  # [B, T, D]
    >>> out = pe(x)  # [B, T, D]
    """
 
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
 
        # Learnable positional embeddings
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
 
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, T, D).
            / 输入张量。
 
        Returns
        -------
        torch.Tensor
            Output with positional encoding added, shape (B, T, D).
            / 加上位置编码后的输出。
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)
 
    def get_encoding(self, seq_len: int) -> torch.Tensor:
        """
        Get positional encoding for a given sequence length.
 
        Parameters
        ----------
        seq_len : int
            Sequence length.
            / 序列长度。
 
        Returns
        -------
        torch.Tensor
            Positional encoding of shape (1, seq_len, d_model).
            / 位置编码。
        """
        return self.pe[:, :seq_len, :]
 
 
class TemporalPositionalEncoding(nn.Module):
    """
    Temporal positional encoding with time-aware features.
 
    时间感知的位置编码，支持不规则时间间隔。
 
    This encoding combines:
    1. Sinusoidal position encoding (for relative position)
    2. Optional time delta encoding (for irregular time intervals)
 
    Parameters
    ----------
    d_model : int
        Model dimension.
        / 模型维度。
    max_len : int
        Maximum sequence length.
        / 最大序列长度。
    dropout : float
        Dropout probability.
        / Dropout 概率。
    use_time_delta : bool
        Whether to use time delta features.
        / 是否使用时间差特征。
 
    Examples
    --------
    >>> pe = TemporalPositionalEncoding(d_model=256, use_time_delta=True)
    >>> x = torch.randn(8, 100, 256)  # [B, T, D]
    >>> timestamps = torch.arange(100).float().unsqueeze(0).expand(8, -1)
    >>> out = pe(x, timestamps)  # [B, T, D]
    """
 
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        use_time_delta: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_time_delta = use_time_delta
        self.dropout = nn.Dropout(p=dropout)
 
        # Sinusoidal position encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
 
        # Time delta projection (if enabled)
        if use_time_delta:
            self.time_proj = nn.Linear(1, d_model)
 
    def forward(
        self,
        x: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Add temporal positional encoding to input.
 
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, T, D).
            / 输入张量。
        timestamps : torch.Tensor, optional
            Timestamps of shape (B, T) in seconds.
            / 时间戳（秒）。
 
        Returns
        -------
        torch.Tensor
            Output with positional encoding added, shape (B, T, D).
            / 加上位置编码后的输出。
        """
        seq_len = x.size(1)
 
        # Add sinusoidal position encoding
        x = x + self.pe[:, :seq_len, :]
 
        # Add time delta encoding if enabled and timestamps provided
        if self.use_time_delta and timestamps is not None:
            # Compute time deltas
            time_delta = torch.zeros_like(timestamps)
            time_delta[:, 1:] = timestamps[:, 1:] - timestamps[:, :-1]
            time_delta = time_delta.unsqueeze(-1)  # [B, T, 1]
 
            # Project and add
            time_enc = self.time_proj(time_delta)
            x = x + time_enc
 
        return self.dropout(x)
 
 
if __name__ == "__main__":
    # Demo
    print("Testing Positional Encoding modules...")
 
    # Test sinusoidal
    pe_sin = SinusoidalPositionalEncoding(d_model=256)
    x = torch.randn(4, 100, 256)
    out = pe_sin(x)
    print(f"Sinusoidal PE - Input: {x.shape}, Output: {out.shape}")
 
    # Test learnable
    pe_learn = LearnablePositionalEncoding(d_model=256)
    out = pe_learn(x)
    print(f"Learnable PE - Input: {x.shape}, Output: {out.shape}")
    print(f"Learnable PE parameters: {sum(p.numel() for p in pe_learn.parameters()):,}")
 
    # Test temporal
    pe_temp = TemporalPositionalEncoding(d_model=256, use_time_delta=True)
    timestamps = torch.arange(100).float().unsqueeze(0).expand(4, -1) * 0.2
    out = pe_temp(x, timestamps)
    print(f"Temporal PE - Input: {x.shape}, Output: {out.shape}")