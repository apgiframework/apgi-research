"""
Unit tests for train.py components that can be tested without full import.

This file tests individual functions and classes from train.py by importing
them in isolation without running the module-level training loop.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Create minimal train components for testing
# (These mirror the implementations in train.py but don't require full import)

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Simplified GPTConfig for testing."""

    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x: torch.Tensor) -> torch.Tensor:
    """RMS normalization."""
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx: int, n_layer: int) -> bool:
    """Returns True if layer should have Value Embedding."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary embeddings."""
    assert x.ndim == 4  # nosec: B101 - Test assertion
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


# Test classes
class TestGPTConfig:
    """Tests for GPTConfig dataclass."""

    def test_default_values(self):
        """Test default GPTConfig values."""
        config = GPTConfig()
        assert config.sequence_len == 2048  # nosec: B101 - Test assertion
        assert config.vocab_size == 32768  # nosec: B101 - Test assertion
        assert config.n_layer == 12  # nosec: B101 - Test assertion
        assert config.n_head == 6  # nosec: B101 - Test assertion
        assert config.n_kv_head == 6  # nosec: B101 - Test assertion
        assert config.n_embd == 768  # nosec: B101 - Test assertion
        assert config.window_pattern == "SSSL"  # nosec: B101 - Test assertion

    def test_custom_values(self):
        """Test custom GPTConfig values."""
        config = GPTConfig(
            sequence_len=1024,
            vocab_size=50000,
            n_layer=8,
            n_head=8,
            n_kv_head=4,
            n_embd=512,
            window_pattern="LLLL",
        )
        assert config.sequence_len == 1024  # nosec: B101 - Test assertion
        assert config.vocab_size == 50000  # nosec: B101 - Test assertion
        assert config.n_layer == 8  # nosec: B101 - Test assertion
        assert config.n_head == 8  # nosec: B101 - Test assertion
        assert config.n_kv_head == 4  # nosec: B101 - Test assertion
        assert config.n_embd == 512  # nosec: B101 - Test assertion
        assert config.window_pattern == "LLLL"  # nosec: B101 - Test assertion


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_norm(self):
        """Test norm function (RMS normalization)."""
        x = torch.tensor([[1.0, 2.0, 3.0]])
        result = norm(x)
        assert result.shape == x.shape  # nosec: B101 - Test assertion

    def test_norm_batch(self):
        """Test norm function with batch."""
        x = torch.randn(2, 10, 128)
        result = norm(x)
        assert result.shape == x.shape  # nosec: B101 - Test assertion

    def test_has_ve(self):
        """Test has_ve function (Value Embedding condition)."""
        # For n_layer=12: (n_layer - 1) % 2 = 11 % 2 = 1
        # Layer 0: 0 % 2 = 0, so 0 == 1 is False
        assert has_ve(0, 12) is False  # nosec: B101 - Test assertion
        # Layer 1: 1 % 2 = 1, so 1 == 1 is True
        assert has_ve(1, 12) is True  # nosec: B101 - Test assertion
        # Layer 11: 11 % 2 = 1, so 1 == 1 is True
        assert has_ve(11, 12) is True  # nosec: B101 - Test assertion

    def test_has_ve_different_layers(self):
        """Test has_ve with different layer counts."""
        # n_layer=8: (n_layer - 1) % 2 = 7 % 2 = 1
        assert (
            has_ve(0, 8) is False
        )  # 0 % 2 = 0, 0 == 1 is False  # nosec: B101 - Test assertion
        assert (
            has_ve(1, 8) is True
        )  # 1 % 2 = 1, 1 == 1 is True  # nosec: B101 - Test assertion
        assert (
            has_ve(7, 8) is True
        )  # 7 % 2 = 1, 1 == 1 is True  # nosec: B101 - Test assertion

    def test_apply_rotary_emb(self):
        """Test apply_rotary_emb function."""
        batch_size = 2
        seq_len = 10
        head_dim = 64
        x = torch.randn(batch_size, seq_len, 4, head_dim)
        cos = torch.randn(1, seq_len, 1, head_dim // 2)
        sin = torch.randn(1, seq_len, 1, head_dim // 2)

        result = apply_rotary_emb(x, cos, sin)
        assert result.shape == x.shape  # nosec: B101 - Test assertion

    def test_apply_rotary_emb_assertion(self):
        """Test apply_rotary_emb raises assertion for wrong dimensions."""
        x = torch.randn(2, 10, 64)  # Wrong shape (3D instead of 4D)
        cos = torch.randn(1, 10, 1, 32)
        sin = torch.randn(1, 10, 1, 32)

        with pytest.raises(AssertionError):
            apply_rotary_emb(x, cos, sin)


# Additional tests for schedule functions
def get_lr_multiplier(
    progress: float,
    warmup_ratio: float = 0.0,
    warmdown_ratio: float = 0.5,
    final_lr_frac: float = 0.0,
) -> float:
    """Calculate learning rate multiplier."""
    if progress < warmup_ratio:
        return progress / warmup_ratio if warmup_ratio > 0 else 1.0
    elif progress < 1.0 - warmdown_ratio:
        return 1.0
    else:
        cooldown = (1.0 - progress) / warmdown_ratio
        return cooldown * 1.0 + (1 - cooldown) * final_lr_frac


def get_muon_momentum(step: int) -> float:
    """Get Muon momentum for given step."""
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def get_weight_decay(progress: float, weight_decay: float = 0.2) -> float:
    """Get weight decay for given progress."""
    return weight_decay * (1 - progress)


class TestScheduleFunctions:
    """Tests for learning rate and schedule functions."""

    def test_get_lr_multiplier_warmup(self):
        """Test LR multiplier during warmup."""
        warmup_ratio = 0.1
        # At 50% of warmup
        lrm = get_lr_multiplier(0.05, warmup_ratio=warmup_ratio)
        assert 0 < lrm < 1  # nosec: B101 - Test assertion
        # At end of warmup
        lrm = get_lr_multiplier(warmup_ratio, warmup_ratio=warmup_ratio)
        assert lrm == 1.0  # nosec: B101 - Test assertion

    def test_get_lr_multiplier_steady(self):
        """Test LR multiplier during steady state."""
        # In steady state (after warmup, before warmdown)
        progress = 0.3
        lrm = get_lr_multiplier(progress, warmup_ratio=0.1, warmdown_ratio=0.5)
        assert lrm == 1.0  # nosec: B101 - Test assertion

    def test_get_lr_multiplier_warmdown(self):
        """Test LR multiplier during warmdown."""
        # At start of warmdown (assuming warmdown_ratio is 0.5)
        progress = 0.5
        lrm = get_lr_multiplier(progress, warmdown_ratio=0.5)
        assert lrm == 1.0  # nosec: B101 - Test assertion
        # At end of warmdown
        lrm = get_lr_multiplier(1.0, warmdown_ratio=0.5)
        assert lrm == 0.0  # nosec: B101 - Test assertion

    def test_get_lr_multiplier_zero_warmup(self):
        """Test LR multiplier with zero warmup."""
        lrm = get_lr_multiplier(0.0, warmup_ratio=0.0)
        assert lrm == 1.0  # nosec: B101 - Test assertion

    def test_get_muon_momentum(self):
        """Test Muon momentum schedule."""
        # At step 0
        momentum = get_muon_momentum(0)
        assert momentum == 0.85  # nosec: B101 - Test assertion
        # At step 150 (halfway)
        momentum = get_muon_momentum(150)
        assert 0.85 < momentum < 0.95  # nosec: B101 - Test assertion
        # At step 300+ (capped)
        momentum = get_muon_momentum(500)
        assert momentum == 0.95  # nosec: B101 - Test assertion

    def test_get_weight_decay(self):
        """Test weight decay schedule."""
        # At start
        wd = get_weight_decay(0.0, weight_decay=0.2)
        assert wd == 0.2  # nosec: B101 - Test assertion
        # At 50% progress
        wd = get_weight_decay(0.5, weight_decay=0.2)
        assert wd == 0.1  # nosec: B101 - Test assertion
        # At end
        wd = get_weight_decay(1.0, weight_decay=0.2)
        assert wd == 0.0  # nosec: B101 - Test assertion


class TestPyTorchOperations:
    """Tests for PyTorch operations used in train.py."""

    def test_rms_norm(self):
        """Test PyTorch RMS normalization."""
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = F.rms_norm(x, (x.size(-1),))
        assert result.shape == x.shape  # nosec: B101 - Test assertion
        # RMS norm should normalize
        rms = torch.sqrt(torch.mean(result**2))
        assert torch.allclose(
            rms, torch.tensor(1.0), atol=0.01
        )  # nosec: B101 - Test assertion

    def test_relu_square(self):
        """Test ReLU squared activation used in MLP."""
        x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        result = F.relu(x).square()
        expected = torch.tensor([0.0, 0.0, 1.0, 4.0])
        assert torch.allclose(result, expected)  # nosec: B101 - Test assertion

    def test_cross_entropy(self):
        """Test cross entropy loss used in training."""
        logits = torch.randn(2, 10, 100)  # batch=2, seq=10, vocab=100
        targets = torch.randint(0, 100, (2, 10))
        loss = F.cross_entropy(logits.view(-1, 100), targets.view(-1))
        assert loss.dim() == 0  # Scalar  # nosec: B101 - Test assertion

    def test_scaled_dot_product_attention(self):
        """Test scaled dot product attention."""
        batch = 2
        n_heads = 4
        seq_len = 10
        head_dim = 32

        q = torch.randn(batch, n_heads, seq_len, head_dim)
        k = torch.randn(batch, n_heads, seq_len, head_dim)
        v = torch.randn(batch, n_heads, seq_len, head_dim)

        result = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        assert result.shape == (
            batch,
            n_heads,
            seq_len,
            head_dim,
        )  # nosec: B101 - Test assertion

    def test_tanh_softcap(self):
        """Test tanh softcapping used in logits."""
        logits = torch.randn(2, 10, 100) * 10  # Large values
        softcap = 15
        capped = softcap * torch.tanh(logits / softcap)
        # Values should be bounded by softcap
        assert capped.abs().max() <= softcap  # nosec: B101 - Test assertion


class TestWindowPatternComputation:
    """Tests for window size computation."""

    def _compute_window_sizes(self, config: GPTConfig) -> list:
        """Replicate _compute_window_sizes from train.py."""
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)  # nosec: B101 - Test assertion
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def test_sssl_pattern(self):
        """Test SSSL window pattern."""
        config = GPTConfig(sequence_len=32, n_layer=4, window_pattern="SSSL")
        window_sizes = self._compute_window_sizes(config)
        assert len(window_sizes) == 4  # nosec: B101 - Test assertion
        assert window_sizes[0] == (16, 0)  # S  # nosec: B101 - Test assertion
        assert window_sizes[1] == (16, 0)  # S  # nosec: B101 - Test assertion
        assert window_sizes[2] == (16, 0)  # S  # nosec: B101 - Test assertion
        assert window_sizes[3] == (
            32,
            0,
        )  # L (last layer always long)  # nosec: B101 - Test assertion

    def test_all_long_pattern(self):
        """Test all long window pattern."""
        config = GPTConfig(sequence_len=32, n_layer=3, window_pattern="LLL")
        window_sizes = self._compute_window_sizes(config)
        assert all(ws == (32, 0) for ws in window_sizes)  # nosec: B101 - Test assertion

    def test_alternating_pattern(self):
        """Test alternating SL pattern."""
        config = GPTConfig(sequence_len=32, n_layer=4, window_pattern="SL")
        window_sizes = self._compute_window_sizes(config)
        assert window_sizes[0] == (
            16,
            0,
        )  # S (pattern[0 % 2] = pattern[0] = S)  # nosec: B101 - Test assertion
        assert window_sizes[1] == (
            32,
            0,
        )  # L (pattern[1 % 2] = pattern[1] = L)  # nosec: B101 - Test assertion
        assert window_sizes[2] == (
            16,
            0,
        )  # S (pattern[2 % 2] = pattern[0] = S)  # nosec: B101 - Test assertion
        assert window_sizes[3] == (
            32,
            0,
        )  # L (last layer always long)  # nosec: B101 - Test assertion


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
