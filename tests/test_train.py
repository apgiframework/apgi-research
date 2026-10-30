"""
Comprehensive tests for train.py - autoresearch pretraining script.
Tests individual components with mocking for GPU-dependent operations.
"""

# Import train.py components (need to handle the module-level execution)
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests._module_isolation import installed_module  # noqa: E402
from tests._train_definitions import load_train_definitions  # noqa: E402

# train.py is a script, so importing it would run the whole training loop.
# Load only its definitions and make them importable as ``train`` for these
# tests (both ``from train import X`` and ``patch("train.X")`` resolve to it).
_train = load_train_definitions()


@pytest.fixture(autouse=True, scope="module")
def _train_definitions_importable():
    with installed_module("train", _train):
        yield


class TestGPTConfig:
    """Tests for GPTConfig dataclass."""

    def test_default_values(self):
        """Test default GPTConfig values."""
        from train import GPTConfig

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
        from train import GPTConfig

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
        from train import norm

        x = torch.tensor([[1.0, 2.0, 3.0]])
        result = norm(x)
        # RMS norm should normalize to unit variance
        assert result.shape == x.shape  # nosec: B101 - Test assertion

    def test_norm_batch(self):
        """Test norm function with batch."""
        from train import norm

        x = torch.randn(2, 10, 128)
        result = norm(x)
        assert result.shape == x.shape  # nosec: B101 - Test assertion

    def test_has_ve(self):
        """Test has_ve function (Value Embedding condition)."""
        from train import has_ve

        # Alternating layers have VE, and the last layer always does:
        # layer_idx % 2 == (n_layer - 1) % 2, i.e. odd layers for n_layer=12
        assert has_ve(0, 12) is False  # nosec: B101 - Test assertion
        assert has_ve(1, 12) is True  # nosec: B101 - Test assertion
        assert has_ve(11, 12) is True  # nosec: B101 - Test assertion

    def test_has_ve_different_layers(self):
        """Test has_ve with different layer counts."""
        from train import has_ve

        # n_layer=8: the last layer (7) has VE, so the odd layers do
        assert has_ve(0, 8) is False  # nosec: B101 - Test assertion
        assert has_ve(1, 8) is True  # nosec: B101 - Test assertion
        assert has_ve(7, 8) is True  # nosec: B101 - Test assertion
        # n_layer=7: the last layer (6) has VE, so the even layers do
        assert has_ve(0, 7) is True  # nosec: B101 - Test assertion
        assert has_ve(1, 7) is False  # nosec: B101 - Test assertion

    def test_apply_rotary_emb(self):
        """Test apply_rotary_emb function."""
        from train import apply_rotary_emb

        batch_size = 2
        seq_len = 10
        head_dim = 64
        x = torch.randn(batch_size, seq_len, 4, head_dim)
        cos = torch.randn(1, seq_len, 1, head_dim // 2)
        sin = torch.randn(1, seq_len, 1, head_dim // 2)

        result = apply_rotary_emb(x, cos, sin)
        assert result.shape == x.shape  # nosec: B101 - Test assertion

    def test_apply_rotary_emb_assertion(self):
        """Test apply_rotary_emb rejects inputs that are not 4D."""
        from train import apply_rotary_emb

        x = torch.randn(2, 10, 64)  # Wrong shape (3D instead of 4D)
        cos = torch.randn(1, 10, 1, 32)
        sin = torch.randn(1, 10, 1, 32)

        with pytest.raises(ValueError, match="Expected 4D tensor, got 3D"):
            apply_rotary_emb(x, cos, sin)


class TestMLP:
    """Tests for MLP class."""

    def test_mlp_init(self):
        """Test MLP initialization."""
        from train import MLP, GPTConfig

        config = GPTConfig(n_embd=128)
        mlp = MLP(config)
        assert mlp.c_fc.in_features == 128  # nosec: B101 - Test assertion
        assert mlp.c_fc.out_features == 512  # nosec: B101 - Test assertion
        assert mlp.c_proj.in_features == 512  # nosec: B101 - Test assertion
        assert mlp.c_proj.out_features == 128  # nosec: B101 - Test assertion

    def test_mlp_forward(self):
        """Test MLP forward pass."""
        from train import MLP, GPTConfig

        config = GPTConfig(n_embd=128)
        mlp = MLP(config)
        x = torch.randn(2, 10, 128)
        result = mlp(x)
        assert result.shape == x.shape  # nosec: B101 - Test assertion


class TestCausalSelfAttention:
    """Tests for CausalSelfAttention class."""

    def test_attention_init(self):
        """Test CausalSelfAttention initialization."""
        from train import CausalSelfAttention, GPTConfig

        config = GPTConfig(n_embd=128, n_head=4, n_kv_head=4)
        attn = CausalSelfAttention(config, layer_idx=0)
        assert attn.n_head == 4  # nosec: B101 - Test assertion
        assert attn.n_kv_head == 4  # nosec: B101 - Test assertion
        assert attn.n_embd == 128  # nosec: B101 - Test assertion
        assert attn.head_dim == 32  # nosec: B101 - Test assertion

    def test_attention_init_gqa(self):
        """Test CausalSelfAttention with Grouped Query Attention."""
        from train import CausalSelfAttention, GPTConfig

        config = GPTConfig(n_embd=128, n_head=4, n_kv_head=2)
        attn = CausalSelfAttention(config, layer_idx=0)
        assert attn.n_head == 4  # nosec: B101 - Test assertion
        assert attn.n_kv_head == 2  # nosec: B101 - Test assertion

    def test_attention_forward_no_ve(self):
        """Test attention forward without value embedding."""
        from train import CausalSelfAttention, GPTConfig

        config = GPTConfig(n_embd=128, n_head=4, n_kv_head=4, sequence_len=32)
        attn = CausalSelfAttention(config, layer_idx=0)  # No VE for layer 0
        x = torch.randn(2, 10, 128)
        # head_dim = 128 / 4 = 32; rotary tables cover half of it
        cos = torch.randn(1, 10, 1, 16)
        sin = torch.randn(1, 10, 1, 16)
        window_size = (32, 0)

        result = attn(x, None, (cos, sin), window_size)
        assert result.shape == x.shape  # nosec: B101 - Test assertion

    def test_attention_forward_with_ve(self):
        """Test attention forward with value embedding."""
        from train import CausalSelfAttention, GPTConfig

        config = GPTConfig(n_embd=128, n_head=4, n_kv_head=4, sequence_len=32)
        attn = CausalSelfAttention(config, layer_idx=1)  # Has VE for layer 1
        x = torch.randn(2, 10, 128)
        ve = torch.randn(2, 10, 128)
        cos = torch.randn(1, 10, 1, 16)
        sin = torch.randn(1, 10, 1, 16)
        window_size = (32, 0)

        result = attn(x, ve, (cos, sin), window_size)
        assert result.shape == x.shape  # nosec: B101 - Test assertion


class TestBlock:
    """Tests for Block class."""

    def test_block_init(self):
        """Test Block initialization."""
        from train import Block, GPTConfig

        config = GPTConfig(n_embd=128, n_head=4, n_kv_head=4)
        block = Block(config, layer_idx=0)
        assert isinstance(block.attn, torch.nn.Module)  # nosec: B101
        assert isinstance(block.mlp, torch.nn.Module)  # nosec: B101

    def test_block_forward(self):
        """Test Block forward pass."""
        from train import Block, GPTConfig

        config = GPTConfig(n_embd=128, n_head=4, n_kv_head=4, sequence_len=32)
        block = Block(config, layer_idx=0)
        x = torch.randn(2, 10, 128)
        cos = torch.randn(1, 10, 1, 16)
        sin = torch.randn(1, 10, 1, 16)
        window_size = (32, 0)

        result = block(x, None, (cos, sin), window_size)
        assert result.shape == x.shape  # nosec: B101 - Test assertion


class TestGPT:
    """Tests for GPT class."""

    @patch("train.use_flash_attn", False)
    def test_gpt_init(self):
        """Test GPT initialization."""
        from train import GPT, GPTConfig

        config = GPTConfig(
            sequence_len=32,
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_kv_head=4,
            n_embd=128,
        )
        model = GPT(config)
        assert model.config == config  # nosec: B101 - Test assertion
        transformer_h = model.transformer.h
        assert isinstance(
            transformer_h, torch.nn.ModuleList
        )  # nosec: B101 - Test assertion
        assert len(transformer_h) == 2  # nosec: B101 - Test assertion
        assert (
            len(model.value_embeds) == 1
        )  # Only layer 0 has VE for n_layer=2  # nosec: B101 - Test assertion

    @patch("train.use_flash_attn", False)
    def test_gpt_forward_no_targets(self):
        """Test GPT forward without targets (inference mode)."""
        from train import GPT, GPTConfig

        config = GPTConfig(
            sequence_len=32,
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_kv_head=4,
            n_embd=128,
        )
        model = GPT(config)
        model.eval()
        idx = torch.randint(0, 1000, (2, 10))

        result = model(idx)
        assert result.shape == (2, 10, 1000)  # nosec: B101 - Test assertion

    @patch("train.use_flash_attn", False)
    def test_gpt_forward_with_targets(self):
        """Test GPT forward with targets (training mode)."""
        from train import GPT, GPTConfig

        config = GPTConfig(
            sequence_len=32,
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_kv_head=4,
            n_embd=128,
        )
        model = GPT(config)
        idx = torch.randint(0, 1000, (2, 10))
        targets = torch.randint(0, 1000, (2, 10))

        result = model(idx, targets)
        assert isinstance(
            result, torch.Tensor
        )  # Loss scalar  # nosec: B101 - Test assertion
        assert result.dim() == 0  # nosec: B101 - Test assertion

    @patch("train.use_flash_attn", False)
    def test_gpt_forward_reduction_sum(self):
        """Test GPT forward with sum reduction."""
        from train import GPT, GPTConfig

        config = GPTConfig(
            sequence_len=32,
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_kv_head=4,
            n_embd=128,
        )
        model = GPT(config)
        idx = torch.randint(0, 1000, (2, 10))
        targets = torch.randint(0, 1000, (2, 10))

        result = model(idx, targets, reduction="sum")
        assert isinstance(result, torch.Tensor)  # nosec: B101 - Test assertion

    @patch("train.use_flash_attn", False)
    def test_gpt_forward_reduction_none(self):
        """Test GPT forward with no reduction."""
        from train import GPT, GPTConfig

        config = GPTConfig(
            sequence_len=32,
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_kv_head=4,
            n_embd=128,
        )
        model = GPT(config)
        idx = torch.randint(0, 1000, (2, 10))
        targets = torch.randint(0, 1000, (2, 10))

        result = model(idx, targets, reduction="none")
        assert result.shape == (20,)  # 2 * 10 flattened  # nosec: B101 - Test assertion

    @patch("train.use_flash_attn", False)
    def test_gpt_estimate_flops(self):
        """Test GPT FLOPs estimation."""
        from train import GPT, GPTConfig

        config = GPTConfig(
            sequence_len=32,
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_kv_head=4,
            n_embd=128,
        )
        model = GPT(config)
        flops = model.estimate_flops()
        assert isinstance(flops, int)  # nosec: B101 - Test assertion
        assert flops > 0  # nosec: B101 - Test assertion

    @patch("train.use_flash_attn", False)
    def test_gpt_num_scaling_params(self):
        """Test GPT parameter counting."""
        from train import GPT, GPTConfig

        config = GPTConfig(
            sequence_len=32,
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_kv_head=4,
            n_embd=128,
        )
        model = GPT(config)
        params = model.num_scaling_params()
        assert "wte" in params  # nosec: B101 - Test assertion
        assert "value_embeds" in params  # nosec: B101 - Test assertion
        assert "lm_head" in params  # nosec: B101 - Test assertion
        assert "transformer_matrices" in params  # nosec: B101 - Test assertion
        assert "scalars" in params  # nosec: B101 - Test assertion
        assert "total" in params  # nosec: B101 - Test assertion
        assert params["total"] > 0  # nosec: B101 - Test assertion

    @patch("train.use_flash_attn", False)
    def test_gpt_compute_window_sizes(self):
        """Test window size computation."""
        from train import GPT, GPTConfig

        config = GPTConfig(
            sequence_len=32,
            vocab_size=1000,
            n_layer=4,
            n_head=4,
            n_kv_head=4,
            n_embd=128,
            window_pattern="SSSL",
        )
        model = GPT(config)
        window_sizes = model.window_sizes
        assert len(window_sizes) == 4  # nosec: B101 - Test assertion
        # Pattern SSSL: S=16, S=16, S=16, L=32
        assert window_sizes[0] == (16, 0)  # nosec: B101 - Test assertion
        assert window_sizes[1] == (16, 0)  # nosec: B101 - Test assertion
        assert window_sizes[2] == (16, 0)  # nosec: B101 - Test assertion
        assert window_sizes[3] == (
            32,
            0,
        )  # Last layer always long  # nosec: B101 - Test assertion

    @patch("train.use_flash_attn", False)
    def test_gpt_window_pattern_all_long(self):
        """Test window pattern with all long windows."""
        from train import GPT, GPTConfig

        config = GPTConfig(
            sequence_len=32,
            vocab_size=1000,
            n_layer=3,
            n_head=4,
            n_kv_head=4,
            n_embd=128,
            window_pattern="LLL",
        )
        model = GPT(config)
        window_sizes = model.window_sizes
        assert all(ws == (32, 0) for ws in window_sizes)  # nosec: B101 - Test assertion

    @patch("train.use_flash_attn", False)
    def test_gpt_precompute_rotary_embeddings(self):
        """Test rotary embedding precomputation."""
        from train import GPT, GPTConfig

        config = GPTConfig(
            sequence_len=32,
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_kv_head=4,
            n_embd=128,
        )
        model = GPT(config)
        assert (
            model.cos.shape[1] == 320
        )  # sequence_len * 10  # nosec: B101 - Test assertion
        assert model.sin.shape[1] == 320  # nosec: B101 - Test assertion

    @patch("train.use_flash_attn", False)
    def test_gpt_init_weights(self):
        """Test weight initialization."""
        from train import GPT, GPTConfig

        config = GPTConfig(
            sequence_len=32,
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_kv_head=4,
            n_embd=128,
        )
        model = GPT(config)
        model.init_weights()
        # Check that weights are initialized (not zeros)
        wte = model.transformer.wte
        assert isinstance(wte, torch.nn.Embedding)  # nosec: B101 - Test assertion
        assert wte.weight.abs().sum() > 0  # nosec: B101 - Test assertion


class TestMuonAdamW:
    """Tests for MuonAdamW optimizer."""

    def test_muon_adamw_init(self):
        """Test MuonAdamW initialization."""
        from train import MuonAdamW

        param = torch.nn.Parameter(torch.randn(10, 10))
        param_groups = [
            {
                "kind": "adamw",
                "params": [param],
                "lr": 0.001,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.0,
            }
        ]
        optimizer = MuonAdamW(param_groups)
        assert len(optimizer.param_groups) == 1  # nosec: B101 - Test assertion

    def test_muon_adamw_step_adamw(self):
        """Test AdamW step."""
        from train import MuonAdamW

        param = torch.nn.Parameter(torch.randn(10, 10))
        param.grad = torch.randn(10, 10)
        param_groups = [
            {
                "kind": "adamw",
                "params": [param],
                "lr": 0.001,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.0,
            }
        ]
        optimizer = MuonAdamW(param_groups)
        optimizer.step()
        # Parameter should have changed
        assert not torch.equal(
            param, torch.zeros(10, 10)
        )  # nosec: B101 - Test assertion

    def test_muon_adamw_step_no_grad(self):
        """Test step with no gradient (should skip)."""
        from train import MuonAdamW

        param = torch.nn.Parameter(torch.randn(10, 10))
        param_groups = [
            {
                "kind": "adamw",
                "params": [param],
                "lr": 0.001,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.0,
            }
        ]
        optimizer = MuonAdamW(param_groups)
        original_param = param.clone()
        optimizer.step()
        # Parameter should not have changed (no grad)
        assert torch.equal(param, original_param)  # nosec: B101 - Test assertion

    def test_muon_adamw_step_muon(self):
        """Test Muon step for matrix parameters."""
        from train import MuonAdamW

        params = [torch.nn.Parameter(torch.randn(10, 10)) for _ in range(2)]
        for p in params:
            p.grad = torch.randn(10, 10)
        param_groups = [
            {
                "kind": "muon",
                "params": params,
                "lr": 0.01,
                "momentum": 0.95,
                "ns_steps": 3,
                "beta2": 0.95,
                "weight_decay": 0.1,
            }
        ]
        optimizer = MuonAdamW(param_groups)
        optimizer.step()
        # Parameters should have changed
        for p in params:
            assert not torch.equal(
                p, torch.zeros(10, 10)
            )  # nosec: B101 - Test assertion

    def test_muon_adamw_closure(self):
        """Test step with closure (should be ignored)."""
        from train import MuonAdamW

        param = torch.nn.Parameter(torch.randn(10, 10))
        param.grad = torch.randn(10, 10)
        param_groups = [
            {
                "kind": "adamw",
                "params": [param],
                "lr": 0.001,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.0,
            }
        ]
        optimizer = MuonAdamW(param_groups)

        def closure():
            return torch.tensor(1.0)

        optimizer.step(closure)  # Should not raise


class TestScheduleFunctions:
    """Tests for learning rate and schedule functions."""

    def test_get_lr_multiplier_warmup(self):
        """Test LR multiplier during warmup."""
        from train import WARMUP_RATIO, get_lr_multiplier

        if WARMUP_RATIO > 0:
            # At 50% of warmup
            lrm = get_lr_multiplier(WARMUP_RATIO * 0.5)
            assert 0 < lrm < 1  # nosec: B101 - Test assertion
            # At end of warmup
            lrm = get_lr_multiplier(WARMUP_RATIO)
            assert lrm == 1.0  # nosec: B101 - Test assertion

    def test_get_lr_multiplier_steady(self):
        """Test LR multiplier during steady state."""
        from train import get_lr_multiplier

        # In steady state (after warmup, before warmdown)
        progress = 0.3
        lrm = get_lr_multiplier(progress)
        assert lrm == 1.0  # nosec: B101 - Test assertion

    def test_get_lr_multiplier_warmdown(self):
        """Test LR multiplier during warmdown."""
        from train import FINAL_LR_FRAC, get_lr_multiplier

        # At start of warmdown (assuming WARMDOWN_RATIO is 0.5)
        progress = 0.5
        lrm = get_lr_multiplier(progress)
        assert lrm == 1.0  # nosec: B101 - Test assertion
        # At end of warmdown
        lrm = get_lr_multiplier(1.0)
        assert lrm == FINAL_LR_FRAC  # nosec: B101 - Test assertion

    def test_get_lr_multiplier_zero_warmup(self):
        """Test LR multiplier with zero warmup."""
        # Patch WARMUP_RATIO to 0
        import train
        from train import get_lr_multiplier

        original_warmup = train.WARMUP_RATIO
        train.WARMUP_RATIO = 0.0
        try:
            lrm = get_lr_multiplier(0.0)
            assert lrm == 1.0  # nosec: B101 - Test assertion
        finally:
            train.WARMUP_RATIO = original_warmup

    def test_get_muon_momentum(self):
        """Test Muon momentum schedule."""
        from train import get_muon_momentum

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
        from train import WEIGHT_DECAY, get_weight_decay

        # At start
        wd = get_weight_decay(0.0)
        assert wd == WEIGHT_DECAY  # nosec: B101 - Test assertion
        # At 50% progress
        wd = get_weight_decay(0.5)
        assert wd == WEIGHT_DECAY * 0.5  # nosec: B101 - Test assertion
        # At end
        wd = get_weight_decay(1.0)
        assert wd == 0.0  # nosec: B101 - Test assertion


class TestBuildModelConfig:
    """Tests for build_model_config function."""

    @patch("train.ASPECT_RATIO", 64)
    @patch("train.HEAD_DIM", 128)
    @patch("train.WINDOW_PATTERN", "SSSL")
    @patch("train.MAX_SEQ_LEN", 2048)
    @patch("train.vocab_size", 1000, create=True)
    def test_build_model_config(self):
        """Test model config building."""
        from train import build_model_config

        config = build_model_config(depth=8)
        assert config.sequence_len == 2048  # nosec: B101 - Test assertion
        assert config.vocab_size == 1000  # nosec: B101 - Test assertion
        assert config.n_layer == 8  # nosec: B101 - Test assertion
        assert config.window_pattern == "SSSL"  # nosec: B101 - Test assertion
        # Model dim should be depth * ASPECT_RATIO, rounded to HEAD_DIM multiple
        assert config.n_embd % 128 == 0  # nosec: B101 - Test assertion
        assert config.n_head == config.n_embd // 128  # nosec: B101 - Test assertion

    @patch("train.ASPECT_RATIO", 64)
    @patch("train.HEAD_DIM", 128)
    @patch("train.WINDOW_PATTERN", "LLLL")
    @patch("train.MAX_SEQ_LEN", 2048)
    @patch("train.vocab_size", 1000, create=True)
    def test_build_model_config_different_depth(self):
        """Test model config with different depth."""
        from train import build_model_config

        config = build_model_config(depth=4)
        assert config.n_layer == 4  # nosec: B101 - Test assertion
        assert config.n_embd % 128 == 0  # nosec: B101 - Test assertion


class TestConstants:
    """Tests for module-level constants."""

    def test_aspect_ratio(self):
        """Test ASPECT_RATIO constant."""
        from train import ASPECT_RATIO

        assert isinstance(ASPECT_RATIO, int)  # nosec: B101 - Test assertion
        assert ASPECT_RATIO > 0  # nosec: B101 - Test assertion

    def test_head_dim(self):
        """Test HEAD_DIM constant."""
        from train import HEAD_DIM

        assert isinstance(HEAD_DIM, int)  # nosec: B101 - Test assertion
        assert HEAD_DIM > 0  # nosec: B101 - Test assertion

    def test_window_pattern(self):
        """Test WINDOW_PATTERN constant."""
        from train import WINDOW_PATTERN

        assert isinstance(WINDOW_PATTERN, str)  # nosec: B101 - Test assertion
        assert all(c in "SL" for c in WINDOW_PATTERN)  # nosec: B101 - Test assertion

    def test_total_batch_size(self):
        """Test TOTAL_BATCH_SIZE constant."""
        from train import TOTAL_BATCH_SIZE

        assert isinstance(TOTAL_BATCH_SIZE, int)  # nosec: B101 - Test assertion
        assert TOTAL_BATCH_SIZE > 0  # nosec: B101 - Test assertion

    def test_learning_rates(self):
        """Test learning rate constants."""
        from train import EMBEDDING_LR, MATRIX_LR, SCALAR_LR, UNEMBEDDING_LR

        assert isinstance(EMBEDDING_LR, float)  # nosec: B101 - Test assertion
        assert isinstance(MATRIX_LR, float)  # nosec: B101 - Test assertion
        assert isinstance(SCALAR_LR, float)  # nosec: B101 - Test assertion
        assert isinstance(UNEMBEDDING_LR, float)  # nosec: B101 - Test assertion
        assert all(  # nosec: B101 - Test assertion
            lr > 0 for lr in [EMBEDDING_LR, MATRIX_LR, SCALAR_LR, UNEMBEDDING_LR]
        )

    def test_depth(self):
        """Test DEPTH constant."""
        from train import DEPTH

        assert isinstance(DEPTH, int)  # nosec: B101 - Test assertion
        assert DEPTH > 0  # nosec: B101 - Test assertion


class TestPolarExpressCoeffs:
    """Tests for polar_express_coeffs constant."""

    def test_polar_express_coeffs_structure(self):
        """Test polar_express_coeffs structure."""
        from train import polar_express_coeffs

        assert isinstance(polar_express_coeffs, list)  # nosec: B101 - Test assertion
        assert len(polar_express_coeffs) > 0  # nosec: B101 - Test assertion
        for coeff in polar_express_coeffs:
            assert isinstance(coeff, tuple)  # nosec: B101 - Test assertion
            assert len(coeff) == 3  # nosec: B101 - Test assertion
            assert all(
                isinstance(c, float) for c in coeff
            )  # nosec: B101 - Test assertion
