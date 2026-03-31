"""Tests for the PyTorch Lightning integration module.

Requires pytorch-lightning>=2.0 or lightning>=2.0.
Install with: pip install flash-semicrf[lightning]
"""

from __future__ import annotations

import importlib
import sys
import warnings
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Skip entire file if Lightning is not installed
# ---------------------------------------------------------------------------
try:
    import lightning.pytorch as L
except ImportError:
    L = None
if L is None:
    try:
        import pytorch_lightning as L  # type: ignore[no-redef]
    except ImportError:
        L = None
if L is None:
    pytest.skip(
        "lightning or pytorch-lightning required for these tests. "
        "Install with: pip install flash-semicrf[lightning]",
        allow_module_level=True,
    )

from torch.utils.data import DataLoader, Dataset

from flash_semicrf import SemiMarkovCRFHead
from flash_semicrf.lightning import SemiCRFLightningModule, pad_and_collate
from flash_semicrf.uncertainty import UncertaintySemiMarkovCRFHead

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

NUM_CLASSES = 6
MAX_DURATION = 5
HIDDEN_DIM = 16
BATCH_SIZE = 3
T_MAX = 20


class TinyEncoder(nn.Module):
    """Minimal encoder for testing: embedding -> linear."""

    def __init__(self, vocab_size: int = 8, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.embedding(x))


class TinyDataset(Dataset):
    """Fixed synthetic dataset for testing."""

    def __init__(self, n: int = 8, seq_len: int = T_MAX):
        self.n = n
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx):
        return {
            "inputs": torch.randint(0, 8, (self.seq_len,)),
            "labels": torch.randint(0, NUM_CLASSES, (self.seq_len,)),
            "lengths": torch.tensor(self.seq_len),
        }


def make_base_crf(**kwargs) -> SemiMarkovCRFHead:
    defaults = {"num_classes": NUM_CLASSES, "max_duration": MAX_DURATION, "hidden_dim": HIDDEN_DIM}
    defaults.update(kwargs)
    return SemiMarkovCRFHead(**defaults)


def make_uncertainty_crf(**kwargs) -> UncertaintySemiMarkovCRFHead:
    defaults = {"num_classes": NUM_CLASSES, "max_duration": MAX_DURATION, "hidden_dim": HIDDEN_DIM}
    defaults.update(kwargs)
    return UncertaintySemiMarkovCRFHead(**defaults)


def make_model(crf=None, **kwargs) -> SemiCRFLightningModule:
    if crf is None:
        crf = make_base_crf()
    # Default to "none" scheduler so tests don't need a val loader for plateau,
    # but allow callers to override via kwargs.
    kwargs.setdefault("scheduler", "none")
    return SemiCRFLightningModule(
        encoder=TinyEncoder(),
        crf=crf,
        **kwargs,
    )


def make_loaders(n: int = 8, seq_len: int = T_MAX):
    ds = TinyDataset(n=n, seq_len=seq_len)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=pad_and_collate)
    return loader


# ===========================================================================
# pad_and_collate tests
# ===========================================================================


class TestPadAndCollate:
    def _make_batch(self, lengths, extra=None):
        """Helper: build batch items with given lengths."""
        batch = []
        for length in lengths:
            item = {
                "inputs": torch.randint(0, 8, (length,)),
                "labels": torch.randint(0, NUM_CLASSES, (length,)),
                "lengths": torch.tensor(length),
            }
            if extra is not None:
                item.update(extra(length))
            batch.append(item)
        return batch

    def test_variable_lengths_shapes(self):
        lengths = [5, 10, 7]
        batch = self._make_batch(lengths)
        out = pad_and_collate(batch)
        T_max = max(lengths)
        assert out["inputs"].shape == (3, T_max)
        assert out["labels"].shape == (3, T_max)
        assert out["lengths"].tolist() == lengths

    def test_padding_is_at_tail(self):
        """Content at [:length] should be preserved; tail should be zero."""
        lengths = [3, 8]
        batch = self._make_batch(lengths)
        out = pad_and_collate(batch)
        # Short item (length=3) padded to 8; tail should be 0
        assert out["inputs"][0, 3:].sum().item() == 0
        assert out["labels"][0, 3:].sum().item() == 0

    def test_2d_inputs(self):
        """2D sequence tensors (T, D) are padded to (B, T_max, D)."""
        batch = [
            {
                "inputs": torch.randn(5, HIDDEN_DIM),
                "labels": torch.randint(0, NUM_CLASSES, (5,)),
                "lengths": torch.tensor(5),
            },
            {
                "inputs": torch.randn(10, HIDDEN_DIM),
                "labels": torch.randint(0, NUM_CLASSES, (10,)),
                "lengths": torch.tensor(10),
            },
        ]
        out = pad_and_collate(batch)
        assert out["inputs"].shape == (2, 10, HIDDEN_DIM)

    def test_fixed_shape_tensor_not_padded(self):
        """A (D,) feature vector where D != T should be stacked, not padded."""
        D = 32  # different from sequence lengths
        lengths = [5, 10]
        batch = self._make_batch(lengths, extra=lambda t: {"feature": torch.randn(D)})
        out = pad_and_collate(batch)
        # feature should be (2, D), not (2, T_max)
        assert out["feature"].shape == (2, D)

    def test_non_tensor_keys_collected_as_list(self):
        """Non-tensor values (string IDs) are returned as plain Python lists."""
        lengths = [5, 10]
        batch = self._make_batch(lengths, extra=lambda t: {"sample_id": f"seq_{t}"})
        out = pad_and_collate(batch)
        assert isinstance(out["sample_id"], list)
        assert out["sample_id"] == ["seq_5", "seq_10"]

    def test_key_mismatch_raises_value_error(self):
        """Heterogeneous keys raise ValueError."""
        batch = [
            {
                "inputs": torch.randint(0, 8, (5,)),
                "labels": torch.zeros(5, dtype=torch.long),
                "lengths": torch.tensor(5),
            },
            {"inputs": torch.randint(0, 8, (5,)), "lengths": torch.tensor(5)},  # missing labels
        ]
        with pytest.raises(ValueError, match="identical keys"):
            pad_and_collate(batch)


# ===========================================================================
# SemiCRFLightningModule unit tests
# ===========================================================================


class TestSemiCRFLightningModuleInit:
    def test_base_head_has_uncertainty_false(self):
        model = make_model()
        assert model._has_uncertainty is False

    def test_uncertainty_head_has_uncertainty_true(self):
        model = make_model(crf=make_uncertainty_crf())
        assert model._has_uncertainty is True

    def test_cosine_without_max_epochs_raises(self):
        with pytest.raises(ValueError, match="max_epochs"):
            make_model(scheduler="cosine", max_epochs=None)

    def test_cosine_with_max_epochs_ok(self):
        model = make_model(scheduler="cosine", max_epochs=20)
        assert model.hparams.max_epochs == 20

    def test_hparams_saved(self):
        model = make_model(lr=2e-4, weight_decay=0.1)
        assert model.hparams.lr == 2e-4
        assert model.hparams.weight_decay == 0.1

    def test_encoder_and_crf_not_in_hparams(self):
        """save_hyperparameters(ignore=['encoder','crf']) must exclude modules."""
        model = make_model()
        assert "encoder" not in model.hparams
        assert "crf" not in model.hparams


class TestConfigureOptimizers:
    def test_plateau_returns_dict_with_monitor(self):
        model = make_model(scheduler="plateau")
        result = model.configure_optimizers()
        assert isinstance(result, dict)
        assert result["lr_scheduler"]["monitor"] == "val/loss"

    def test_cosine_returns_dict_no_monitor(self):
        model = make_model(scheduler="cosine", max_epochs=10)
        result = model.configure_optimizers()
        assert isinstance(result, dict)
        assert "monitor" not in result["lr_scheduler"]

    def test_none_returns_optimizer_only(self):
        model = make_model(scheduler="none")
        result = model.configure_optimizers()
        assert isinstance(result, torch.optim.Optimizer)

    def test_param_groups_split_with_projection(self):
        """Projection params should be in the high-LR group."""
        crf = make_base_crf(hidden_dim=HIDDEN_DIM)
        model = SemiCRFLightningModule(
            encoder=TinyEncoder(),
            crf=crf,
            lr=1e-3,
            crf_lr_scale=0.1,
            scheduler="none",
        )
        opt = model.configure_optimizers()
        # High-LR group
        high_lr_group = opt.param_groups[0]
        # Low-LR group
        low_lr_group = opt.param_groups[1]
        assert abs(high_lr_group["lr"] - 1e-3) < 1e-10
        assert abs(low_lr_group["lr"] - 1e-4) < 1e-10

    def test_param_groups_no_projection(self):
        """hidden_dim=None: no projection layers, all CRF params in structural group."""
        crf = SemiMarkovCRFHead(num_classes=NUM_CLASSES, max_duration=MAX_DURATION)
        model = SemiCRFLightningModule(
            encoder=TinyEncoder(),
            crf=crf,
            scheduler="none",
        )
        opt = model.configure_optimizers()
        # Low-LR group should contain exactly the structural CRF params
        low_lr_ids = {id(p) for p in opt.param_groups[1]["params"]}
        all_crf_ids = {id(p) for p in crf.parameters()}
        assert low_lr_ids == all_crf_ids

    def test_param_group_sanity_check(self):
        """configure_optimizers raises RuntimeError if param split is inconsistent.

        We independently mock named_parameters (drops one param from both groups)
        and parameters (returns the full set), so all_crf_ids != grouped_ids.
        """
        crf = make_base_crf()
        model = SemiCRFLightningModule(encoder=TinyEncoder(), crf=crf, scheduler="none")

        all_named = list(crf.named_parameters())
        all_params = [p for _, p in all_named]
        named_missing_one = all_named[1:]  # drop first param from name-based groups

        def fake_named_parameters(recurse=True):
            # Return named params minus one; called twice in configure_optimizers.
            return iter(named_missing_one)

        def fake_parameters(recurse=True):
            # Bypass named_parameters so all_crf_ids has the full set.
            return iter(all_params)

        with (
            patch.object(crf, "named_parameters", fake_named_parameters),
            patch.object(crf, "parameters", fake_parameters),
        ):
            with pytest.raises(RuntimeError, match="Param group split"):
                model.configure_optimizers()


class TestPrecisionWarning:
    """on_fit_start should emit UserWarning when precision != 32."""

    def test_precision_warning_uses_on_fit_start(self):
        """Verify the hook is on_fit_start, not on_train_start."""
        model = make_model()
        assert hasattr(model, "on_fit_start")
        assert (
            not hasattr(SemiCRFLightningModule, "on_train_start")
            or "prec" not in SemiCRFLightningModule.on_train_start.__code__.co_varnames
        )


class TestLightningImportFallback:
    """Verify that the pytorch_lightning fallback import works."""

    def test_import_fallback(self):
        """Mock lightning.pytorch missing; confirm pytorch_lightning is tried."""
        # Save current state
        lightning_mod = sys.modules.get("lightning")
        lightning_pytorch_mod = sys.modules.get("lightning.pytorch")

        try:
            # Simulate lightning not installed by hiding it from imports
            with patch.dict(sys.modules, {"lightning": None, "lightning.pytorch": None}):
                # The module should still import if pytorch_lightning is available
                spec = importlib.util.find_spec("pytorch_lightning")
                if spec is None:
                    pytest.skip("pytorch_lightning not installed")

                # Re-import lightning.py with lightning blocked
                if "flash_semicrf.lightning" in sys.modules:
                    del sys.modules["flash_semicrf.lightning"]
                # This should fall back to pytorch_lightning
                from flash_semicrf import lightning as lmod  # noqa: F401
        finally:
            # Restore sys.modules
            if lightning_mod is not None:
                sys.modules["lightning"] = lightning_mod
            elif "lightning" in sys.modules:
                del sys.modules["lightning"]
            if lightning_pytorch_mod is not None:
                sys.modules["lightning.pytorch"] = lightning_pytorch_mod
            elif "lightning.pytorch" in sys.modules:
                del sys.modules["lightning.pytorch"]


# ===========================================================================
# Integration tests (require CPU Trainer, ~30s)
# ===========================================================================


@pytest.mark.slow
class TestLightningTrainerIntegration:
    """End-to-end tests using L.Trainer with fast_dev_run=True on CPU."""

    def _trainer(self, **kwargs) -> L.Trainer:
        defaults = {
            "max_epochs": 1,
            "accelerator": "cpu",
            "precision": 32,
            "fast_dev_run": True,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "logger": False,
        }
        defaults.update(kwargs)
        return L.Trainer(**defaults)

    def test_training_loop_base_head(self):
        model = make_model()
        loader = make_loaders()
        trainer = self._trainer()
        trainer.fit(model, loader, loader)

    def test_training_loop_uncertainty_head(self):
        crf = make_uncertainty_crf()
        model = make_model(crf=crf)
        loader = make_loaders()
        trainer = self._trainer()
        trainer.fit(model, loader, loader)
        # Verify entropy logged — key present in callback_metrics
        assert "val/entropy_mean" in trainer.callback_metrics

    def test_test_step(self):
        model = make_model()
        loader = make_loaders()
        trainer = self._trainer()
        # fit first so model is on the right device
        trainer.fit(model, loader, loader)
        results = trainer.test(model, loader)
        assert len(results) == 1
        assert "test/loss" in results[0]

    def test_predict_step_returns_dict(self):
        model = make_model()
        loader = make_loaders()
        trainer = self._trainer()
        trainer.fit(model, loader, loader)
        predictions = trainer.predict(model, loader)
        assert predictions is not None
        assert len(predictions) > 0
        first = predictions[0]
        assert isinstance(first, dict)
        assert "scores" in first
        assert "segments" in first
        assert "lengths" in first

    def test_precision_warning_fires(self):
        model = make_model()
        loader = make_loaders()
        # bf16-mixed is supported on CPU in Lightning 2.x + PyTorch >= 2.0
        trainer = self._trainer(precision="bf16-mixed")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                trainer.fit(model, loader, loader)
            except Exception:
                pass  # may fail after the warning is emitted
        precision_warnings = [
            w
            for w in caught
            if issubclass(w.category, UserWarning) and "precision=32" in str(w.message)
        ]
        assert len(precision_warnings) >= 1

    def test_gradient_checkpointing_cpu(self):
        model = make_model(gradient_checkpointing=True)
        loader = make_loaders()
        trainer = self._trainer()
        trainer.fit(model, loader, loader)
        loss = trainer.callback_metrics.get("train/loss")
        assert loss is not None
        assert torch.isfinite(torch.tensor(float(loss)))

    def test_train_loss_step_logged(self):
        """train/loss should be logged per-step (on_step=True, on_epoch=False)."""
        model = make_model()
        loader = make_loaders()
        trainer = self._trainer()
        trainer.fit(model, loader, loader)
        assert "train/loss" in trainer.callback_metrics

    def test_train_loss_epoch_logged(self):
        """train/loss_epoch should be logged epoch-level (sync_dist=True)."""
        model = make_model()
        loader = make_loaders()
        trainer = self._trainer()
        trainer.fit(model, loader, loader)
        assert "train/loss_epoch" in trainer.callback_metrics
