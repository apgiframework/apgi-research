"""
Comprehensive tests for prepare.py to achieve 90%+ coverage.
Tests all major functions, data processing, error handling, and edge cases.

Filesystem work happens in pytest temp directories (module constants are
patched), and the network, process pools and sleeps are mocked.
"""

from __future__ import annotations

import itertools
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

import prepare
from prepare import (
    EVAL_TOKENS,
    MAX_SEQ_LEN,
    MAX_SHARD,
    TIME_BUDGET,
    VAL_SHARD,
    Tokenizer,
    download_data,
    download_single_shard,
    make_dataloader,
    read_shard,
)


@pytest.fixture
def data_dir(tmp_path: Path) -> Iterator[Path]:
    """Point prepare.DATA_DIR at an empty temp directory."""
    directory = tmp_path / "data"
    directory.mkdir()
    with patch.object(prepare, "DATA_DIR", directory):
        yield directory


@pytest.fixture
def mock_pool() -> Iterator[MagicMock]:
    """Replace the download process pool; yields the pool instance."""
    with patch.object(prepare, "Pool") as pool_class, patch("builtins.print"):
        pool = pool_class.return_value.__enter__.return_value
        pool.map.side_effect = lambda func, ids: [True] * len(ids)
        pool.pool_class = pool_class
        yield pool


@pytest.fixture
def unpinned_empty() -> Iterator[None]:
    """Make ``torch.empty`` ignore ``pin_memory`` so CPU-only machines work.

    make_dataloader pins its CPU buffer unconditionally, which current torch
    refuses without CUDA. This keeps the packing logic testable everywhere.
    """
    real_empty = torch.empty

    def empty(*args, pin_memory=False, **kwargs):
        return real_empty(*args, **kwargs)

    with patch.object(prepare.torch, "empty", side_effect=empty):
        yield


def _encoder(vocab_size: int = 8192) -> Mock:
    """A fake tiktoken-style encoder."""
    enc = Mock()
    enc.n_vocab = vocab_size
    enc.encode_single_token.return_value = 0
    enc.encode_ordinary.side_effect = lambda text: [ord(c) for c in text]
    enc.encode_ordinary_batch.side_effect = lambda texts, num_threads=8: [
        [ord(c) for c in text] for text in texts
    ]
    enc.decode.side_effect = lambda ids: "".join(chr(i) for i in ids)
    return enc


class TestTokenizer:
    """Tests for Tokenizer class."""

    def test_tokenizer_initialization(self):
        """Test tokenizer initialization."""
        enc = _encoder(vocab_size=1234)
        enc.encode_single_token.return_value = 7

        tokenizer = Tokenizer(enc)

        assert tokenizer.enc is enc  # nosec: B101 - Test assertion
        assert tokenizer.get_vocab_size() == 1234  # nosec: B101
        assert tokenizer.get_bos_token_id() == 7  # nosec: B101
        enc.encode_single_token.assert_called_once_with(prepare.BOS_TOKEN)

    def test_bpe_evaluation(self):
        """Encoded ids are what the byte-level evaluation metric consumes."""
        tokenizer = Tokenizer(_encoder())

        batch = tokenizer.encode(["ab", "c"], prepend=5)

        # A BOS id is prepended to every row of a batch
        assert batch == [[5, 97, 98], [5, 99]]  # nosec: B101 - Test assertion

    def test_encoding_decoding(self):
        """Test encoding and decoding."""
        tokenizer = Tokenizer(_encoder())

        encoded = tokenizer.encode("test string")
        assert isinstance(encoded, list)  # nosec: B101 - Test assertion
        assert tokenizer.decode(encoded) == "test string"  # nosec: B101

    def test_encode_prepend_by_token_name(self):
        """A string ``prepend`` is resolved to its token id first."""
        enc = _encoder()
        enc.encode_single_token.side_effect = lambda token: 9 if token == "<bos>" else 0
        tokenizer = Tokenizer(enc)

        assert tokenizer.encode("a", prepend="<bos>") == [9, 97]  # nosec: B101

    def test_encode_rejects_unsupported_input(self):
        """Only strings and lists of strings can be encoded."""
        with pytest.raises(ValueError, match="Invalid input type"):
            Tokenizer(_encoder()).encode(123)


class TestDataPreparation:
    """Tests for data preparation functions."""

    def test_download_data(self, data_dir, mock_pool):
        """Test data shard downloading."""
        download_data(num_shards=5)

        # Five training shards plus the pinned validation shard
        _, ids = mock_pool.map.call_args.args
        assert ids == [*range(5), VAL_SHARD]  # nosec: B101 - Test assertion

    def test_download_data_error_handling(self, data_dir, mock_pool):
        """Test error handling in data download."""
        mock_pool.map.side_effect = Exception("Download failed")

        with pytest.raises(Exception, match="Download failed"):
            download_data(num_shards=5)

    def test_dataloader_creation(self, unpinned_empty):
        """Test dataloader creation."""
        tokenizer = Mock()
        tokenizer.get_bos_token_id.return_value = 1
        tokenizer.encode.side_effect = lambda batch, prepend=None: [
            [prepend, 2, 3, 4] for _ in batch
        ]
        batches = itertools.cycle([(["a"], 1)])

        with patch.object(prepare, "_document_batches", return_value=batches):
            loader = make_dataloader(tokenizer, 4, 8, "train", buffer_size=4)
            inputs, targets, epoch = next(loader)

        assert inputs.shape == (4, 8)  # nosec: B101 - Test assertion
        assert targets.shape == (4, 8)  # nosec: B101 - Test assertion
        assert epoch == 1  # nosec: B101 - Test assertion

    def test_integration(self):
        """Test complete data preparation workflow."""
        with (
            patch.object(prepare, "download_data") as mock_download,
            patch.object(
                prepare, "list_parquet_files", return_value=[Path("shard.parquet")]
            ),
            patch.object(
                prepare, "train_tokenizer", return_value=True
            ) as mock_tokenizer,
        ):
            exit_code = prepare.main(["--num-shards", "3"])

            # Verify workflow steps
            assert exit_code == 0  # nosec: B101 - Test assertion
            mock_download.assert_called_once_with(3, 8)
            mock_tokenizer.assert_called_once()

    def test_with_custom_params(self):
        """Test data preparation with custom parameters."""
        with (
            patch.object(prepare, "download_data") as mock_download,
            patch.object(
                prepare, "list_parquet_files", return_value=[Path("shard.parquet")]
            ),
            patch.object(prepare, "train_tokenizer", return_value=True) as mock_train,
        ):
            exit_code = prepare.main(
                ["--num-shards", "10", "--download-workers", "4", "--skip-tokenizer"]
            )

        assert exit_code == 0  # nosec: B101 - Test assertion
        mock_download.assert_called_once_with(10, 4)
        mock_train.assert_not_called()  # nosec: B101 - Test assertion


class TestConstants:
    """Tests for module constants."""

    def test_max_seq_len(self):
        """Test MAX_SEQ_LEN constant."""
        assert MAX_SEQ_LEN == 2048  # nosec: B101 - Test assertion

    def test_time_budget(self):
        """Test TIME_BUDGET constant."""
        assert TIME_BUDGET == 600  # 10 minutes  # nosec: B101 - Test assertion

    def test_eval_tokens(self):
        """Test EVAL_TOKENS constant."""
        assert EVAL_TOKENS == 40 * 524288  # nosec: B101 - Test assertion

    def test_constant_values(self):
        """Test that constants have expected values."""
        # Verify constants are reasonable
        assert MAX_SEQ_LEN > 0  # nosec: B101 - Test assertion
        assert TIME_BUDGET > 0  # nosec: B101 - Test assertion
        assert EVAL_TOKENS > 0  # nosec: B101 - Test assertion

        # Verify relationships
        assert (  # nosec: B101 - Test assertion
            EVAL_TOKENS > MAX_SEQ_LEN
        )  # Should evaluate more tokens than sequence length


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_missing_data_directory(self, tmp_path, mock_pool):
        """download_data creates the data directory when it is missing."""
        missing = tmp_path / "missing" / "data"

        with patch.object(prepare, "DATA_DIR", missing):
            download_data(num_shards=1)

        assert missing.is_dir()  # nosec: B101 - Test assertion

    def test_insufficient_disk_space(self, tmp_path):
        """A full disk surfaces as an OSError rather than being swallowed."""
        with (
            patch.object(prepare, "DATA_DIR", tmp_path / "data"),
            patch.object(
                prepare.os, "makedirs", side_effect=OSError("No space left on device")
            ),
        ):
            with pytest.raises(OSError, match="No space left"):
                download_data(num_shards=1)

    def test_network_timeout(self, data_dir):
        """Test handling of network timeouts."""
        with (
            patch.object(
                prepare.requests, "get", side_effect=Exception("Network timeout")
            ),
            patch.object(prepare.time, "sleep"),
            patch("builtins.print"),
        ):
            assert download_single_shard(1) is False  # nosec: B101

    def test_corrupted_data_recovery(self, tmp_path):
        """A shard that cannot be parsed reads as missing, not as a crash."""
        (tmp_path / "shard_00001.parquet").write_bytes(b"not parquet")

        with patch.object(prepare.pq, "read_table", side_effect=Exception("corrupt")):
            assert read_shard(1, tmp_path) is None  # nosec: B101

        # A shard that does not exist at all also reads as missing
        assert read_shard(2, tmp_path) is None  # nosec: B101 - Test assertion


class TestParameterValidation:
    """Tests for parameter handling."""

    def test_num_shards_validation(self, data_dir, mock_pool):
        """Shard counts are clamped; the validation shard is always included."""
        download_data(num_shards=MAX_SHARD + 100)

        _, ids = mock_pool.map.call_args.args
        assert ids == [*range(MAX_SHARD), VAL_SHARD]  # nosec: B101

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_batch_size_validation(self, batch_size, unpinned_empty):
        """The dataloader honours the requested batch size."""
        inputs, targets, _ = self._first_batch(B=batch_size, T=8)

        assert inputs.shape == (batch_size, 8)  # nosec: B101 - Test assertion
        assert targets.shape == (batch_size, 8)  # nosec: B101 - Test assertion

    @pytest.mark.parametrize("seq_len", [4, 8, 16])
    def test_seq_len_validation(self, seq_len, unpinned_empty):
        """The dataloader honours the requested sequence length."""
        inputs, targets, _ = self._first_batch(B=2, T=seq_len)

        assert inputs.shape == (2, seq_len)  # nosec: B101 - Test assertion
        assert targets.shape == (2, seq_len)  # nosec: B101 - Test assertion

    def test_split_validation(self):
        """Only the train and val splits are accepted."""
        with pytest.raises(AssertionError):
            next(make_dataloader(Mock(), B=2, T=8, split="test"))

    def test_data_dir_validation(self):
        """The command line rejects a non-integer shard count."""
        with pytest.raises(SystemExit):
            prepare.parse_args(["--num-shards", "many"])

    @staticmethod
    def _first_batch(B: int, T: int):
        tokenizer = Mock()
        tokenizer.get_bos_token_id.return_value = 1
        tokenizer.encode.side_effect = lambda batch, prepend=None: [
            [prepend, 2, 3, 4, 5] for _ in batch
        ]
        batches = itertools.cycle([(["a", "b"], 1)])
        with patch.object(prepare, "_document_batches", return_value=batches):
            return next(
                make_dataloader(tokenizer, B=B, T=T, split="train", buffer_size=4)
            )


class TestIntegration:
    """Integration tests for prepare.py."""

    def test_full_preparation_workflow(self):
        """Test complete data preparation workflow."""
        with (
            patch.object(prepare, "download_data") as mock_download,
            patch.object(
                prepare, "list_parquet_files", return_value=[Path("shard.parquet")]
            ),
            patch.object(
                prepare, "train_tokenizer", return_value=True
            ) as mock_tokenizer,
        ):
            exit_code = prepare.main(["--num-shards", "2", "--download-workers", "2"])

            # Verify all steps were called
            assert exit_code == 0  # nosec: B101 - Test assertion
            mock_download.assert_called_once_with(2, 2)
            mock_tokenizer.assert_called_once()

    def test_preparation_with_tokenizer(self):
        """The tokenizer step runs after the download and gates success."""
        with (
            patch.object(prepare, "download_data"),
            patch.object(
                prepare, "list_parquet_files", return_value=[Path("shard.parquet")]
            ),
            patch.object(prepare, "train_tokenizer", return_value=False),
        ):
            assert prepare.main(["--num-shards", "1"]) == 1  # nosec: B101

    def test_error_recovery_workflow(self):
        """Test error recovery in data preparation."""
        with (
            patch.object(prepare, "download_data") as mock_download,
            patch.object(
                prepare, "list_parquet_files", return_value=[Path("shard.parquet")]
            ),
            patch.object(prepare, "train_tokenizer", return_value=True),
        ):

            # First run fails on a network error and reports failure
            mock_download.side_effect = Exception("Network error")
            assert prepare.main(["--num-shards", "1"]) == 1  # nosec: B101

            # Retry succeeds once the network recovers
            mock_download.side_effect = None
            assert prepare.main(["--num-shards", "1"]) == 0  # nosec: B101

            assert mock_download.call_count == 2  # nosec: B101 - Test assertion

    def test_concurrent_preparation(self):
        """Test concurrent data preparation."""
        import threading

        exit_codes = []

        with (
            patch.object(prepare, "download_data") as mock_download,
            patch.object(
                prepare, "list_parquet_files", return_value=[Path("shard.parquet")]
            ),
            patch.object(prepare, "train_tokenizer", return_value=True),
        ):
            threads = [
                threading.Thread(
                    target=lambda: exit_codes.append(
                        prepare.main(["--num-shards", "1"])
                    )
                )
                for _ in range(3)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            # Every concurrent run completes successfully
            assert exit_codes == [0, 0, 0]  # nosec: B101 - Test assertion
            assert mock_download.call_count == 3  # nosec: B101 - Test assertion


class TestPerformanceAndScalability:
    """Tests for performance and scalability."""

    def test_large_dataset_handling(self, data_dir, mock_pool):
        """Test handling of large datasets."""
        download_data(num_shards=100, download_workers=8)

        # 100 training shards plus the validation shard, bounded worker count
        mock_pool.pool_class.assert_called_once_with(processes=8)
        _, ids = mock_pool.map.call_args.args
        assert len(ids) == 101  # nosec: B101 - Test assertion

    def test_memory_efficiency(self):
        """make_dataloader is lazy: nothing is read until a batch is requested."""
        with patch.object(prepare, "_document_batches") as mock_batches:
            loader = make_dataloader(Mock(), B=8, T=512, split="train", buffer_size=4)

            mock_batches.assert_not_called()  # nosec: B101 - Test assertion
            assert hasattr(loader, "__next__")  # nosec: B101 - Test assertion

    def test_checkpoint_resumption(self, data_dir, mock_pool):
        """Shards that already exist are not downloaded again."""
        for index in [*range(3), VAL_SHARD]:
            (data_dir / f"shard_{index:05d}.parquet").touch()

        download_data(num_shards=3)

        mock_pool.pool_class.assert_not_called()  # nosec: B101

    def test_data_integrity_checks(self, data_dir):
        """A download that fails midway leaves neither a partial nor a final file."""

        def flaky_chunks(chunk_size=None):
            yield b"partial"
            raise ConnectionError("connection dropped")

        response = Mock()
        response.raise_for_status.return_value = None
        response.iter_content.side_effect = flaky_chunks

        with (
            patch.object(prepare.requests, "get", return_value=response),
            patch.object(prepare.time, "sleep"),
            patch("builtins.print"),
        ):
            assert download_single_shard(1) is False  # nosec: B101

        assert list(data_dir.iterdir()) == []  # nosec: B101 - Test assertion

    def test_configuration_preservation(self, data_dir):
        """An existing shard is preserved, never overwritten, by a later run."""
        shard = data_dir / "shard_00001.parquet"
        shard.write_bytes(b"original")

        with patch.object(prepare.requests, "get") as mock_get:
            assert download_single_shard(1) is True  # nosec: B101

        mock_get.assert_not_called()  # nosec: B101 - Test assertion
        assert shard.read_bytes() == b"original"  # nosec: B101
