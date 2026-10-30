"""
Enhanced test suite for prepare.py to achieve 90%+ coverage.

Targets uncovered areas:
- Data download functionality and error handling
- Tokenizer training and validation
- Data loading and batch processing
- Caching mechanisms and file operations
- Utility functions and edge cases
- Multiprocessing and parallel operations

All filesystem work happens in pytest temp directories (the module constants
are patched), and network, process pools and sleeps are mocked.
"""

import itertools
import math
from pathlib import Path
from typing import Iterator, List
from unittest.mock import Mock, patch

import pytest
import torch

import prepare
from prepare import (
    CACHE_DIR,
    VAL_FILENAME,
    VAL_SHARD,
    VOCAB_SIZE,
    cleanup_cache,
    download_data,
    download_file,
    download_shard,
    download_single_shard,
    download_worker,
    evaluate_bpb,
    get_cache_info,
    get_token_bytes,
    main,
    make_dataloader,
    parse_args,
    read_shard,
    text_iterator,
    train_tokenizer,
    validate_data_format,
    validate_tokenizer,
)


@pytest.fixture
def data_dir(tmp_path: Path) -> Iterator[Path]:
    """Point prepare.DATA_DIR at an empty temp directory."""
    directory = tmp_path / "data"
    directory.mkdir()
    with patch.object(prepare, "DATA_DIR", directory):
        yield directory


def _response(raise_for_status=None, chunks=(b"test data",), headers=None) -> Mock:
    """A fake ``requests`` response."""
    response = Mock()
    response.raise_for_status.side_effect = raise_for_status
    response.iter_content.return_value = list(chunks)
    response.headers = headers or {}
    return response


def _parquet_file(texts: List[str], row_groups: int = 1) -> Mock:
    """A fake ``pyarrow.parquet.ParquetFile`` yielding ``texts`` per row group."""
    row_group = Mock()
    row_group.column.return_value.to_pylist.return_value = texts
    parquet_file = Mock()
    parquet_file.num_row_groups = row_groups
    parquet_file.read_row_group.return_value = row_group
    return parquet_file


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


class TestDownloadFunctions:
    """Test data download functionality."""

    def test_download_single_shard_success(self, data_dir):
        """Test successful single shard download."""
        response = _response(chunks=[b"test ", b"data"])

        with patch.object(prepare.requests, "get", return_value=response):
            result = download_single_shard(1)

        assert result is True  # nosec: B101 - Test assertion
        shard = data_dir / "shard_00001.parquet"
        assert shard.read_bytes() == b"test data"  # nosec: B101 - Test assertion
        # The temporary download file is renamed into place, not left behind
        assert list(data_dir.glob("*.tmp")) == []  # nosec: B101 - Test assertion

    def test_download_single_shard_exists(self, data_dir):
        """Test shard download when file exists."""
        (data_dir / "shard_00001.parquet").touch()

        with patch.object(prepare.requests, "get") as mock_get:
            result = download_single_shard(1)

        assert result is True  # nosec: B101 - Test assertion
        mock_get.assert_not_called()  # nosec: B101 - Test assertion

    def test_download_single_shard_failure(self, data_dir):
        """Test shard download failure handling."""
        response = _response(raise_for_status=Exception("Network error"))

        with (
            patch.object(prepare.requests, "get", return_value=response) as mock_get,
            patch.object(prepare.time, "sleep"),
            patch("builtins.print"),
        ):
            result = download_single_shard(1)

        assert result is False  # nosec: B101 - Test assertion
        assert mock_get.call_count == 5  # nosec: B101 - Test assertion
        assert not (data_dir / "shard_00001.parquet").exists()  # nosec: B101

    def test_download_single_shard_retry_logic(self, data_dir):
        """Test retry logic with exponential backoff."""
        responses = [
            _response(raise_for_status=Exception("First failure")),
            _response(raise_for_status=Exception("Second failure")),
            _response(),  # Success on third try
        ]

        with (
            patch.object(prepare.requests, "get", side_effect=responses),
            patch.object(prepare.time, "sleep") as mock_sleep,
            patch("builtins.print"),
        ):
            result = download_single_shard(1)

        assert result is True  # nosec: B101 - Test assertion
        # Backoff doubles between attempts: 2s then 4s
        assert [c.args[0] for c in mock_sleep.call_args_list] == [
            2,
            4,
        ]  # nosec: B101 - Test assertion

    def test_download_data_parallel(self, data_dir):
        """Test parallel download functionality."""
        with (
            patch.object(prepare, "Pool") as mock_pool_class,
            patch("builtins.print"),
        ):
            pool = mock_pool_class.return_value.__enter__.return_value
            pool.map.return_value = [True] * 9

            download_data(8, download_workers=4)

        # 8 training shards plus the pinned validation shard, four workers
        mock_pool_class.assert_called_once_with(processes=4)
        func, ids = pool.map.call_args.args
        assert func is prepare.download_single_shard  # nosec: B101
        assert ids == [*range(8), VAL_SHARD]  # nosec: B101 - Test assertion

    def test_download_data_partial_existing(self, data_dir):
        """Test download with some shards already existing."""
        (data_dir / "shard_00000.parquet").touch()
        (data_dir / VAL_FILENAME).touch()

        with (
            patch.object(prepare, "Pool") as mock_pool_class,
            patch("builtins.print"),
        ):
            pool = mock_pool_class.return_value.__enter__.return_value
            pool.map.return_value = [True] * 4

            download_data(3, download_workers=4)

        # Two of the four shards are missing, so only two workers are needed
        mock_pool_class.assert_called_once_with(processes=2)
        _, ids = pool.map.call_args.args
        assert ids == [0, 1, 2, VAL_SHARD]  # nosec: B101 - Test assertion

    def test_download_file_function(self, tmp_path):
        """Test generic file download function."""
        target = tmp_path / "nested" / "test_file"
        response = _response(chunks=[b"test content"], headers={"content-length": "12"})

        with (
            patch.object(prepare.requests, "get", return_value=response),
            patch.object(prepare, "tqdm") as mock_tqdm,
        ):
            result = download_file("http://example.com/file", target)

        assert result is True  # nosec: B101 - Test assertion
        assert target.read_bytes() == b"test content"  # nosec: B101
        mock_tqdm.assert_called_once()

    def test_download_shard_worker_function(self, tmp_path):
        """download_shard delegates to download_single_shard."""
        with patch.object(
            prepare, "download_single_shard", return_value=True
        ) as mock_download:
            result = download_shard(1, tmp_path)

        assert result is True  # nosec: B101 - Test assertion
        mock_download.assert_called_once_with(1)

    def test_download_worker_function(self, tmp_path):
        """download_worker returns the shard index alongside its result."""
        with patch.object(
            prepare, "download_single_shard", return_value=True
        ) as mock_download:
            result = download_worker((1, tmp_path))

        assert result == (1, True)  # nosec: B101 - Test assertion
        mock_download.assert_called_once_with(1)


class TestTokenizerTraining:
    """Test tokenizer training functionality."""

    def test_train_tokenizer_success(self, tmp_path):
        """Test successful tokenizer training."""
        data = tmp_path / "data"
        tokenizer_dir = tmp_path / "tokenizer"
        trainer = Mock()

        # rustbpe.Trainer is patched with create=True so this tests prepare's
        # logic whichever rustbpe release is installed.
        with patch.object(
            prepare.rustbpe, "Trainer", create=True, return_value=trainer
        ) as mock_trainer:
            result = train_tokenizer(data_dir=data, tokenizer_dir=tokenizer_dir)

        assert result is True  # nosec: B101 - Test assertion
        mock_trainer.assert_called_once_with(vocab_size=VOCAB_SIZE, special_tokens=[""])
        trainer.load.assert_called_once_with(data)
        trainer.train.assert_called_once()
        trainer.save_as_tiktoken.assert_called_once_with(
            tokenizer_dir / "tokenizer.pkl"
        )
        # The byte table covers every byte value
        token_bytes = torch.load(tokenizer_dir / "token_bytes.pt")
        assert token_bytes.dtype == torch.uint8  # nosec: B101 - Test assertion
        assert token_bytes.tolist() == list(range(256))  # nosec: B101

    def test_train_tokenizer_failure(self, tmp_path):
        """Test tokenizer training failure handling."""
        with patch.object(
            prepare.rustbpe,
            "Trainer",
            create=True,
            side_effect=Exception("Training failed"),
        ):
            result = train_tokenizer(
                data_dir=tmp_path / "data", tokenizer_dir=tmp_path / "tokenizer"
            )

        assert result is False  # nosec: B101 - Test assertion

    def test_train_tokenizer_already_exists(self, tmp_path):
        """Test training when tokenizer already exists."""
        tokenizer_dir = tmp_path / "tokenizer"
        tokenizer_dir.mkdir()
        (tokenizer_dir / "tokenizer.pkl").touch()
        (tokenizer_dir / "token_bytes.pt").touch()

        with patch.object(prepare.rustbpe, "Trainer", create=True) as mock_trainer:
            result = train_tokenizer(data_dir=tmp_path, tokenizer_dir=tokenizer_dir)

        assert result is True  # nosec: B101 - Test assertion
        mock_trainer.assert_not_called()  # nosec: B101 - Test assertion

    def test_tokenizer_files_creation(self, tmp_path):
        """Training creates the tokenizer directory before writing into it."""
        tokenizer_dir = tmp_path / "tokenizer"

        with patch.object(
            prepare.rustbpe, "Trainer", create=True, side_effect=Exception("stop")
        ):
            train_tokenizer(data_dir=tmp_path, tokenizer_dir=tokenizer_dir)

        assert tokenizer_dir.is_dir()  # nosec: B101 - Test assertion

    def test_get_token_bytes_device_handling(self, tmp_path):
        """Token bytes are loaded onto the requested device."""
        (tmp_path / "token_bytes.pt").touch()
        tensor = torch.randn(256)

        with (
            patch.object(prepare, "TOKENIZER_DIR", tmp_path),
            patch.object(prepare.torch, "load", return_value=tensor) as mock_load,
        ):
            assert get_token_bytes("cpu") is tensor  # nosec: B101
            assert mock_load.call_args.kwargs["map_location"] == "cpu"  # nosec
            assert get_token_bytes("cuda") is tensor  # nosec: B101
            assert mock_load.call_args.kwargs["map_location"] == "cuda"  # nosec
            assert mock_load.call_count == 2  # nosec: B101 - Test assertion


class TestDataLoading:
    """Test data loading and batch processing."""

    def test_text_iterator_basic(self):
        """Test basic text iteration."""
        texts = ["text1", "text2", "text3"]
        parquet_file = _parquet_file(texts, row_groups=2)

        with (
            patch.object(
                prepare,
                "list_parquet_files",
                return_value=[Path("shard_00001.parquet")],
            ),
            patch.object(prepare.pq, "ParquetFile", return_value=parquet_file),
        ):
            docs = list(text_iterator())

        assert docs == texts * 2  # 2 row groups x 3 texts  # nosec: B101
        assert parquet_file.read_row_group.call_count == 2  # nosec: B101

    def test_text_iterator_doc_cap(self):
        """Test text iterator with document cap."""
        parquet_file = _parquet_file(["a" * 1000] * 10)  # Very long texts

        with (
            patch.object(
                prepare,
                "list_parquet_files",
                return_value=[Path("shard_00001.parquet")],
            ),
            patch.object(prepare.pq, "ParquetFile", return_value=parquet_file),
        ):
            docs = list(text_iterator(doc_cap=100))

        # Every document is kept but truncated to the cap
        assert len(docs) == 10  # nosec: B101 - Test assertion
        assert all(len(doc) == 100 for doc in docs)  # nosec: B101

    def test_text_iterator_no_files(self):
        """Test text iterator with no parquet files."""
        with patch.object(prepare, "list_parquet_files", return_value=[]):
            assert list(text_iterator()) == []  # nosec: B101 - Test assertion

    def test_text_iterator_skips_validation_shard(self):
        """The pinned validation shard is never used as training text."""
        parquet_file = _parquet_file(["train text"])

        with (
            patch.object(
                prepare,
                "list_parquet_files",
                return_value=[Path(VAL_FILENAME)],
            ),
            patch.object(prepare.pq, "ParquetFile", return_value=parquet_file),
        ):
            assert list(text_iterator()) == []  # nosec: B101 - Test assertion

        parquet_file.read_row_group.assert_not_called()  # nosec: B101

    @staticmethod
    def _tokenizer() -> Mock:
        """A tokenizer whose documents are ``[BOS, 2, 3, ...]`` of varying length."""
        tokenizer = Mock()
        tokenizer.get_bos_token_id.return_value = 1
        lengths = itertools.cycle([2, 3, 5, 4])

        def encode(batch, prepend=None):
            return [[prepend, *range(2, next(lengths) + 1)] for _ in batch]

        tokenizer.encode.side_effect = encode
        return tokenizer

    def test_make_dataloader_basic(self, unpinned_empty):
        """Test basic dataloader functionality."""
        tokenizer = self._tokenizer()
        batches = itertools.cycle([(["a", "b"], 1)])

        with patch.object(prepare, "_document_batches", return_value=batches):
            loader = make_dataloader(tokenizer, B=2, T=8, split="train", buffer_size=4)
            inputs, targets, epoch = next(loader)

        assert inputs.shape == (2, 8)  # nosec: B101 - Test assertion
        assert targets.shape == (2, 8)  # nosec: B101 - Test assertion
        assert epoch == 1  # nosec: B101 - Test assertion
        tokenizer.get_bos_token_id.assert_called_once()  # nosec: B101

    def test_make_dataloader_buffer_management(self, unpinned_empty):
        """Rows start with BOS, and targets are the inputs shifted by one."""
        tokenizer = self._tokenizer()
        batches = itertools.cycle([(["short", "longer text"], 3)])

        with patch.object(prepare, "_document_batches", return_value=batches):
            loader = make_dataloader(tokenizer, B=3, T=6, split="val", buffer_size=6)
            inputs, targets, epoch = next(loader)
            next_inputs, _, _ = next(loader)

        assert epoch == 3  # nosec: B101 - Test assertion
        # Best-fit packing never pads and always starts a row on a document
        assert bool((inputs[:, 0] == 1).all())  # nosec: B101 - Test assertion
        assert torch.equal(targets[:, :-1], inputs[:, 1:])  # nosec: B101
        assert next_inputs.shape == inputs.shape  # nosec: B101 - Test assertion

    def test_evaluate_bpb_function(self):
        """BPB is nats-per-byte over non-special tokens, converted to bits."""
        batch_size, steps = 2, 2
        # Token 0 is special (zero bytes); tokens 1-3 are one byte each
        token_bytes = torch.tensor([0, 1, 1, 1])
        x = torch.zeros(2, 2, dtype=torch.long)
        y = torch.tensor([[1, 2], [3, 0]])
        # Every token costs 1 nat; the special token is masked out of both sums
        model = Mock(return_value=torch.ones(4))

        with (
            patch.object(prepare, "get_token_bytes", return_value=token_bytes),
            patch.object(
                prepare, "make_dataloader", return_value=iter([(x, y, 1)] * steps)
            ),
            patch.object(
                prepare, "EVAL_TOKENS", batch_size * prepare.MAX_SEQ_LEN * steps
            ),
        ):
            result = evaluate_bpb(model, Mock(), batch_size)

        # 3 counted tokens x 1 nat over 3 bytes per step -> 1 nat/byte -> 1/ln 2
        assert result == pytest.approx(1 / math.log(2))  # nosec: B101


class TestUtilityFunctions:
    """Test utility functions."""

    def test_parse_args_default(self):
        """Test argument parsing with defaults."""
        args = parse_args([])

        # num_shards is resolved to a default of 10 by main(), not by the parser
        assert args.num_shards is None  # nosec: B101 - Test assertion
        assert args.skip_download is False  # nosec: B101 - Test assertion
        assert args.skip_tokenizer is False  # nosec: B101 - Test assertion
        assert args.download_workers == 8  # nosec: B101 - Test assertion
        assert args.force is False  # nosec: B101 - Test assertion

    def test_parse_args_custom(self):
        """Test argument parsing with custom values."""
        args = parse_args(
            ["--num-shards", "5", "--skip-download", "--download-workers", "4"]
        )

        assert args.num_shards == 5  # nosec: B101 - Test assertion
        assert args.skip_download is True  # nosec: B101 - Test assertion
        assert args.download_workers == 4  # nosec: B101 - Test assertion

    def test_get_cache_info(self, tmp_path):
        """Test cache information retrieval."""
        with patch.object(prepare, "CACHE_DIR", tmp_path):
            result = get_cache_info()

        stat = tmp_path.stat()
        assert result["size_bytes"] == int(stat.st_size)  # nosec: B101
        assert result["last_modified"] == int(stat.st_mtime)  # nosec: B101

    def test_get_cache_info_missing_directory(self, tmp_path):
        """A missing cache directory reports zeros."""
        with patch.object(prepare, "CACHE_DIR", tmp_path / "absent"):
            result = get_cache_info()

        assert result == {"size_bytes": 0, "last_modified": 0}  # nosec: B101

    def test_cleanup_cache_success(self):
        """Test successful cache cleanup."""
        with (
            patch.object(prepare.CACHE_DIR.__class__, "exists", return_value=True),
            patch("prepare.shutil.rmtree") as mock_rmtree,
        ):
            result = cleanup_cache()

            assert result is True  # nosec: B101 - Test assertion
            mock_rmtree.assert_called_once_with(CACHE_DIR)

    def test_cleanup_cache_no_directory(self):
        """Test cache cleanup when directory doesn't exist."""
        with (
            patch.object(prepare.CACHE_DIR.__class__, "exists", return_value=True),
            patch("prepare.shutil.rmtree") as mock_rmtree,
        ):
            mock_rmtree.side_effect = Exception("Directory not found")

            result = cleanup_cache()

            assert result is False  # nosec: B101 - Test assertion
            mock_rmtree.assert_called_once_with(CACHE_DIR)

    def test_validate_tokenizer_basic(self):
        """Test tokenizer validation."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "Hello, world!"

        result = validate_tokenizer(mock_tokenizer)

        assert result is True  # nosec: B101 - Test assertion
        mock_tokenizer.encode.assert_called_once_with("Hello, world!")
        mock_tokenizer.decode.assert_called_once_with([1, 2, 3])

    def test_validate_tokenizer_round_trip_mismatch(self):
        """A tokenizer that does not round-trip the probe text is rejected."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "hello world"

        assert validate_tokenizer(mock_tokenizer) is False  # nosec: B101

    def test_validate_tokenizer_failure(self):
        """Test tokenizer validation failure."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.side_effect = Exception("Encoding failed")

        result = validate_tokenizer(mock_tokenizer)

        assert result is False  # nosec: B101 - Test assertion

    def test_validate_data_format_valid(self):
        """Test valid data format validation."""
        valid_data = [{"text": "test"}, {"text": "another test"}]

        result = validate_data_format(valid_data)

        assert result is True  # nosec: B101 - Test assertion

    def test_validate_data_format_invalid(self):
        """Test invalid data format validation."""
        invalid_data: List[dict] = [{"not_text": "test"}]

        result = validate_data_format(invalid_data)

        assert result is False  # nosec: B101 - Test assertion

    def test_validate_data_format_missing_fields(self):
        """Test data format with missing required fields."""
        invalid_data = [{"not_text": "test"}]  # Missing 'text' field

        result = validate_data_format(invalid_data)

        assert result is False  # nosec: B101 - Test assertion


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_main_fails_when_download_produces_no_data(self):
        """main() reports failure when no parquet files exist after downloading."""
        with (
            patch.object(prepare, "download_data") as mock_download,
            patch.object(prepare, "list_parquet_files", return_value=[]),
            patch.object(prepare, "train_tokenizer") as mock_tokenizer,
        ):
            result = main(["--num-shards", "2"])

        assert result == 1  # nosec: B101 - Test assertion
        mock_download.assert_called_once_with(2, 8)
        mock_tokenizer.assert_not_called()  # nosec: B101 - Test assertion

    def test_network_timeout_handling(self, data_dir):
        """Test network timeout handling."""
        with (
            patch.object(
                prepare.requests, "get", side_effect=Exception("Request timeout")
            ),
            patch.object(prepare.time, "sleep"),
            patch("builtins.print"),
        ):
            result = download_single_shard(1)

        assert result is False  # nosec: B101 - Test assertion

    def test_insufficient_disk_space(self, tmp_path):
        """A full disk surfaces as an OSError rather than being swallowed."""
        with patch.object(
            prepare.os, "makedirs", side_effect=OSError("No space left on device")
        ):
            with pytest.raises(OSError, match="No space left"):
                train_tokenizer(
                    data_dir=tmp_path / "data", tokenizer_dir=tmp_path / "tokenizer"
                )

    def test_corrupted_file_handling(self, tmp_path):
        """Test handling of corrupted files."""
        (tmp_path / "shard_00001.parquet").touch()

        with patch.object(prepare.pq, "read_table", side_effect=Exception("corrupt")):
            result = read_shard(1, tmp_path)

        assert result is None  # nosec: B101 - Test assertion

    def test_memory_efficiency_large_dataset(self):
        """text_iterator streams documents and stops at the character budget."""
        parquet_file = _parquet_file(["x" * 5000] * 1000)  # 5M chars available

        with (
            patch.object(
                prepare,
                "list_parquet_files",
                return_value=[Path("shard_00001.parquet")],
            ),
            patch.object(prepare.pq, "ParquetFile", return_value=parquet_file),
        ):
            docs = list(text_iterator(max_chars=10_000, doc_cap=1000))

        # Each document is capped at 1000 chars, so a 10k budget stops at 10 docs
        assert len(docs) == 10  # nosec: B101 - Test assertion
        assert all(len(doc) <= 1000 for doc in docs)  # nosec: B101


if __name__ == "__main__":
    pytest.main([__file__])
