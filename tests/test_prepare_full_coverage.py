"""
Full coverage tests for prepare.py module.

This test file achieves 100% coverage by properly importing and testing
all code paths without blanket module-level mocking.
"""

from __future__ import annotations

import itertools
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, mock_open, patch

import pytest

from tests._module_isolation import mocked_modules

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def _isolated_prepare_imports() -> Generator[None, None, None]:
    """Mock tqdm and give each test a fresh ``prepare`` import.

    Tests here re-import ``prepare`` and replace modules in ``sys.modules``;
    scoping that to each test keeps it from leaking into other files.
    """
    with mocked_modules({"tqdm": MagicMock()}, fresh=["prepare"]):
        yield


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary cache directory."""
    cache_dir = tmp_path / ".cache" / "autoresearch"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yield cache_dir


@pytest.fixture
def mock_requests() -> Generator[Any, None, None]:
    """Mock requests module."""
    with patch.dict("sys.modules", {"requests": MagicMock()}):
        mock_requests = sys.modules["requests"]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "1000"}
        mock_response.iter_content.return_value = [b"test data"]
        mock_response.raise_for_status.return_value = None
        mock_requests.get.return_value = mock_response
        yield mock_requests


@pytest.fixture
def mock_pyarrow() -> Generator[MagicMock, None, None]:
    """Mock pyarrow.parquet module."""
    mock_pq = MagicMock()
    mock_table = MagicMock()
    mock_df = MagicMock()
    mock_df.__getitem__ = MagicMock(return_value=mock_df)
    mock_df.to_pylist.return_value = ["test text 1", "test text 2"]
    mock_table.to_pandas.return_value = mock_df
    mock_pq.read_table.return_value = mock_table

    # Mock ParquetFile
    mock_pf = MagicMock()
    mock_pf.num_row_groups = 1
    mock_rg = MagicMock()
    mock_rg.column.return_value.to_pylist.return_value = ["test text"]
    mock_pf.read_row_group.return_value = mock_rg
    mock_pq.ParquetFile.return_value = mock_pf

    # ``import pyarrow.parquet as pq`` binds the ``parquet`` attribute of the
    # ``pyarrow`` module object, so the package mock must expose the same mock.
    mock_pyarrow_pkg = MagicMock()
    mock_pyarrow_pkg.parquet = mock_pq

    with patch.dict(
        "sys.modules", {"pyarrow": mock_pyarrow_pkg, "pyarrow.parquet": mock_pq}
    ):
        yield mock_pq


@pytest.fixture
def mock_rustbpe() -> Generator[MagicMock, None, None]:
    """Mock rustbpe module."""
    mock_trainer = MagicMock()
    mock_trainer_instance = MagicMock()
    mock_trainer_instance.load.return_value = None
    mock_trainer_instance.train.return_value = None
    mock_trainer_instance.save_as_tiktoken.return_value = None
    mock_trainer.return_value = mock_trainer_instance

    mock_encoder = MagicMock()
    mock_encoder.n_vocab = 8192
    mock_encoder.encode_single_token.return_value = 0
    mock_encoder.encode_ordinary.return_value = [1, 2, 3]
    mock_encoder.encode_ordinary_batch.return_value = [[1, 2], [3, 4]]
    mock_encoder.decode.return_value = "decoded text"

    mock_rustbpe = MagicMock()
    mock_rustbpe.Trainer = mock_trainer
    mock_rustbpe.Encoder = mock_encoder

    with patch.dict("sys.modules", {"rustbpe": mock_rustbpe}):
        yield mock_rustbpe


@pytest.fixture
def mock_torch() -> Generator[MagicMock, None, None]:
    """Mock torch module."""
    mock_tensor = MagicMock()
    mock_tensor.item.return_value = 1.0
    mock_tensor.sum.return_value = mock_tensor
    mock_tensor.view.return_value = mock_tensor

    mock_torch = MagicMock()
    mock_tensor_class = MagicMock()
    mock_tensor_class.return_value = mock_tensor
    mock_torch.Tensor = mock_tensor_class
    mock_torch.tensor = MagicMock(return_value=mock_tensor)
    mock_torch.empty = MagicMock(return_value=mock_tensor)
    mock_torch.zeros = MagicMock(return_value=mock_tensor)
    mock_torch.uint8 = MagicMock()
    mock_torch.long = MagicMock()
    mock_torch.no_grad = MagicMock(
        return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
    )
    mock_torch.cuda.is_available.return_value = False

    with patch.dict("sys.modules", {"torch": mock_torch}):
        yield mock_torch


# =============================================================================
# TESTS FOR MULTIPROCESSING START METHOD (lines 12-22)
# =============================================================================


class TestMultiprocessingSetup:
    """Test multiprocessing start method setup for macOS."""

    @patch.object(sys, "platform", "darwin")
    @patch("multiprocessing.set_start_method")
    def test_macos_spawn_setup(self, mock_set_start: MagicMock) -> None:
        """Test macOS multiprocessing spawn setup."""
        # Force reimport by clearing prepare from cache
        if "prepare" in sys.modules:
            del sys.modules["prepare"]

        with patch.dict("sys.modules", {"multiprocessing": MagicMock()}):
            mock_mp = sys.modules["multiprocessing"]
            mock_mp.set_start_method = mock_set_start  # type: ignore[attr-defined]

            # Import should call set_start_method
            import prepare as _prep  # noqa: F401

            mock_set_start.assert_called_once_with("spawn", force=True)

    @patch.object(sys, "platform", "darwin")
    @patch("multiprocessing.set_start_method")
    def test_macos_spawn_already_set(self, mock_set_start: MagicMock) -> None:
        """Test when spawn method is already set."""
        mock_set_start.side_effect = RuntimeError("Already set")

        if "prepare" in sys.modules:
            del sys.modules["prepare"]

        with patch.dict("sys.modules", {"multiprocessing": MagicMock()}):
            mock_mp = sys.modules["multiprocessing"]
            mock_mp.set_start_method = mock_set_start  # type: ignore[attr-defined]

            # Should not raise exception
            import prepare as _prep  # noqa: F401

            # Should have been called but handled gracefully

    @patch.object(sys, "platform", "linux")
    def test_non_macos_no_spawn_setup(self) -> None:
        """Test that non-macOS platforms don't set spawn method."""
        if "prepare" in sys.modules:
            del sys.modules["prepare"]

        with patch.dict("sys.modules", {"multiprocessing": MagicMock()}):
            mock_mp = sys.modules["multiprocessing"]
            mock_mp.set_start_method = MagicMock()  # type: ignore[attr-defined]

            import prepare as _prep  # noqa: F401

            mock_mp.set_start_method.assert_not_called()


# =============================================================================
# TESTS FOR DOWNLOAD SINGLE SHARD (lines 78-118)
# =============================================================================


class TestDownloadSingleShard:
    """Test download_single_shard function with all code paths."""

    def test_shard_already_exists(
        self, mock_requests: MagicMock, temp_cache_dir: Path
    ) -> None:
        """Test when shard already exists."""
        with patch("prepare.DATA_DIR", temp_cache_dir):
            # Create the file
            shard_file = temp_cache_dir / "shard_00001.parquet"
            shard_file.touch()

            from prepare import download_single_shard

            result = download_single_shard(1)
            assert result is True  # nosec: B101 - Test assertion
            mock_requests.get.assert_not_called()

    def test_successful_download(
        self, mock_requests: MagicMock, temp_cache_dir: Path
    ) -> None:
        """Test successful download."""
        with patch("prepare.DATA_DIR", temp_cache_dir):
            with patch("tempfile.NamedTemporaryFile") as mock_temp:
                mock_temp_file = MagicMock()
                mock_temp_file.name = str(temp_cache_dir / "temp.tmp")
                mock_temp.return_value.__enter__.return_value = mock_temp_file
                mock_temp.return_value.__exit__ = MagicMock(return_value=None)

                with patch("os.replace"):
                    with patch("builtins.print"):
                        from prepare import download_single_shard

                        result = download_single_shard(1)
                        assert result is True  # nosec: B101 - Test assertion

    def test_download_with_retries(
        self, mock_requests: MagicMock, temp_cache_dir: Path
    ) -> None:
        """Test download with retry logic."""
        with patch("prepare.DATA_DIR", temp_cache_dir):
            # Make first calls fail, then succeed
            mock_response_fail = MagicMock()
            mock_response_fail.raise_for_status.side_effect = Exception("Network error")

            mock_response_success = MagicMock()
            mock_response_success.raise_for_status.return_value = None
            mock_response_success.iter_content.return_value = [b"data"]

            mock_requests.get.side_effect = [
                mock_response_fail,
                mock_response_fail,
                mock_response_success,
            ]

            with patch("tempfile.NamedTemporaryFile") as mock_temp:
                mock_temp_file = MagicMock()
                mock_temp_file.name = str(temp_cache_dir / "temp.tmp")
                mock_temp.return_value.__enter__.return_value = mock_temp_file
                mock_temp.return_value.__exit__ = MagicMock(return_value=None)

                with patch("os.replace"):
                    with patch("os.remove"):
                        with patch("time.sleep"):
                            with patch("builtins.print"):
                                from prepare import download_single_shard

                                result = download_single_shard(1)
                                assert result is True  # nosec: B101 - Test assertion

    def test_all_retries_fail(
        self, mock_requests: MagicMock, temp_cache_dir: Path
    ) -> None:
        """Test when all retry attempts fail."""
        with patch("prepare.DATA_DIR", temp_cache_dir):
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = Exception("Network error")
            mock_requests.get.return_value = mock_response

            with patch("time.sleep"):
                with patch("builtins.print"):
                    with patch("os.remove"):
                        from prepare import download_single_shard

                        result = download_single_shard(1)
                        assert result is False  # nosec: B101 - Test assertion


# =============================================================================
# TESTS FOR DOWNLOAD DATA (lines 121-143)
# =============================================================================


class TestDownloadData:
    """Test download_data function."""

    def test_all_shards_exist(self, temp_cache_dir: Path) -> None:
        """Test when all shards already exist."""
        with patch("prepare.DATA_DIR", temp_cache_dir):
            from prepare import VAL_FILENAME, download_data

            # Create existing shards, including the pinned validation shard
            for i in range(3):
                shard = temp_cache_dir / f"shard_{i:05d}.parquet"
                shard.touch()
            (temp_cache_dir / VAL_FILENAME).touch()

            with patch("builtins.print"):

                # Should complete without downloading
                download_data(3, 8)

    @patch("prepare.Pool")
    def test_partial_download(self, mock_pool: MagicMock, temp_cache_dir: Path) -> None:
        """Test downloading when some shards exist."""
        with patch("prepare.DATA_DIR", temp_cache_dir):
            # Create only first shard
            shard = temp_cache_dir / "shard_00000.parquet"
            shard.touch()

            mock_pool_instance = MagicMock()
            mock_pool.return_value.__enter__.return_value = mock_pool_instance
            mock_pool_instance.map.return_value = [True, True]  # shards 1, 2 succeed

            with patch("builtins.print"):
                from prepare import download_data

                download_data(3, 8)

            mock_pool_instance.map.assert_called_once()


# =============================================================================
# TESTS FOR LIST PARQUET FILES (lines 151-158)
# =============================================================================


class TestListParquetFiles:
    """Test list_parquet_files function."""

    def test_list_parquet_files(self, temp_cache_dir: Path) -> None:
        """Test listing parquet files."""
        with patch("prepare.DATA_DIR", temp_cache_dir):
            # Create test files
            (temp_cache_dir / "shard_00001.parquet").touch()
            (temp_cache_dir / "shard_00002.parquet").touch()
            (temp_cache_dir / "shard_00003.parquet.tmp").touch()  # Should be excluded
            (temp_cache_dir / "readme.txt").touch()  # Should be excluded

            from prepare import list_parquet_files

            files = list_parquet_files()
            assert len(files) == 2  # nosec: B101 - Test assertion
            assert all(
                f.suffix == ".parquet" for f in files
            )  # nosec: B101 - Test assertion
            assert not any(
                str(f).endswith(".tmp") for f in files
            )  # nosec: B101 - Test assertion


# =============================================================================
# TESTS FOR TEXT ITERATOR (lines 161-176)
# =============================================================================


class TestTextIterator:
    """Test text_iterator function."""

    def test_text_iterator_with_limit(
        self, temp_cache_dir: Path, mock_pyarrow: MagicMock
    ) -> None:
        """Test text iterator with character limit."""
        with patch("prepare.DATA_DIR", temp_cache_dir):
            # Create a mock parquet file
            (temp_cache_dir / "shard_00001.parquet").touch()

            from prepare import text_iterator

            texts = list(text_iterator(max_chars=50))
            assert len(texts) > 0  # nosec: B101 - Test assertion

    def test_text_iterator_doc_cap(
        self, temp_cache_dir: Path, mock_pyarrow: MagicMock
    ) -> None:
        """Test text iterator with document cap."""
        with patch("prepare.DATA_DIR", temp_cache_dir):
            (temp_cache_dir / "shard_00001.parquet").touch()

            from prepare import text_iterator

            texts = list(text_iterator(max_chars=1000, doc_cap=10))
            for text in texts:
                assert len(text) <= 10  # nosec: B101 - Test assertion


# =============================================================================
# TESTS FOR TRAIN TOKENIZER (lines 179-212)
# =============================================================================


class TestTrainTokenizer:
    """Test train_tokenizer function."""

    def test_tokenizer_already_exists(
        self, temp_cache_dir: Path, mock_rustbpe: MagicMock, mock_torch: MagicMock
    ) -> None:
        """Test when tokenizer already exists."""
        tokenizer_dir = temp_cache_dir / "tokenizer"
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        (tokenizer_dir / "tokenizer.pkl").touch()
        (tokenizer_dir / "token_bytes.pt").touch()

        from prepare import train_tokenizer

        # The defaults are bound at definition time, so patching the module
        # constants would not affect them; pass the directories explicitly.
        result = train_tokenizer(data_dir=temp_cache_dir, tokenizer_dir=tokenizer_dir)
        assert result is True  # nosec: B101 - Test assertion
        mock_rustbpe.Trainer.assert_not_called()

    def test_train_tokenizer_success(
        self, temp_cache_dir: Path, mock_rustbpe: MagicMock, mock_torch: MagicMock
    ) -> None:
        """Test successful tokenizer training."""
        tokenizer_dir = temp_cache_dir / "tokenizer"
        data_dir = temp_cache_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        with patch("prepare.DATA_DIR", data_dir):
            with patch("prepare.TOKENIZER_DIR", tokenizer_dir):
                with patch("builtins.open", mock_open()):
                    with patch("os.makedirs"):
                        from prepare import train_tokenizer

                        result = train_tokenizer(data_dir, tokenizer_dir)
                        assert result is True  # nosec: B101 - Test assertion

    def test_train_tokenizer_failure(
        self, temp_cache_dir: Path, mock_rustbpe: MagicMock, mock_torch: MagicMock
    ) -> None:
        """Test tokenizer training failure."""
        mock_rustbpe.Trainer.side_effect = Exception("Training failed")

        tokenizer_dir = temp_cache_dir / "tokenizer"
        data_dir = temp_cache_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        with patch("prepare.DATA_DIR", data_dir):
            with patch("prepare.TOKENIZER_DIR", tokenizer_dir):
                from prepare import train_tokenizer

                result = train_tokenizer(data_dir, tokenizer_dir)
                assert result is False  # nosec: B101 - Test assertion


# =============================================================================
# TESTS FOR TOKENIZER CLASS (lines 220-264)
# =============================================================================


class TestTokenizer:
    """Test Tokenizer class."""

    def test_tokenizer_init(self, mock_rustbpe: MagicMock) -> None:
        """Test Tokenizer initialization."""
        mock_enc = MagicMock()
        mock_enc.encode_single_token.return_value = 0

        from prepare import Tokenizer

        tokenizer = Tokenizer(mock_enc)
        assert tokenizer.bos_token_id == 0  # nosec: B101 - Test assertion

    def test_get_vocab_size(self, mock_rustbpe: MagicMock) -> None:
        """Test get_vocab_size method."""
        mock_enc = MagicMock()
        mock_enc.n_vocab = 8192

        from prepare import Tokenizer

        tokenizer = Tokenizer(mock_enc)
        assert tokenizer.get_vocab_size() == 8192  # nosec: B101 - Test assertion

    def test_get_bos_token_id(self, mock_rustbpe: MagicMock) -> None:
        """Test get_bos_token_id method."""
        mock_enc = MagicMock()
        mock_enc.encode_single_token.return_value = 0

        from prepare import Tokenizer

        tokenizer = Tokenizer(mock_enc)
        assert tokenizer.get_bos_token_id() == 0  # nosec: B101 - Test assertion

    def test_encode_string(self, mock_rustbpe: MagicMock) -> None:
        """Test encode with string input."""
        mock_enc = MagicMock()
        mock_enc.encode_ordinary.return_value = [1, 2, 3]

        from prepare import Tokenizer

        tokenizer = Tokenizer(mock_enc)
        result = tokenizer.encode("test text")
        assert result == [1, 2, 3]  # nosec: B101 - Test assertion

    def test_encode_with_prepend_int(self, mock_rustbpe: MagicMock) -> None:
        """Test encode with integer prepend."""
        mock_enc = MagicMock()
        mock_enc.encode_ordinary.return_value = [1, 2, 3]

        from prepare import Tokenizer

        tokenizer = Tokenizer(mock_enc)
        result = tokenizer.encode("test", prepend=0)
        assert result[0] == 0  # nosec: B101 - Test assertion

    def test_encode_with_prepend_str(self, mock_rustbpe: MagicMock) -> None:
        """Test encode with string prepend."""
        mock_enc = MagicMock()
        mock_enc.encode_ordinary.return_value = [1, 2, 3]
        mock_enc.encode_single_token.return_value = 0

        from prepare import Tokenizer

        tokenizer = Tokenizer(mock_enc)
        result = tokenizer.encode("test", prepend="<bos>")
        assert result[0] == 0  # nosec: B101 - Test assertion

    def test_encode_list(self, mock_rustbpe: MagicMock) -> None:
        """Test encode with list input."""
        mock_enc = MagicMock()
        mock_enc.encode_ordinary_batch.return_value = [[1, 2], [3, 4]]

        from prepare import Tokenizer

        tokenizer = Tokenizer(mock_enc)
        result = tokenizer.encode(["text1", "text2"])
        assert len(result) == 2  # nosec: B101 - Test assertion

    def test_encode_list_with_prepend(self, mock_rustbpe: MagicMock) -> None:
        """Test encode list with prepend."""
        mock_enc = MagicMock()
        mock_enc.encode_ordinary_batch.return_value = [[1, 2], [3, 4]]
        mock_enc.encode_single_token.return_value = 0

        from prepare import Tokenizer

        tokenizer = Tokenizer(mock_enc)
        result = tokenizer.encode(["text1", "text2"], prepend=0)
        assert all(r[0] == 0 for r in result)  # type: ignore[index]  # nosec: B101 - Test assertion

    def test_encode_invalid_type(self, mock_rustbpe: MagicMock) -> None:
        """Test encode with invalid input type."""
        mock_enc = MagicMock()

        from prepare import Tokenizer

        tokenizer = Tokenizer(mock_enc)
        with pytest.raises(ValueError):
            tokenizer.encode(123)  # Invalid type

    def test_decode(self, mock_rustbpe: MagicMock) -> None:
        """Test decode method."""
        mock_enc = MagicMock()
        mock_enc.decode.return_value = "decoded text"

        from prepare import Tokenizer

        tokenizer = Tokenizer(mock_enc)
        result = tokenizer.decode([1, 2, 3])
        assert result == "decoded text"  # nosec: B101 - Test assertion

    @patch("pickle.load")
    @patch("builtins.open", mock_open())
    def test_from_directory(
        self, mock_pickle: MagicMock, mock_rustbpe: MagicMock
    ) -> None:
        """Test from_directory class method."""
        mock_enc = MagicMock()
        mock_enc.n_vocab = 8192
        mock_enc.encode_single_token.return_value = 0
        mock_pickle.return_value = mock_enc

        with patch("prepare.TOKENIZER_DIR", Path("/tmp/tokenizer")):
            from prepare import Tokenizer

            tokenizer = Tokenizer.from_directory()
            assert tokenizer.get_vocab_size() == 8192  # nosec: B101 - Test assertion


# =============================================================================
# TESTS FOR GET TOKEN BYTES (lines 266-270)
# =============================================================================


class TestGetTokenBytes:
    """Test get_token_bytes function."""

    def test_get_token_bytes(self, temp_cache_dir: Path, mock_torch: MagicMock) -> None:
        """Test getting token bytes."""
        mock_tensor = MagicMock()
        mock_torch.load.return_value = mock_tensor
        tokenizer_dir = temp_cache_dir / "tokenizer"
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        (tokenizer_dir / "token_bytes.pt").touch()

        with patch("prepare.TOKENIZER_DIR", tokenizer_dir):
            from prepare import get_token_bytes

            result = get_token_bytes("cpu")
            assert result is mock_tensor  # nosec: B101 - Test assertion
            assert mock_torch.load.call_args.kwargs["map_location"] == "cpu"  # nosec


# =============================================================================
# TESTS FOR DOCUMENT BATCHES (lines 273-294)
# =============================================================================


class TestDocumentBatches:
    """Test _document_batches function."""

    def test_document_batches_train(
        self, temp_cache_dir: Path, mock_pyarrow: MagicMock
    ) -> None:
        """Test document batches for train split."""
        with patch("prepare.DATA_DIR", temp_cache_dir):
            (temp_cache_dir / "shard_00001.parquet").touch()

            from prepare import _document_batches

            batches_gen = _document_batches("train", tokenizer_batch_size=2)
            # Get first batch
            batch, epoch = next(batches_gen)
            assert isinstance(batch, list)  # nosec: B101 - Test assertion
            assert epoch >= 1  # nosec: B101 - Test assertion

    def test_document_batches_val(
        self, temp_cache_dir: Path, mock_pyarrow: MagicMock
    ) -> None:
        """Test document batches for val split."""
        with patch("prepare.DATA_DIR", temp_cache_dir):
            (temp_cache_dir / "shard_06542.parquet").touch()

            from prepare import _document_batches

            batches_gen = _document_batches("val", tokenizer_batch_size=2)
            batch, epoch = next(batches_gen)
            assert isinstance(batch, list)  # nosec: B101 - Test assertion


# =============================================================================
# TESTS FOR MAKE DATALOADER (lines 297-370)
# =============================================================================


class TestMakeDataloader:
    """Test make_dataloader function."""

    @patch("torch.cuda.is_available")
    @patch("prepare._document_batches")
    def test_make_dataloader_cpu(
        self,
        mock_batches: MagicMock,
        mock_cuda: MagicMock,
        mock_torch: MagicMock,
        mock_rustbpe: MagicMock,
    ) -> None:
        """Test dataloader on CPU."""
        mock_cuda.return_value = False

        mock_tokenizer = MagicMock()
        mock_tokenizer.get_bos_token_id.return_value = 0
        mock_tokenizer.encode.return_value = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

        # The dataloader refills its buffer in a loop, so the source must not run dry
        mock_batches.return_value = itertools.cycle([(["text1"], 1)])

        mock_tensor = MagicMock()
        mock_tensor.copy_ = MagicMock()
        mock_torch.empty.return_value = mock_tensor

        from prepare import make_dataloader

        loader = make_dataloader(mock_tokenizer, 2, 4, "train")
        x, y, epoch = next(loader)
        assert x is not None  # nosec: B101 - Test assertion
        assert y is not None  # nosec: B101 - Test assertion


# =============================================================================
# TESTS FOR EVALUATE BPB (lines 378-402)
# =============================================================================


class TestEvaluateBpb:
    """Test evaluate_bpb function."""

    def test_evaluate_bpb(self) -> None:
        """BPB is nats-per-byte over non-special tokens, converted to bits."""
        import torch

        # evaluate_bpb is decorated with @torch.no_grad() at import time, so it
        # must be imported against the real torch, not the mock_torch fixture.
        import prepare as real_prep

        batch_size, steps = 2, 2
        # Token 0 is special (zero bytes); tokens 1-3 are one byte each
        token_bytes = torch.tensor([0, 1, 1, 1])
        x = torch.zeros(2, 2, dtype=torch.long)
        y = torch.tensor([[1, 2], [3, 0]])

        # Every token costs 1 nat; the special token is masked out of both sums
        model = MagicMock(return_value=torch.ones(4))

        with (
            patch.object(real_prep, "get_token_bytes", return_value=token_bytes),
            patch.object(
                real_prep, "make_dataloader", return_value=iter([(x, y, 1)] * steps)
            ),
            patch.object(
                real_prep, "EVAL_TOKENS", batch_size * real_prep.MAX_SEQ_LEN * steps
            ),
        ):
            result = real_prep.evaluate_bpb(model, MagicMock(), batch_size, "cpu")

        # 3 counted tokens x 1 nat over 3 bytes per step -> 1 nat/byte -> 1/ln 2
        assert result == pytest.approx(1 / math.log(2))  # nosec: B101 - Test assertion


# =============================================================================
# TESTS FOR ADDITIONAL FUNCTIONS (lines 410-507)
# =============================================================================


class TestAdditionalFunctions:
    """Test additional helper functions."""

    def test_download_file_success(self, mock_requests: MagicMock) -> None:
        """Test successful file download."""
        with tempfile.TemporaryDirectory() as tmp:
            filepath = Path(tmp) / "test.txt"

            from prepare import download_file

            with patch("tqdm.tqdm") as mock_tqdm:
                mock_tqdm.return_value.__enter__ = MagicMock(return_value=mock_tqdm)
                mock_tqdm.return_value.__exit__ = MagicMock(return_value=None)

                result = download_file("http://example.com/test", filepath)
                assert result is True  # nosec: B101 - Test assertion

    def test_download_file_already_exists(self, temp_cache_dir: Path) -> None:
        """Test download when file already exists."""
        filepath = temp_cache_dir / "test.txt"
        filepath.touch()

        from prepare import download_file

        result = download_file("http://example.com/test", filepath)
        assert result is True  # nosec: B101 - Test assertion

    def test_download_file_error(self, mock_requests: MagicMock) -> None:
        """Test download with error."""
        mock_requests.get.side_effect = Exception("Network error")

        with tempfile.TemporaryDirectory() as tmp:
            filepath = Path(tmp) / "test.txt"

            from prepare import download_file

            result = download_file("http://example.com/test", filepath)
            assert result is False  # nosec: B101 - Test assertion

    def test_tokenize_text(self, mock_rustbpe: MagicMock) -> None:
        """Test tokenize_text function."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]

        from prepare import tokenize_text

        result = tokenize_text("test", mock_tokenizer)
        assert result == [1, 2, 3]  # nosec: B101 - Test assertion

    def test_tokenize_batch(self, mock_rustbpe: MagicMock) -> None:
        """Test tokenize_batch function."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]

        from prepare import tokenize_batch

        result = tokenize_batch(["text1", "text2"], mock_tokenizer)
        assert len(result) == 2  # nosec: B101 - Test assertion
        assert all(
            isinstance(tokens, list) for tokens in result
        )  # nosec: B101 - Test assertion

    def test_validate_tokenizer_success(self, mock_rustbpe: MagicMock) -> None:
        """Test successful tokenizer validation."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "Hello, world!"

        from prepare import validate_tokenizer

        result = validate_tokenizer(mock_tokenizer)
        assert result is True  # nosec: B101 - Test assertion

    def test_validate_tokenizer_failure(self, mock_rustbpe: MagicMock) -> None:
        """Test tokenizer validation failure."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = Exception("Validation error")

        from prepare import validate_tokenizer

        result = validate_tokenizer(mock_tokenizer)
        assert result is False  # nosec: B101 - Test assertion

    def test_validate_data_format_valid(self) -> None:
        """Test valid data format."""
        valid_data = [{"text": "sample 1"}, {"text": "sample 2"}]

        from prepare import validate_data_format

        result = validate_data_format(valid_data)
        assert result is True  # nosec: B101 - Test assertion

    def test_validate_data_format_empty(self) -> None:
        """Test empty data format."""
        from prepare import validate_data_format

        result = validate_data_format([])
        assert result is False  # nosec: B101 - Test assertion

    def test_validate_data_format_missing_text(self) -> None:
        """Test data format missing text key."""
        invalid_data = [{"content": "sample"}]

        from prepare import validate_data_format

        result = validate_data_format(invalid_data)
        assert result is False  # nosec: B101 - Test assertion

    def test_validate_data_format_non_string_text(self) -> None:
        """Test data format with non-string text."""
        invalid_data = [{"text": 123}]

        from prepare import validate_data_format

        result = validate_data_format(invalid_data)
        assert result is False  # nosec: B101 - Test assertion


# =============================================================================
# TESTS FOR PARSE ARGS (lines 510-521)
# =============================================================================


class TestParseArgs:
    """Test parse_args function."""

    def test_parse_args_defaults(self) -> None:
        """Test default argument parsing."""
        from prepare import parse_args

        args = parse_args([])
        assert args.num_shards is None  # nosec: B101 - Test assertion
        assert args.skip_download is False  # nosec: B101 - Test assertion
        assert args.skip_tokenizer is False  # nosec: B101 - Test assertion
        assert args.force is False  # nosec: B101 - Test assertion
        assert args.download_workers == 8  # nosec: B101 - Test assertion

    def test_parse_args_custom(self) -> None:
        """Test custom argument parsing."""
        from prepare import parse_args

        args = parse_args(
            [
                "--num-shards",
                "10",
                "--skip-download",
                "--skip-tokenizer",
                "--force",
                "--download-workers",
                "4",
            ]
        )
        assert args.num_shards == 10  # nosec: B101 - Test assertion
        assert args.skip_download is True  # nosec: B101 - Test assertion
        assert args.skip_tokenizer is True  # nosec: B101 - Test assertion
        assert args.force is True  # nosec: B101 - Test assertion
        assert args.download_workers == 4  # nosec: B101 - Test assertion


# =============================================================================
# TESTS FOR MAIN FUNCTION (lines 524-547)
# =============================================================================


class TestMain:
    """Test main function."""

    @patch("prepare.download_data")
    @patch("prepare.train_tokenizer")
    @patch("prepare.list_parquet_files")
    def test_main_success(
        self, mock_list: MagicMock, mock_train: MagicMock, mock_download: MagicMock
    ) -> None:
        """Test successful main execution."""
        mock_list.return_value = [Path("test.parquet")]
        mock_train.return_value = True

        from prepare import main

        result = main([])
        assert result == 0  # nosec: B101 - Test assertion
        mock_download.assert_called_once()
        mock_train.assert_called_once()

    @patch("prepare.download_data")
    @patch("prepare.train_tokenizer")
    @patch("prepare.list_parquet_files")
    def test_main_skip_download(
        self, mock_list: MagicMock, mock_train: MagicMock, mock_download: MagicMock
    ) -> None:
        """Test main with skip download."""
        mock_list.return_value = [Path("test.parquet")]
        mock_train.return_value = True

        from prepare import main

        result = main(["--skip-download"])
        assert result == 0  # nosec: B101 - Test assertion
        mock_download.assert_not_called()
        mock_train.assert_called_once()

    @patch("prepare.download_data")
    @patch("prepare.train_tokenizer")
    @patch("prepare.list_parquet_files")
    def test_main_skip_tokenizer(
        self, mock_list: MagicMock, mock_train: MagicMock, mock_download: MagicMock
    ) -> None:
        """Test main with skip tokenizer."""
        mock_list.return_value = [Path("test.parquet")]

        from prepare import main

        result = main(["--skip-tokenizer"])
        assert result == 0  # nosec: B101 - Test assertion
        mock_download.assert_called_once()
        mock_train.assert_not_called()

    @patch("prepare.download_data")
    @patch("prepare.list_parquet_files")
    def test_main_download_failure(
        self, mock_list: MagicMock, mock_download: MagicMock
    ) -> None:
        """Test main with download failure."""
        mock_list.return_value = []  # No files downloaded

        from prepare import main

        result = main([])
        assert result == 1  # nosec: B101 - Test assertion

    @patch("prepare.download_data")
    @patch("prepare.train_tokenizer")
    @patch("prepare.list_parquet_files")
    def test_main_tokenizer_failure(
        self, mock_list: MagicMock, mock_train: MagicMock, mock_download: MagicMock
    ) -> None:
        """Test main with tokenizer failure."""
        mock_list.return_value = [Path("test.parquet")]
        mock_train.return_value = False

        from prepare import main

        result = main([])
        assert result == 1  # nosec: B101 - Test assertion

    @patch("prepare.download_data")
    @patch("prepare.train_tokenizer")
    @patch("prepare.list_parquet_files")
    def test_main_exception(
        self, mock_list: MagicMock, mock_train: MagicMock, mock_download: MagicMock
    ) -> None:
        """Test main with exception."""
        mock_download.side_effect = Exception("Unexpected error")

        from prepare import main

        result = main([])
        assert result == 1  # nosec: B101 - Test assertion


# =============================================================================
# TESTS FOR CACHE FUNCTIONS (lines 549-571)
# =============================================================================


class TestCacheFunctions:
    """Test cache utility functions."""

    def test_get_cache_info_exists(self, temp_cache_dir: Path) -> None:
        """Test getting cache info when cache exists."""
        with patch("prepare.CACHE_DIR", temp_cache_dir):
            # Create a file in cache
            (temp_cache_dir / "test.txt").write_text("test content")

            from prepare import get_cache_info

            info = get_cache_info()
            assert isinstance(info, dict)  # nosec: B101 - Test assertion
            assert "size_bytes" in info  # nosec: B101 - Test assertion
            assert "last_modified" in info  # nosec: B101 - Test assertion

    def test_get_cache_info_not_exists(self) -> None:
        """Test getting cache info when cache doesn't exist."""
        with patch("prepare.CACHE_DIR", Path("/nonexistent/path")):
            from prepare import get_cache_info

            info = get_cache_info()
            assert isinstance(info, dict)  # nosec: B101 - Test assertion
            assert info["size_bytes"] == 0  # nosec: B101 - Test assertion

    def test_cleanup_cache_success(self, temp_cache_dir: Path) -> None:
        """Test successful cache cleanup."""
        with patch("prepare.CACHE_DIR", temp_cache_dir):
            (temp_cache_dir / "test.txt").write_text("test")

            from prepare import cleanup_cache

            result = cleanup_cache()
            assert result is True  # nosec: B101 - Test assertion

    def test_cleanup_cache_not_exists(self) -> None:
        """Test cleanup when cache doesn't exist."""
        with patch("prepare.CACHE_DIR", Path("/nonexistent/path")):
            from prepare import cleanup_cache

            result = cleanup_cache()
            assert result is True  # nosec: B101 - Test assertion

    def test_cleanup_cache_failure(self, temp_cache_dir: Path) -> None:
        """Test cache cleanup failure."""
        with patch("prepare.CACHE_DIR", temp_cache_dir):
            with patch("shutil.rmtree", side_effect=Exception("Permission denied")):
                from prepare import cleanup_cache

                result = cleanup_cache()
                assert result is False  # nosec: B101 - Test assertion


# =============================================================================
# TESTS FOR DOWNLOAD WORKER (lines 454-457)
# =============================================================================


class TestDownloadWorker:
    """Test download_worker function."""

    @patch("prepare.download_shard")
    def test_download_worker_success(self, mock_download: MagicMock) -> None:
        """Test download worker success."""
        mock_download.return_value = True

        from prepare import download_worker

        result = download_worker((1, Path("/tmp")))
        assert result == (1, True)  # nosec: B101 - Test assertion

    @patch("prepare.download_shard")
    def test_download_worker_failure(self, mock_download: MagicMock) -> None:
        """Test download worker failure."""
        mock_download.return_value = False

        from prepare import download_worker

        result = download_worker((1, Path("/tmp")))
        assert result == (1, False)  # nosec: B101 - Test assertion


# =============================================================================
# TESTS FOR DOWNLOAD SHARDS PARALLEL (lines 441-451)
# =============================================================================


class TestDownloadShardsParallel:
    """Test download_shards_parallel function."""

    @patch("prepare.Pool")
    def test_download_shards_parallel_success(self, mock_pool: MagicMock) -> None:
        """Test parallel shard download success."""
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        mock_pool_instance.map.return_value = [(1, True), (2, True), (3, True)]

        from prepare import download_shards_parallel

        result = download_shards_parallel([1, 2, 3], Path("/tmp"))
        assert result is True  # nosec: B101 - Test assertion

    @patch("prepare.Pool")
    def test_download_shards_parallel_failure(self, mock_pool: MagicMock) -> None:
        """Test parallel shard download with failure."""
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        mock_pool_instance.map.return_value = [(1, True), (2, False), (3, True)]

        from prepare import download_shards_parallel

        result = download_shards_parallel([1, 2, 3], Path("/tmp"))
        assert result is False  # nosec: B101 - Test assertion


# =============================================================================
# TESTS FOR READ SHARD (lines 460-472)
# =============================================================================


class TestReadShard:
    """Test read_shard function."""

    @patch("pyarrow.parquet.read_table")
    def test_read_shard_success(self, mock_read: MagicMock) -> None:
        """Test successful shard read."""
        mock_df = MagicMock()
        mock_table = MagicMock()
        mock_table.to_pandas.return_value = mock_df
        mock_read.return_value = mock_table

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            shard_file = data_dir / "shard_00001.parquet"
            shard_file.touch()

            from prepare import read_shard

            result = read_shard(1, data_dir)
            assert result is not None  # nosec: B101 - Test assertion

    def test_read_shard_not_exists(self) -> None:
        """Test read when shard doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)

            from prepare import read_shard

            result = read_shard(1, data_dir)
            assert result is None  # nosec: B101 - Test assertion

    @patch("pyarrow.parquet.read_table")
    def test_read_shard_error(self, mock_read: MagicMock) -> None:
        """Test shard read error."""
        mock_read.side_effect = Exception("Read error")

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            shard_file = data_dir / "shard_00001.parquet"
            shard_file.touch()

            from prepare import read_shard

            result = read_shard(1, data_dir)
            assert result is None  # nosec: B101 - Test assertion


# =============================================================================
# TESTS FOR MULTIPROCESSING POOL USAGE (lines 139-143)
# =============================================================================


class TestMultiprocessingPool:
    """Test multiprocessing Pool usage."""

    @patch("prepare.Pool")
    @patch("prepare.Path.exists")
    @patch("os.makedirs")
    def test_download_data_pool_cleanup(
        self, mock_makedirs: MagicMock, mock_exists: MagicMock, mock_pool: MagicMock
    ) -> None:
        """Test that Pool is properly cleaned up."""
        mock_exists.return_value = True  # All files exist
        mock_makedirs.return_value = None

        from prepare import download_data

        with patch("builtins.print"):
            download_data(3, 8)

        # Pool should be used as context manager


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
