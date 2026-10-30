"""
Test suite for prepare.py module.

Tests data preparation functionality including downloading and tokenizer training.
"""

import os

# Add the parent directory to the path to import the module
import sys
from itertools import islice
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from tests._module_isolation import (
    import_with_mocks,
    installed_module,
    mocked_modules,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Heavy dependencies are mocked for ``prepare``. They are installed in
# ``sys.modules`` only while this file's tests run (see the fixture below), so
# they do not leak into other test files.
_MOCKED_DEPENDENCIES = {
    "requests": MagicMock(),
    "pyarrow.parquet": MagicMock(),
    "rustbpe": MagicMock(),
    "tiktoken": MagicMock(),
}

prep = import_with_mocks("prepare", {**_MOCKED_DEPENDENCIES, "torch": MagicMock()})


@pytest.fixture(autouse=True, scope="module")
def _prepare_resolves_to_mocked_import():
    """Make ``patch("prepare.X")`` and ``patch("pyarrow.parquet.X")`` hit the
    mocked module and dependencies that ``prep`` was imported against."""
    with mocked_modules(_MOCKED_DEPENDENCIES), installed_module("prepare", prep):
        yield


class TestConstants:
    """Test module constants."""

    def test_max_seq_len(self):
        """Test MAX_SEQ_LEN constant."""
        assert prep.MAX_SEQ_LEN == 2048  # nosec: B101 - Test assertion

    def test_time_budget(self):
        """Test TIME_BUDGET constant."""
        assert prep.TIME_BUDGET == 600  # nosec: B101 - Test assertion

    def test_eval_tokens(self):
        """Test EVAL_TOKENS constant."""
        assert prep.EVAL_TOKENS == 40 * 524288  # nosec: B101 - Test assertion

    def test_vocab_size(self):
        """Test VOCAB_SIZE constant."""
        assert prep.VOCAB_SIZE == 8192  # nosec: B101 - Test assertion

    def test_cache_dir(self):
        """Test CACHE_DIR constant."""
        expected = Path.home() / ".cache" / "autoresearch"
        assert prep.CACHE_DIR == expected  # nosec: B101 - Test assertion

    def test_data_dir(self):
        """Test DATA_DIR constant."""
        expected = prep.CACHE_DIR / "data"
        assert prep.DATA_DIR == expected  # nosec: B101 - Test assertion

    def test_tokenizer_dir(self):
        """Test TOKENIZER_DIR constant."""
        expected = prep.CACHE_DIR / "tokenizer"
        assert prep.TOKENIZER_DIR == expected  # nosec: B101 - Test assertion

    def test_base_url(self):
        """Test BASE_URL constant."""
        expected = (
            "https://huggingface.co/datasets/karpathy/climbix-400b-shuffle/resolve/main"
        )
        assert prep.BASE_URL == expected  # nosec: B101 - Test assertion

    def test_max_shard(self):
        """Test MAX_SHARD constant."""
        assert prep.MAX_SHARD == 6542  # nosec: B101 - Test assertion

    def test_val_shard(self):
        """Test VAL_SHARD constant."""
        assert prep.VAL_SHARD == 6542  # nosec: B101 - Test assertion

    def test_val_filename(self):
        """Test VAL_FILENAME constant."""
        assert (
            prep.VAL_FILENAME == "shard_06542.parquet"
        )  # nosec: B101 - Test assertion


class TestDownloadFunction:
    """Test download functionality."""

    @patch("prepare.requests.get")
    @patch("prepare.Path.exists")
    @patch("prepare.Path.mkdir")
    def test_download_file_success(self, mock_mkdir, mock_exists, mock_get):
        """Test successful file download."""
        mock_exists.return_value = False
        mock_mkdir.return_value = None

        # Mock response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {"Content-length": "1000"}
        mock_response.iter_content.return_value = [b"data chunk"]
        mock_get.return_value = mock_response

        # Mock file operations
        with patch("builtins.open", mock_open()):
            with patch("tqdm.tqdm") as mock_tqdm:
                mock_tqdm.return_value.__enter__.return_value = mock_tqdm
                mock_tqdm.return_value.__iter__.return_value = iter([b"data chunk"])

                result = prep.download_file(
                    "http://example.com/file", Path("/tmp/file")
                )
                assert result is True  # nosec: B101 - Test assertion

    @patch("prepare.requests.get")
    @patch("prepare.Path.exists")
    def test_download_file_already_exists(self, mock_exists, mock_get):
        """Test download when file already exists."""
        mock_exists.return_value = True

        result = prep.download_file("http://example.com/file", Path("/tmp/file"))
        assert result is True  # nosec: B101 - Test assertion
        mock_get.assert_not_called()

    @patch("prepare.requests.get")
    @patch("prepare.Path.exists")
    @patch("prepare.Path.mkdir")
    def test_download_file_http_error(self, mock_mkdir, mock_exists, mock_get):
        """Test download with HTTP error."""
        mock_exists.return_value = False
        mock_mkdir.return_value = None

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP Error")
        mock_get.return_value = mock_response

        result = prep.download_file("http://example.com/file", Path("/tmp/file"))
        assert result is False  # nosec: B101 - Test assertion

    @patch("prepare.Path.exists")
    @patch("prepare.Path.mkdir")
    def test_download_shard(self, mock_mkdir, mock_exists):
        """Test downloading a shard."""
        mock_exists.return_value = False
        mock_mkdir.return_value = None

        with patch.object(
            prep, "download_single_shard", return_value=True
        ) as mock_download:
            result = prep.download_shard(1, prep.DATA_DIR)
            assert result is True  # nosec: B101 - Test assertion
            mock_download.assert_called_once()

    @patch("prepare.Path.exists")
    @patch("prepare.Path.mkdir")
    def test_download_shard_failure(self, mock_mkdir, mock_exists):
        """Test shard download failure."""
        mock_exists.return_value = False
        mock_mkdir.return_value = None

        with patch.object(
            prep, "download_single_shard", return_value=False
        ) as mock_download:
            result = prep.download_shard(1, prep.DATA_DIR)
            assert result is False  # nosec: B101 - Test assertion
            mock_download.assert_called_once()


class TestTokenizerTraining:
    """Test tokenizer training functionality."""

    @patch("prepare.rustbpe.Trainer")
    @patch("prepare.Path.exists")
    @patch("prepare.Path.mkdir")
    def test_train_tokenizer_success(self, mock_mkdir, mock_exists, mock_trainer):
        """Test successful tokenizer training."""
        mock_exists.return_value = False
        mock_mkdir.return_value = None

        # Mock trainer
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.load.return_value = None
        mock_trainer_instance.train.return_value = None

        # Mock file operations
        with patch("builtins.open", mock_open()):
            result = prep.train_tokenizer(prep.DATA_DIR, prep.TOKENIZER_DIR)
            assert result is True  # nosec: B101 - Test assertion

    @patch("prepare.Path.exists")
    def test_train_tokenizer_already_exists(self, mock_exists):
        """Test training when tokenizer already exists."""
        # Mock tokenizer file exists
        mock_exists.return_value = True

        result = prep.train_tokenizer(prep.DATA_DIR, prep.TOKENIZER_DIR)
        assert result is True  # nosec: B101 - Test assertion

    @patch("prepare.rustbpe.Trainer")
    @patch("prepare.Path.exists")
    @patch("prepare.Path.mkdir")
    def test_train_tokenizer_failure(self, mock_mkdir, mock_exists, mock_trainer):
        """Test tokenizer training failure."""
        mock_exists.return_value = False
        mock_mkdir.return_value = None

        mock_trainer.side_effect = Exception("Training failed")

        result = prep.train_tokenizer(prep.DATA_DIR, prep.TOKENIZER_DIR)
        assert result is False  # nosec: B101 - Test assertion


class TestDataProcessing:
    """Test data processing functionality."""

    @patch("pyarrow.parquet.read_table")
    @patch("prepare.Path.exists")
    def test_read_shard_success(self, mock_exists, mock_read):
        """Test successful shard reading."""
        mock_exists.return_value = True

        # Mock parquet table
        mock_table = MagicMock()
        mock_table.to_pandas.return_value = MagicMock()
        mock_read.return_value = mock_table

        result = prep.read_shard(1, prep.DATA_DIR)
        assert result is not None  # nosec: B101 - Test assertion

    @patch("prepare.Path.exists")
    def test_read_shard_not_exists(self, mock_exists):
        """Test reading non-existent shard."""
        mock_exists.return_value = False

        result = prep.read_shard(1, prep.DATA_DIR)
        assert result is None  # nosec: B101 - Test assertion

    @patch("pyarrow.parquet.read_table")
    @patch("prepare.Path.exists")
    def test_read_shard_error(self, mock_exists, mock_read):
        """Test shard reading with error."""
        mock_exists.return_value = True
        mock_read.side_effect = Exception("Read error")

        result = prep.read_shard(1, prep.DATA_DIR)
        assert result is None  # nosec: B101 - Test assertion

    def test_tokenize_text(self):
        """Test text tokenization."""
        with patch("prepare.rustbpe.Encoder") as mock_encoder:
            mock_encoder_instance = MagicMock()
            mock_encoder_instance.encode.return_value = [1, 2, 3, 4, 5]
            mock_encoder.return_value = mock_encoder_instance

            result = prep.tokenize_text("test text", mock_encoder_instance)
            assert result == [1, 2, 3, 4, 5]  # nosec: B101 - Test assertion

    def test_tokenize_batch(self):
        """Test batch tokenization."""
        with patch("prepare.rustbpe.Encoder") as mock_encoder:
            mock_encoder_instance = MagicMock()
            mock_encoder_instance.encode.return_value = [1, 2, 3]
            mock_encoder.return_value = mock_encoder_instance

            texts = ["text1", "text2", "text3"]
            result = prep.tokenize_batch(texts, mock_encoder_instance)
            assert len(result) == 3  # nosec: B101 - Test assertion
            assert all(
                isinstance(tokens, list) for tokens in result
            )  # nosec: B101 - Test assertion


class TestValidation:
    """Test validation functionality."""

    def test_validate_tokenizer_success(self):
        """Test successful tokenizer validation."""
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "Hello, world!"

        result = prep.validate_tokenizer(mock_tokenizer)
        assert result is True  # nosec: B101 - Test assertion
        mock_tokenizer.encode.assert_called_once()

    def test_validate_tokenizer_failure(self):
        """Test tokenizer validation failure."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = Exception("Validation failed")

        result = prep.validate_tokenizer(mock_tokenizer)
        assert result is False  # nosec: B101 - Test assertion

    def test_validate_data_format(self):
        """Test data format validation."""
        # Mock valid data
        valid_data = [{"text": "Sample text 1"}, {"text": "Sample text 2"}]

        result = prep.validate_data_format(valid_data)
        assert result is True  # nosec: B101 - Test assertion

    def test_validate_data_format_invalid(self):
        """Test invalid data format validation."""
        # Mock invalid data
        invalid_data = [
            {"content": "Sample text 1"},  # Wrong key
            {"text": "Sample text 2"},
        ]

        result = prep.validate_data_format(invalid_data)
        assert result is False  # nosec: B101 - Test assertion


class TestArgumentParsing:
    """Test command line argument parsing."""

    def test_parse_args_defaults(self):
        """Test default argument parsing."""
        args = prep.parse_args([])
        assert args.num_shards is None  # nosec: B101 - Test assertion
        assert args.skip_download is False  # nosec: B101 - Test assertion
        assert args.skip_tokenizer is False  # nosec: B101 - Test assertion
        assert args.force is False  # nosec: B101 - Test assertion

    def test_parse_args_custom_values(self):
        """Test custom argument parsing."""
        args = prep.parse_args(
            ["--num-shards", "10", "--skip-download", "--skip-tokenizer", "--force"]
        )
        assert args.num_shards == 10  # nosec: B101 - Test assertion
        assert args.skip_download is True  # nosec: B101 - Test assertion
        assert args.skip_tokenizer is True  # nosec: B101 - Test assertion
        assert args.force is True  # nosec: B101 - Test assertion

    def test_parse_args_num_shards_validation(self):
        """Test num_shards validation."""
        # Valid num_shards
        args = prep.parse_args(["--num-shards", "100"])
        assert args.num_shards == 100  # nosec: B101 - Test assertion

        # Invalid num_shards (should still parse, validation happens later)
        args = prep.parse_args(["--num-shards", "10000"])
        assert args.num_shards == 10000  # nosec: B101 - Test assertion


class TestMainFunction:
    """Test main function execution."""

    @patch("prepare.Path.exists")
    @patch("prepare.Path.mkdir")
    def test_main_full_prep_success(self, mock_mkdir, mock_exists):
        """Test successful full preparation."""
        mock_exists.return_value = False
        mock_mkdir.return_value = None

        with patch.object(prep, "download_data") as mock_download:
            with patch.object(prep, "train_tokenizer", return_value=True) as mock_train:
                mock_download.return_value = None
                with patch.object(
                    prep, "list_parquet_files", return_value=[Path("test.parquet")]
                ):
                    result = prep.main([])
                    assert result == 0  # nosec: B101 - Test assertion
                    mock_download.assert_called_once()
                    mock_train.assert_called_once()

    @patch("prepare.Path.exists")
    @patch("prepare.Path.mkdir")
    def test_main_download_only(self, mock_mkdir, mock_exists):
        """Test download only mode."""
        mock_exists.return_value = False
        mock_mkdir.return_value = None

        with patch.object(prep, "download_data") as mock_download:
            with patch.object(prep, "train_tokenizer", return_value=True) as mock_train:
                mock_download.return_value = None
                with patch.object(
                    prep, "list_parquet_files", return_value=[Path("test.parquet")]
                ):
                    result = prep.main(["--skip-tokenizer"])
                    assert result == 0  # nosec: B101 - Test assertion
                    mock_download.assert_called_once()
                    mock_train.assert_not_called()

    @patch("prepare.Path.exists")
    @patch("prepare.Path.mkdir")
    def test_main_tokenizer_only(self, mock_mkdir, mock_exists):
        """Test tokenizer only mode."""
        mock_exists.return_value = False
        mock_mkdir.return_value = None

        with patch.object(prep, "download_data") as mock_download:
            with patch.object(prep, "train_tokenizer", return_value=True) as mock_train:
                mock_download.return_value = None
                with patch.object(
                    prep, "list_parquet_files", return_value=[Path("test.parquet")]
                ):
                    result = prep.main(["--skip-download"])
                    assert result == 0  # nosec: B101 - Test assertion
                    mock_download.assert_not_called()
                    mock_train.assert_called_once()

    @patch("prepare.Path.exists")
    @patch("prepare.Path.mkdir")
    def test_main_download_failure(self, mock_mkdir, mock_exists):
        """Test main function with download failure."""
        mock_exists.return_value = False
        mock_mkdir.return_value = None

        with patch.object(prep, "download_data") as mock_download:
            with patch.object(prep, "list_parquet_files", return_value=[]):
                mock_download.return_value = None
                result = prep.main([])
                assert result == 1  # nosec: B101 - Test assertion
                mock_download.assert_called_once()

    @patch("prepare.Path.exists")
    @patch("prepare.Path.mkdir")
    def test_main_tokenizer_failure(self, mock_mkdir, mock_exists):
        """Test main function with tokenizer failure."""
        mock_exists.return_value = False
        mock_mkdir.return_value = None

        with patch.object(prep, "download_data") as mock_download:
            with patch.object(
                prep, "train_tokenizer", return_value=False
            ) as mock_train:
                mock_download.return_value = None
                with patch.object(
                    prep, "list_parquet_files", return_value=[Path("test.parquet")]
                ):
                    result = prep.main([])
                    assert result == 1  # nosec: B101 - Test assertion
                    mock_download.assert_called_once()
                    mock_train.assert_called_once()

    def test_main_custom_shards(self):
        """Test main function with custom number of shards."""
        with patch.object(prep, "download_data") as mock_download:
            with patch.object(prep, "train_tokenizer", return_value=True):
                with patch.object(
                    prep, "list_parquet_files", return_value=[Path("test.parquet")]
                ):
                    result = prep.main(["--num-shards", "5"])
                    assert result == 0  # nosec: B101 - Test assertion
                    mock_download.assert_called_once_with(5, 8)


class TestMultiprocessing:
    """Test multiprocessing functionality."""

    def test_download_worker(self):
        """Test download worker function."""
        with patch.object(prep, "download_shard", return_value=True) as mock_download:
            result = prep.download_worker((1, prep.DATA_DIR))
            assert result == (1, True)  # nosec: B101 - Test assertion
            mock_download.assert_called_once_with(1, prep.DATA_DIR)

    def test_download_worker_failure(self):
        """Test download worker with failure."""
        with patch.object(prep, "download_shard", return_value=False) as mock_download:
            result = prep.download_worker((1, prep.DATA_DIR))
            assert result == (1, False)  # nosec: B101 - Test assertion
            mock_download.assert_called_once_with(1, prep.DATA_DIR)

    @patch("prepare.Pool")
    def test_download_shards_parallel(self, mock_pool):
        """Test parallel shard downloading."""
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        mock_pool_instance.map.return_value = [(1, True), (2, True), (3, True)]

        result = prep.download_shards_parallel([1, 2, 3], prep.DATA_DIR)
        assert result is True  # nosec: B101 - Test assertion
        mock_pool_instance.map.assert_called_once()

    @patch("prepare.Pool")
    def test_download_shards_parallel_failure(self, mock_pool):
        """Test parallel shard downloading with failure."""
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        mock_pool_instance.map.return_value = [(1, True), (2, False), (3, True)]

        result = prep.download_shards_parallel([1, 2, 3], prep.DATA_DIR)
        assert result is False  # nosec: B101 - Test assertion


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_cache_info(self):
        """Test cache information retrieval."""
        with patch("prepare.Path.exists") as mock_exists:
            with patch("prepare.Path.stat") as mock_stat:
                mock_exists.return_value = True
                mock_stat.return_value.st_size = 1000000
                mock_stat.return_value.st_mtime = 1234567890

                info = prep.get_cache_info()
                assert isinstance(info, dict)  # nosec: B101 - Test assertion
                assert "size_bytes" in info  # nosec: B101 - Test assertion
                assert "last_modified" in info  # nosec: B101 - Test assertion

    def test_get_cache_info_not_exists(self):
        """Test cache info when cache doesn't exist."""
        with patch("prepare.Path.exists") as mock_exists:
            mock_exists.return_value = False

            info = prep.get_cache_info()
            assert isinstance(info, dict)  # nosec: B101 - Test assertion
            assert info["size_bytes"] == 0  # nosec: B101 - Test assertion

    def test_cleanup_cache(self):
        """Test cache cleanup."""
        with patch("prepare.shutil.rmtree") as mock_rmtree:
            with patch("prepare.Path.exists") as mock_exists:
                mock_exists.return_value = True

                result = prep.cleanup_cache()
                assert result is True  # nosec: B101 - Test assertion
                mock_rmtree.assert_called()

    def test_cleanup_cache_not_exists(self):
        """Test cache cleanup when cache doesn't exist."""
        with patch("prepare.Path.exists") as mock_exists:
            mock_exists.return_value = False

            result = prep.cleanup_cache()
            assert result is True  # nosec: B101 - Test assertion


class TestDownloadSingleShard:
    """Test download_single_shard function."""

    @patch("prepare.requests.get")
    @patch("prepare.Path.exists")
    @patch("prepare.Path.mkdir")
    @patch("tempfile.NamedTemporaryFile")
    def test_download_single_shard_success(
        self, mock_temp, mock_mkdir, mock_exists, mock_get
    ):
        """Test successful single shard download."""
        mock_exists.return_value = False
        mock_mkdir.return_value = None

        # Mock response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content.return_value = [b"data"]
        mock_get.return_value = mock_response

        # Mock temp file
        mock_temp.return_value.__enter__.return_value.name = "temp.tmp"
        mock_temp.return_value.__enter__.return_value.write.return_value = None
        mock_temp.return_value.__exit__.return_value = None

        with patch("os.replace"):
            with patch("builtins.print"):
                result = prep.download_single_shard(1)
                assert result is True  # nosec: B101 - Test assertion

    @patch("prepare.Path.exists")
    def test_download_single_shard_already_exists(self, mock_exists):
        """Test download when shard already exists."""
        mock_exists.return_value = True
        result = prep.download_single_shard(1)
        assert result is True  # nosec: B101 - Test assertion

    @patch("prepare.requests.get")
    @patch("prepare.Path.exists")
    @patch("prepare.Path.mkdir")
    @patch("tempfile.NamedTemporaryFile")
    def test_download_single_shard_retry(
        self, mock_temp, mock_mkdir, mock_exists, mock_get
    ):
        """Test download with retry on failure."""
        mock_exists.return_value = False
        mock_mkdir.return_value = None

        # First attempt fails, second succeeds
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = [Exception("Error"), None]
        mock_response.iter_content.return_value = [b"data"]
        mock_get.return_value = mock_response

        mock_temp.return_value.__enter__.return_value.name = "temp.tmp"
        mock_temp.return_value.__enter__.return_value.write.return_value = None
        mock_temp.return_value.__exit__.return_value = None

        with patch("os.replace"):
            with patch("os.remove"):
                with patch("builtins.print"):
                    with patch("time.sleep"):
                        result = prep.download_single_shard(1)
                        assert result is True  # nosec: B101 - Test assertion

    @patch("prepare.requests.get")
    @patch("prepare.Path.exists")
    @patch("prepare.Path.mkdir")
    def test_download_single_shard_all_fail(self, mock_mkdir, mock_exists, mock_get):
        """Test download when all attempts fail."""
        mock_exists.return_value = False
        mock_mkdir.return_value = None

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("Error")
        mock_get.return_value = mock_response

        with patch("builtins.print"), patch.object(prep.time, "sleep") as mock_sleep:
            result = prep.download_single_shard(1)
            assert result is False  # nosec: B101 - Test assertion
            # Every attempt but the last backs off before retrying
            assert mock_sleep.call_count == 4  # nosec: B101 - Test assertion


class TestDownloadData:
    """Test download_data function."""

    @patch("prepare.Pool")
    @patch("prepare.Path.exists")
    @patch("prepare.Path.mkdir")
    def test_download_data_all_exist(self, mock_mkdir, mock_exists, mock_pool):
        """Test download when all shards already exist."""
        mock_exists.return_value = True
        mock_mkdir.return_value = None

        with patch("builtins.print"):
            prep.download_data(5, 8)
            # Should not call pool if all exist

    @patch("prepare.Pool")
    @patch("prepare.Path.exists", autospec=True)
    @patch("prepare.Path.mkdir")
    def test_download_data_partial(self, mock_mkdir, mock_exists, mock_pool):
        """Test download with some shards already existing."""

        # Mock that shard 0 exists, but shard 1 doesn't
        def exists_side_effect(path):
            return "shard_00000" in str(path)

        mock_exists.side_effect = exists_side_effect
        mock_mkdir.return_value = None

        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        mock_pool_instance.map.return_value = [True, True]

        with patch("builtins.print"):
            prep.download_data(2, 8)


class TestListParquetFiles:
    """Test list_parquet_files function."""

    @patch("prepare.DATA_DIR")
    def test_list_parquet_files(self, mock_data_dir):
        """Test listing parquet files."""
        # Mock directory with parquet files
        mock_data_dir.iterdir.return_value = [
            Path("shard_00001.parquet"),
            Path("shard_00002.parquet"),
            Path("shard_00002.parquet.tmp"),  # Should be excluded
            Path("readme.txt"),  # Should be excluded
        ]

        files = prep.list_parquet_files()
        assert len(files) == 2  # nosec: B101 - Test assertion
        assert all(
            f.suffix == ".parquet" for f in files
        )  # nosec: B101 - Test assertion
        assert not any(
            f.name.endswith(".tmp") for f in files
        )  # nosec: B101 - Test assertion

    @patch("prepare.DATA_DIR")
    def test_list_parquet_files_empty(self, mock_data_dir):
        """Test listing when no parquet files."""
        mock_data_dir.iterdir.return_value = []
        files = prep.list_parquet_files()
        assert files == []  # nosec: B101 - Test assertion


class TestTextIterator:
    """Test text_iterator function."""

    @patch("prepare.list_parquet_files")
    def test_text_iterator(self, mock_list):
        """Test text iterator."""
        # Mock parquet files
        mock_list.return_value = [Path("shard_00001.parquet")]

        # Mock parquet file
        mock_pf = MagicMock()
        mock_pf.num_row_groups = 1
        mock_rg = MagicMock()
        mock_rg.column.return_value.to_pylist.return_value = ["text1", "text2"]
        mock_pf.read_row_group.return_value = mock_rg

        with patch("pyarrow.parquet.ParquetFile", return_value=mock_pf):
            texts = list(prep.text_iterator(max_chars=100))
            assert len(texts) > 0  # nosec: B101 - Test assertion


class TestDocumentBatches:
    """Test _document_batches function."""

    @patch("prepare.list_parquet_files")
    def test_document_batches_train(self, mock_list):
        """Test document batches for train split."""
        mock_list.return_value = [Path("shard_00001.parquet")]

        # Mock parquet file
        mock_pf = MagicMock()
        mock_pf.num_row_groups = 1
        mock_rg = MagicMock()
        mock_rg.column.return_value.to_pylist.return_value = ["text1", "text2"]
        mock_pf.read_row_group.return_value = mock_rg

        with patch("pyarrow.parquet.ParquetFile", return_value=mock_pf):
            # The generator is infinite by design; take a few batches
            batches = list(
                islice(prep._document_batches("train", tokenizer_batch_size=2), 3)
            )
            assert len(batches) > 0  # nosec: B101 - Test assertion

    @patch("prepare.list_parquet_files")
    def test_document_batches_val(self, mock_list):
        """Test document batches for val split."""
        mock_list.return_value = [Path("shard_00001.parquet")]

        # Mock parquet file
        mock_pf = MagicMock()
        mock_pf.num_row_groups = 1
        mock_rg = MagicMock()
        mock_rg.column.return_value.to_pylist.return_value = ["text1", "text2"]
        mock_pf.read_row_group.return_value = mock_rg

        with patch("pyarrow.parquet.ParquetFile", return_value=mock_pf):
            # The generator is infinite by design; take a few batches
            batches = list(
                islice(prep._document_batches("val", tokenizer_batch_size=2), 3)
            )
            assert len(batches) > 0  # nosec: B101 - Test assertion


class TestGetTokenBytes:
    """Test get_token_bytes function."""

    @patch("builtins.open", new_callable=mock_open)
    @patch("prepare.TOKENIZER_DIR")
    def test_get_token_bytes(self, mock_token_dir, mock_file):
        """Test getting token bytes."""
        mock_token_dir.__truediv__.return_value = Path("token_bytes.pt")

        with patch.object(prep.torch, "load", return_value=MagicMock()) as mock_load:
            result = prep.get_token_bytes("cpu")

        assert result is mock_load.return_value  # nosec: B101 - Test assertion
        mock_file.assert_called_once_with(Path("token_bytes.pt"), "rb")
        mock_load.assert_called_once()
        assert mock_load.call_args.kwargs["map_location"] == "cpu"  # nosec: B101


class TestTokenizerClass:
    """Test Tokenizer class."""

    @patch("builtins.open", new_callable=mock_open)
    @patch("pickle.load")
    @patch("prepare.TOKENIZER_DIR")
    def test_tokenizer_from_directory(self, mock_token_dir, mock_pickle, mock_file):
        """Test Tokenizer.from_directory class method."""
        mock_token_dir.__truediv__.return_value = Path("tokenizer.pkl")
        mock_enc = MagicMock()
        mock_enc.n_vocab = 8192
        mock_enc.encode_single_token.return_value = 0
        mock_pickle.return_value = mock_enc

        tokenizer = prep.Tokenizer.from_directory()
        mock_file.assert_called_once_with(Path("tokenizer.pkl"), "rb")
        assert tokenizer.get_vocab_size() == 8192  # nosec: B101 - Test assertion
        assert tokenizer.get_bos_token_id() == 0  # nosec: B101 - Test assertion

    def test_tokenizer_encode_string(self):
        """Test Tokenizer.encode with string."""
        mock_enc = MagicMock()
        mock_enc.encode_ordinary.return_value = [1, 2, 3]
        mock_enc.encode_single_token.return_value = 0
        tokenizer = prep.Tokenizer(mock_enc)

        result = tokenizer.encode("test")
        assert result == [1, 2, 3]  # nosec: B101 - Test assertion

    def test_tokenizer_encode_with_prepend(self):
        """Test Tokenizer.encode with prepend."""
        mock_enc = MagicMock()
        mock_enc.encode_ordinary.return_value = [1, 2, 3]
        mock_enc.encode_single_token.return_value = 0
        tokenizer = prep.Tokenizer(mock_enc)

        result = tokenizer.encode("test", prepend="bos")
        assert result == [0, 1, 2, 3]  # nosec: B101 - Test assertion

    def test_tokenizer_encode_list(self):
        """Test Tokenizer.encode with list."""
        mock_enc = MagicMock()
        mock_enc.encode_ordinary_batch.return_value = [[1, 2], [3, 4]]
        mock_enc.encode_single_token.return_value = 0
        tokenizer = prep.Tokenizer(mock_enc)

        result = tokenizer.encode(["text1", "text2"])
        assert len(result) == 2  # nosec: B101 - Test assertion

    def test_tokenizer_encode_invalid_type(self):
        """Test Tokenizer.encode with invalid type."""
        mock_enc = MagicMock()
        tokenizer = prep.Tokenizer(mock_enc)

        with pytest.raises(ValueError):
            tokenizer.encode(123)

    def test_tokenizer_decode(self):
        """Test Tokenizer.decode."""
        mock_enc = MagicMock()
        mock_enc.decode.return_value = "decoded text"
        tokenizer = prep.Tokenizer(mock_enc)

        result = tokenizer.decode([1, 2, 3])
        assert result == "decoded text"  # nosec: B101 - Test assertion


class TestEvaluateBpb:
    """Test evaluate_bpb function."""

    def test_evaluate_bpb(self):
        """BPB is nats-per-byte over non-special tokens, converted to bits."""
        import math

        import torch

        # evaluate_bpb is decorated with @torch.no_grad() at import time, so it
        # must be imported against the real torch (the module-level ``prep`` was
        # imported against a mock, which turns the function into a mock).
        real_prep = import_with_mocks("prepare", {})

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


class TestMakeDataloader:
    """Test make_dataloader function."""

    @patch("prepare._document_batches")
    @patch("torch.cuda.is_available")
    def test_make_dataloader_cpu(self, mock_cuda, mock_batches):
        """Test dataloader on CPU."""
        mock_cuda.return_value = False

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.get_bos_token_id.return_value = 0
        mock_tokenizer.encode.return_value = [[1, 2], [3, 4]]

        # Mock batches
        mock_batches.return_value = iter([(["text1", "text2"], 1)])

        with patch("torch.empty"):
            result = prep.make_dataloader(mock_tokenizer, 2, 10, "train")
            # Should return an iterator
            assert result is not None  # nosec: B101 - Test assertion


if __name__ == "__main__":
    pytest.main([__file__])
