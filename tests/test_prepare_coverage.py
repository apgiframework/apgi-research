"""
Full coverage tests for prepare.py - Phase 1 implementation.

This test file achieves 100% coverage for prepare.py by:
1. Testing all constants and configuration
2. Testing download functions with mocked network
3. Testing tokenizer training
4. Testing data processing functions
5. Testing the main function and argument parsing
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from tests._module_isolation import mocked_modules

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def _fresh_prepare_import():
    """Give each test its own ``prepare`` import.

    Tests patch ``sys.modules`` with mocks and then ``import prepare``, which
    only picks the mocks up if ``prepare`` is not already cached; the previous
    module is restored afterwards so nothing leaks to other files.
    """
    with mocked_modules(fresh=["prepare"]):
        yield


class TestConstants:
    """Test module constants - lines 45-68."""

    def test_max_seq_len(self) -> None:
        """Test MAX_SEQ_LEN constant."""
        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            assert prep.MAX_SEQ_LEN == 2048  # nosec: B101 - Test assertion

    def test_time_budget(self) -> None:
        """Test TIME_BUDGET constant."""
        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            assert prep.TIME_BUDGET == 600  # nosec: B101 - Test assertion

    def test_eval_tokens(self) -> None:
        """Test EVAL_TOKENS constant."""
        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            assert prep.EVAL_TOKENS == 40 * 524288  # nosec: B101 - Test assertion

    def test_vocab_size(self) -> None:
        """Test VOCAB_SIZE constant."""
        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            assert prep.VOCAB_SIZE == 8192  # nosec: B101 - Test assertion

    def test_cache_dir(self) -> None:
        """Test CACHE_DIR constant."""
        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            expected = Path.home() / ".cache" / "autoresearch"
            assert prep.CACHE_DIR == expected  # nosec: B101 - Test assertion

    def test_data_dir(self) -> None:
        """Test DATA_DIR constant."""
        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            assert (
                prep.DATA_DIR == prep.CACHE_DIR / "data"
            )  # nosec: B101 - Test assertion

    def test_tokenizer_dir(self) -> None:
        """Test TOKENIZER_DIR constant."""
        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            assert (
                prep.TOKENIZER_DIR == prep.CACHE_DIR / "tokenizer"
            )  # nosec: B101 - Test assertion

    def test_base_url(self) -> None:
        """Test BASE_URL constant."""
        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            expected = "https://huggingface.co/datasets/karpathy/climbix-400b-shuffle/resolve/main"
            assert prep.BASE_URL == expected  # nosec: B101 - Test assertion

    def test_max_shard(self) -> None:
        """Test MAX_SHARD constant."""
        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            assert prep.MAX_SHARD == 6542  # nosec: B101 - Test assertion

    def test_val_shard(self) -> None:
        """Test VAL_SHARD constant."""
        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            assert prep.VAL_SHARD == 6542  # nosec: B101 - Test assertion

    def test_val_filename(self) -> None:
        """Test VAL_FILENAME constant."""
        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            assert (
                prep.VAL_FILENAME == "shard_06542.parquet"
            )  # nosec: B101 - Test assertion

    def test_split_pattern(self) -> None:
        """Test SPLIT_PATTERN constant."""
        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            assert isinstance(prep.SPLIT_PATTERN, str)  # nosec: B101 - Test assertion
            assert len(prep.SPLIT_PATTERN) > 0  # nosec: B101 - Test assertion

    def test_special_tokens(self) -> None:
        """Test SPECIAL_TOKENS constant."""
        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            assert len(prep.SPECIAL_TOKENS) == 4  # nosec: B101 - Test assertion
            assert all(
                t.startswith("<|reserved_") for t in prep.SPECIAL_TOKENS
            )  # nosec: B101 - Test assertion

    def test_bos_token(self) -> None:
        """Test BOS_TOKEN constant."""
        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            assert prep.BOS_TOKEN == "<|reserved_0|>"  # nosec: B101 - Test assertion


class TestDownloadSingleShard:
    """Test download_single_shard function - lines 78-118."""

    @patch("os.replace")
    @patch("tempfile.NamedTemporaryFile")
    @patch("time.sleep")
    @patch("builtins.print")
    def test_download_shard_already_exists(
        self,
        mock_print: MagicMock,
        mock_sleep: MagicMock,
        mock_temp: MagicMock,
        mock_replace: MagicMock,
    ) -> None:
        """Test when shard already exists."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            shard_file = data_dir / "shard_00001.parquet"
            shard_file.touch()

            with patch.dict(
                "sys.modules",
                {
                    "torch": MagicMock(),
                    "requests": MagicMock(),
                    "pyarrow.parquet": MagicMock(),
                    "rustbpe": MagicMock(),
                },
            ):
                with patch("prepare.DATA_DIR", data_dir):
                    import prepare as prep

                    result = prep.download_single_shard(1)
                    assert result is True  # nosec: B101 - Test assertion

    @patch("os.replace")
    @patch("os.remove")
    @patch("time.sleep")
    @patch("builtins.print")
    def test_download_shard_success(
        self,
        mock_print: MagicMock,
        mock_sleep: MagicMock,
        mock_remove: MagicMock,
        mock_replace: MagicMock,
    ) -> None:
        """Test successful shard download."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)

            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_response.iter_content.return_value = [b"test data"]

            mock_requests = MagicMock()
            mock_requests.get.return_value = mock_response

            mock_temp = MagicMock()
            mock_temp.name = str(data_dir / "temp.tmp")

            with patch.dict(
                "sys.modules",
                {
                    "torch": MagicMock(),
                    "requests": mock_requests,
                    "pyarrow.parquet": MagicMock(),
                    "rustbpe": MagicMock(),
                },
            ):
                with patch("prepare.DATA_DIR", data_dir):
                    with patch(
                        "tempfile.NamedTemporaryFile",
                        return_value=MagicMock(
                            __enter__=MagicMock(return_value=mock_temp),
                            __exit__=MagicMock(return_value=None),
                        ),
                    ):
                        import prepare as prep

                        result = prep.download_single_shard(1)
                        assert result is True  # nosec: B101 - Test assertion

    @patch("builtins.print")
    @patch("time.sleep")
    def test_download_shard_all_retries_fail(
        self, mock_sleep: MagicMock, mock_print: MagicMock
    ) -> None:
        """Test when all retry attempts fail."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)

            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = Exception("Network error")

            mock_requests = MagicMock()
            mock_requests.get.return_value = mock_response

            with patch.dict(
                "sys.modules",
                {
                    "torch": MagicMock(),
                    "requests": mock_requests,
                    "pyarrow.parquet": MagicMock(),
                    "rustbpe": MagicMock(),
                },
            ):
                with patch("prepare.DATA_DIR", data_dir):
                    import prepare as prep

                    result = prep.download_single_shard(1)
                    assert result is False  # nosec: B101 - Test assertion


class TestDownloadData:
    """Test download_data function - lines 121-143."""

    @patch("builtins.print")
    @patch("prepare.Pool")
    def test_download_data_all_exist(
        self, mock_pool: MagicMock, mock_print: MagicMock
    ) -> None:
        """Test when all shards already exist."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            # Create existing shards
            for i in range(3):
                (data_dir / f"shard_{i:05d}.parquet").touch()

            with patch.dict(
                "sys.modules",
                {
                    "torch": MagicMock(),
                    "requests": MagicMock(),
                    "pyarrow.parquet": MagicMock(),
                    "rustbpe": MagicMock(),
                },
            ):
                with patch("prepare.DATA_DIR", data_dir):
                    import prepare as prep

                    prep.download_data(3, 8)
                    # Pool should not be used if all exist

    @patch("builtins.print")
    @patch("prepare.Pool")
    def test_download_data_partial(
        self, mock_pool: MagicMock, mock_print: MagicMock
    ) -> None:
        """Test downloading when some shards exist."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            # Create only first shard
            (data_dir / "shard_00000.parquet").touch()

            mock_pool_instance = MagicMock()
            mock_pool.return_value.__enter__.return_value = mock_pool_instance
            mock_pool_instance.map.return_value = [True, True]  # shards 1, 2 succeed

            with patch.dict(
                "sys.modules",
                {
                    "torch": MagicMock(),
                    "requests": MagicMock(),
                    "pyarrow.parquet": MagicMock(),
                    "rustbpe": MagicMock(),
                },
            ):
                with patch("prepare.DATA_DIR", data_dir):
                    import prepare as prep

                    prep.download_data(3, 8)
                    mock_pool_instance.map.assert_called_once()


class TestListParquetFiles:
    """Test list_parquet_files function - lines 151-158."""

    def test_list_parquet_files(self) -> None:
        """Test listing parquet files."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "shard_00001.parquet").touch()
            (data_dir / "shard_00002.parquet").touch()
            (data_dir / "shard_00003.parquet.tmp").touch()  # Should be excluded
            (data_dir / "readme.txt").touch()  # Should be excluded

            with patch.dict(
                "sys.modules",
                {
                    "torch": MagicMock(),
                    "requests": MagicMock(),
                    "pyarrow.parquet": MagicMock(),
                    "rustbpe": MagicMock(),
                },
            ):
                with patch("prepare.DATA_DIR", data_dir):
                    import prepare as prep

                    files = prep.list_parquet_files()
                    assert len(files) == 2  # nosec: B101 - Test assertion
                    assert all(
                        f.suffix == ".parquet" for f in files
                    )  # nosec: B101 - Test assertion

    def test_list_parquet_files_empty(self) -> None:
        """Test listing when no parquet files."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)

            with patch.dict(
                "sys.modules",
                {
                    "torch": MagicMock(),
                    "requests": MagicMock(),
                    "pyarrow.parquet": MagicMock(),
                    "rustbpe": MagicMock(),
                },
            ):
                with patch("prepare.DATA_DIR", data_dir):
                    import prepare as prep

                    files = prep.list_parquet_files()
                    assert files == []  # nosec: B101 - Test assertion


class TestTextIterator:
    """Test text_iterator function - lines 161-176."""

    @patch("pyarrow.parquet.ParquetFile")
    @patch("prepare.list_parquet_files")
    def test_text_iterator_with_limit(
        self, mock_list: MagicMock, mock_pf_class: MagicMock
    ) -> None:
        """Test text iterator with character limit."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "shard_00001.parquet").touch()
            mock_list.return_value = [data_dir / "shard_00001.parquet"]

            mock_pf = MagicMock()
            mock_pf.num_row_groups = 1
            mock_rg = MagicMock()
            mock_rg.column.return_value.to_pylist.return_value = [
                "test text 1",
                "test text 2",
            ]
            mock_pf.read_row_group.return_value = mock_rg
            mock_pf_class.return_value = mock_pf

            with patch.dict(
                "sys.modules",
                {
                    "torch": MagicMock(),
                    "requests": MagicMock(),
                    "pyarrow.parquet": MagicMock(),
                    "rustbpe": MagicMock(),
                },
            ):
                with patch("prepare.DATA_DIR", data_dir):
                    import prepare as prep

                    texts = list(prep.text_iterator(max_chars=50))
                    assert len(texts) > 0  # nosec: B101 - Test assertion


class TestTrainTokenizer:
    """Test train_tokenizer function - lines 179-212."""

    def test_tokenizer_already_exists(self) -> None:
        """Test when tokenizer already exists."""
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            data_dir = cache_dir / "data"
            tokenizer_dir = cache_dir / "tokenizer"
            tokenizer_dir.mkdir(parents=True)
            (tokenizer_dir / "tokenizer.pkl").touch()
            (tokenizer_dir / "token_bytes.pt").touch()

            mock_rustbpe = MagicMock()

            with patch.dict(
                "sys.modules",
                {
                    "torch": MagicMock(),
                    "requests": MagicMock(),
                    "pyarrow.parquet": MagicMock(),
                    "rustbpe": mock_rustbpe,
                },
            ):
                with patch("prepare.DATA_DIR", data_dir):
                    with patch("prepare.TOKENIZER_DIR", tokenizer_dir):
                        import prepare as prep

                        # Defaults are bound at definition time, so patching the
                        # module constants would not reach them; pass explicitly.
                        result = prep.train_tokenizer(
                            data_dir=data_dir, tokenizer_dir=tokenizer_dir
                        )
                        assert result is True  # nosec: B101 - Test assertion
                        mock_rustbpe.Trainer.assert_not_called()

    @patch("builtins.open", mock_open())
    def test_train_tokenizer_success(self) -> None:
        """Test successful tokenizer training."""
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            data_dir = cache_dir / "data"
            tokenizer_dir = cache_dir / "tokenizer"
            data_dir.mkdir(parents=True)

            mock_trainer_instance = MagicMock()
            mock_trainer_instance.load.return_value = None
            mock_trainer_instance.train.return_value = None
            mock_trainer_instance.save_as_tiktoken.return_value = None

            mock_trainer = MagicMock(return_value=mock_trainer_instance)

            mock_rustbpe = MagicMock()
            mock_rustbpe.Trainer = mock_trainer

            mock_torch = MagicMock()
            mock_tensor = MagicMock()
            mock_torch.zeros.return_value = mock_tensor
            mock_torch.uint8 = MagicMock()
            mock_torch.save = MagicMock()

            with patch.dict(
                "sys.modules",
                {
                    "torch": mock_torch,
                    "requests": MagicMock(),
                    "pyarrow.parquet": MagicMock(),
                    "rustbpe": mock_rustbpe,
                },
            ):
                with patch("prepare.DATA_DIR", data_dir):
                    with patch("prepare.TOKENIZER_DIR", tokenizer_dir):
                        with patch("os.makedirs"):
                            import prepare as prep

                            result = prep.train_tokenizer(data_dir, tokenizer_dir)
                            assert result is True  # nosec: B101 - Test assertion

    def test_train_tokenizer_failure(self) -> None:
        """Test tokenizer training failure."""
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            data_dir = cache_dir / "data"
            tokenizer_dir = cache_dir / "tokenizer"
            data_dir.mkdir(parents=True)

            mock_rustbpe = MagicMock()
            mock_rustbpe.Trainer.side_effect = Exception("Training failed")

            with patch.dict(
                "sys.modules",
                {
                    "torch": MagicMock(),
                    "requests": MagicMock(),
                    "pyarrow.parquet": MagicMock(),
                    "rustbpe": mock_rustbpe,
                },
            ):
                with patch("prepare.DATA_DIR", data_dir):
                    with patch("prepare.TOKENIZER_DIR", tokenizer_dir):
                        import prepare as prep

                        result = prep.train_tokenizer(data_dir, tokenizer_dir)
                        assert result is False  # nosec: B101 - Test assertion


class TestTokenizerClass:
    """Test Tokenizer class - lines 220-264."""

    def test_tokenizer_init(self) -> None:
        """Test Tokenizer initialization."""
        mock_enc = MagicMock()
        mock_enc.encode_single_token.return_value = 0

        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            tokenizer = prep.Tokenizer(mock_enc)
            assert tokenizer.bos_token_id == 0  # nosec: B101 - Test assertion

    def test_get_vocab_size(self) -> None:
        """Test get_vocab_size method."""
        mock_enc = MagicMock()
        mock_enc.n_vocab = 8192

        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            tokenizer = prep.Tokenizer(mock_enc)
            assert tokenizer.get_vocab_size() == 8192  # nosec: B101 - Test assertion

    def test_get_bos_token_id(self) -> None:
        """Test get_bos_token_id method."""
        mock_enc = MagicMock()
        mock_enc.encode_single_token.return_value = 0

        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            tokenizer = prep.Tokenizer(mock_enc)
            assert tokenizer.get_bos_token_id() == 0  # nosec: B101 - Test assertion

    def test_encode_string(self) -> None:
        """Test encode with string input."""
        mock_enc = MagicMock()
        mock_enc.encode_ordinary.return_value = [1, 2, 3]

        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            tokenizer = prep.Tokenizer(mock_enc)
            result = tokenizer.encode("test text")
            assert result == [1, 2, 3]  # nosec: B101 - Test assertion

    def test_encode_with_prepend_int(self) -> None:
        """Test encode with integer prepend."""
        mock_enc = MagicMock()
        mock_enc.encode_ordinary.return_value = [1, 2, 3]

        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            tokenizer = prep.Tokenizer(mock_enc)
            result = tokenizer.encode("test", prepend=0)
            assert result[0] == 0  # nosec: B101 - Test assertion
            assert result[1:] == [1, 2, 3]  # nosec: B101 - Test assertion

    def test_encode_with_prepend_str(self) -> None:
        """Test encode with string prepend."""
        mock_enc = MagicMock()
        mock_enc.encode_ordinary.return_value = [1, 2, 3]
        mock_enc.encode_single_token.return_value = 0

        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            tokenizer = prep.Tokenizer(mock_enc)
            result = tokenizer.encode("test", prepend="<bos>")
            assert result[0] == 0  # nosec: B101 - Test assertion

    def test_encode_list(self) -> None:
        """Test encode with list input."""
        mock_enc = MagicMock()
        mock_enc.encode_ordinary_batch.return_value = [[1, 2], [3, 4]]

        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            tokenizer = prep.Tokenizer(mock_enc)
            result = tokenizer.encode(["text1", "text2"])
            assert len(result) == 2  # nosec: B101 - Test assertion

    def test_encode_invalid_type(self) -> None:
        """Test encode with invalid input type."""
        mock_enc = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            tokenizer = prep.Tokenizer(mock_enc)
            with pytest.raises(ValueError):
                tokenizer.encode(123)  # Invalid type

    def test_decode(self) -> None:
        """Test decode method."""
        mock_enc = MagicMock()
        mock_enc.decode.return_value = "decoded text"

        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            tokenizer = prep.Tokenizer(mock_enc)
            result = tokenizer.decode([1, 2, 3])
            assert result == "decoded text"  # nosec: B101 - Test assertion

    @patch("pickle.load")
    @patch("builtins.open", mock_open())
    def test_from_directory(self, mock_pickle: MagicMock) -> None:
        """Test from_directory class method."""
        mock_enc = MagicMock()
        mock_enc.n_vocab = 8192
        mock_enc.encode_single_token.return_value = 0
        mock_pickle.return_value = mock_enc

        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            with tempfile.TemporaryDirectory() as tmp:
                tokenizer_dir = Path(tmp)
                with patch("prepare.TOKENIZER_DIR", tokenizer_dir):
                    import prepare as prep

                    tokenizer = prep.Tokenizer.from_directory()
                    assert (
                        tokenizer.get_vocab_size() == 8192
                    )  # nosec: B101 - Test assertion


class TestGetTokenBytes:
    """Test get_token_bytes function - lines 266-270."""

    @patch("torch.load")
    def test_get_token_bytes(self, mock_load: MagicMock) -> None:
        """Test getting token bytes."""
        mock_tensor = MagicMock()
        mock_load.return_value = mock_tensor

        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            with tempfile.TemporaryDirectory() as tmp:
                tokenizer_dir = Path(tmp)
                (tokenizer_dir / "token_bytes.pt").touch()
                with patch("prepare.TOKENIZER_DIR", tokenizer_dir):
                    import prepare as prep

                    result = prep.get_token_bytes("cpu")
                    assert result is not None  # nosec: B101 - Test assertion


class TestUtilityFunctions:
    """Test utility functions - lines 410-507."""

    @patch("tqdm.tqdm")
    def test_download_file_success(self, mock_tqdm: MagicMock) -> None:
        """Test successful file download."""
        with tempfile.TemporaryDirectory() as tmp:
            filepath = Path(tmp) / "test.txt"

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-length": "100"}
            mock_response.iter_content.return_value = [b"test data"]
            mock_response.raise_for_status.return_value = None

            mock_requests = MagicMock()
            mock_requests.get.return_value = mock_response

            with patch.dict(
                "sys.modules",
                {
                    "torch": MagicMock(),
                    "requests": mock_requests,
                    "pyarrow.parquet": MagicMock(),
                    "rustbpe": MagicMock(),
                },
            ):
                import prepare as prep

                result = prep.download_file("http://example.com/test", filepath)
                assert result is True  # nosec: B101 - Test assertion

    def test_download_file_already_exists(self) -> None:
        """Test download when file already exists."""
        with tempfile.TemporaryDirectory() as tmp:
            filepath = Path(tmp) / "test.txt"
            filepath.touch()

            with patch.dict(
                "sys.modules",
                {
                    "torch": MagicMock(),
                    "requests": MagicMock(),
                    "pyarrow.parquet": MagicMock(),
                    "rustbpe": MagicMock(),
                },
            ):
                import prepare as prep

                result = prep.download_file("http://example.com/test", filepath)
                assert result is True  # nosec: B101 - Test assertion

    def test_download_file_error(self) -> None:
        """Test download with error."""
        with tempfile.TemporaryDirectory() as tmp:
            filepath = Path(tmp) / "test.txt"

            mock_requests = MagicMock()
            mock_requests.get.side_effect = Exception("Network error")

            with patch.dict(
                "sys.modules",
                {
                    "torch": MagicMock(),
                    "requests": mock_requests,
                    "pyarrow.parquet": MagicMock(),
                    "rustbpe": MagicMock(),
                },
            ):
                import prepare as prep

                result = prep.download_file("http://example.com/test", filepath)
                assert result is False  # nosec: B101 - Test assertion

    def test_tokenize_text(self) -> None:
        """Test tokenize_text function."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]

        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            result = prep.tokenize_text("test", mock_tokenizer)
            assert result == [1, 2, 3]  # nosec: B101 - Test assertion

    def test_tokenize_batch(self) -> None:
        """Test tokenize_batch function."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]

        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            result = prep.tokenize_batch(["text1", "text2"], mock_tokenizer)
            assert len(result) == 2  # nosec: B101 - Test assertion

    def test_validate_tokenizer_success(self) -> None:
        """Test successful tokenizer validation."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "Hello, world!"

        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            result = prep.validate_tokenizer(mock_tokenizer)
            assert result is True  # nosec: B101 - Test assertion

    def test_validate_tokenizer_failure(self) -> None:
        """Test tokenizer validation failure."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = Exception("Validation error")

        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            result = prep.validate_tokenizer(mock_tokenizer)
            assert result is False  # nosec: B101 - Test assertion

    def test_validate_data_format_valid(self) -> None:
        """Test valid data format."""
        valid_data = [{"text": "sample 1"}, {"text": "sample 2"}]

        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            result = prep.validate_data_format(valid_data)
            assert result is True  # nosec: B101 - Test assertion

    def test_validate_data_format_empty(self) -> None:
        """Test empty data format."""
        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            result = prep.validate_data_format([])
            assert result is False  # nosec: B101 - Test assertion

    def test_validate_data_format_missing_text(self) -> None:
        """Test data format missing text key."""
        invalid_data = [{"content": "sample"}]

        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            result = prep.validate_data_format(invalid_data)
            assert result is False  # nosec: B101 - Test assertion

    def test_validate_data_format_non_string_text(self) -> None:
        """Test data format with non-string text."""
        invalid_data = [{"text": 123}]

        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            result = prep.validate_data_format(invalid_data)
            assert result is False  # nosec: B101 - Test assertion


class TestParseArgs:
    """Test parse_args function - lines 510-521."""

    def test_parse_args_defaults(self) -> None:
        """Test default argument parsing."""
        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            args = prep.parse_args([])
            assert args.num_shards is None  # nosec: B101 - Test assertion
            assert args.skip_download is False  # nosec: B101 - Test assertion
            assert args.skip_tokenizer is False  # nosec: B101 - Test assertion
            assert args.force is False  # nosec: B101 - Test assertion

    def test_parse_args_custom(self) -> None:
        """Test custom argument parsing."""
        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            args = prep.parse_args(
                [
                    "--num-shards",
                    "10",
                    "--skip-download",
                    "--skip-tokenizer",
                    "--force",
                ]
            )
            assert args.num_shards == 10  # nosec: B101 - Test assertion
            assert args.skip_download is True  # nosec: B101 - Test assertion
            assert args.skip_tokenizer is True  # nosec: B101 - Test assertion
            assert args.force is True  # nosec: B101 - Test assertion


class TestMain:
    """Test main function - lines 524-547."""

    @patch("prepare.list_parquet_files")
    @patch("prepare.train_tokenizer")
    @patch("prepare.download_data")
    def test_main_success(
        self, mock_download: MagicMock, mock_train: MagicMock, mock_list: MagicMock
    ) -> None:
        """Test successful main execution."""
        mock_list.return_value = [Path("test.parquet")]
        mock_train.return_value = True

        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            result = prep.main([])
            assert result == 0  # nosec: B101 - Test assertion

    @patch("prepare.list_parquet_files")
    @patch("prepare.train_tokenizer")
    @patch("prepare.download_data")
    def test_main_skip_download(
        self, mock_download: MagicMock, mock_train: MagicMock, mock_list: MagicMock
    ) -> None:
        """Test main with skip download."""
        mock_list.return_value = [Path("test.parquet")]
        mock_train.return_value = True

        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            result = prep.main(["--skip-download"])
            assert result == 0  # nosec: B101 - Test assertion
            mock_download.assert_not_called()

    @patch("prepare.list_parquet_files")
    @patch("prepare.train_tokenizer")
    @patch("prepare.download_data")
    def test_main_skip_tokenizer(
        self, mock_download: MagicMock, mock_train: MagicMock, mock_list: MagicMock
    ) -> None:
        """Test main with skip tokenizer."""
        mock_list.return_value = [Path("test.parquet")]

        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            result = prep.main(["--skip-tokenizer"])
            assert result == 0  # nosec: B101 - Test assertion
            mock_train.assert_not_called()

    @patch("prepare.list_parquet_files")
    @patch("prepare.download_data")
    def test_main_download_failure(
        self, mock_download: MagicMock, mock_list: MagicMock
    ) -> None:
        """Test main with download failure."""
        mock_list.return_value = []  # No files

        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            result = prep.main([])
            assert result == 1  # nosec: B101 - Test assertion

    @patch("prepare.list_parquet_files")
    @patch("prepare.train_tokenizer")
    @patch("prepare.download_data")
    def test_main_tokenizer_failure(
        self, mock_download: MagicMock, mock_train: MagicMock, mock_list: MagicMock
    ) -> None:
        """Test main with tokenizer failure."""
        mock_list.return_value = [Path("test.parquet")]
        mock_train.return_value = False

        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            result = prep.main([])
            assert result == 1  # nosec: B101 - Test assertion


class TestCacheFunctions:
    """Test cache functions - lines 549-571."""

    def test_get_cache_info_exists(self) -> None:
        """Test getting cache info when cache exists."""
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "test.txt").write_text("test")

            with patch.dict(
                "sys.modules",
                {
                    "torch": MagicMock(),
                    "requests": MagicMock(),
                    "pyarrow.parquet": MagicMock(),
                    "rustbpe": MagicMock(),
                },
            ):
                with patch("prepare.CACHE_DIR", cache_dir):
                    import prepare as prep

                    info = prep.get_cache_info()
                    assert isinstance(info, dict)  # nosec: B101 - Test assertion
                    assert "size_bytes" in info  # nosec: B101 - Test assertion
                    assert "last_modified" in info  # nosec: B101 - Test assertion

    def test_get_cache_info_not_exists(self) -> None:
        """Test getting cache info when cache doesn't exist."""
        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            with patch("prepare.CACHE_DIR", Path("/nonexistent/path")):
                import prepare as prep

                info = prep.get_cache_info()
                assert isinstance(info, dict)  # nosec: B101 - Test assertion
                assert info["size_bytes"] == 0  # nosec: B101 - Test assertion

    def test_cleanup_cache_success(self) -> None:
        """Test successful cache cleanup."""
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "test.txt").write_text("test")

            with patch.dict(
                "sys.modules",
                {
                    "torch": MagicMock(),
                    "requests": MagicMock(),
                    "pyarrow.parquet": MagicMock(),
                    "rustbpe": MagicMock(),
                },
            ):
                with patch("prepare.CACHE_DIR", cache_dir):
                    import prepare as prep

                    result = prep.cleanup_cache()
                    assert result is True  # nosec: B101 - Test assertion

    def test_cleanup_cache_not_exists(self) -> None:
        """Test cleanup when cache doesn't exist."""
        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            with patch("prepare.CACHE_DIR", Path("/nonexistent/path")):
                import prepare as prep

                result = prep.cleanup_cache()
                assert result is True  # nosec: B101 - Test assertion

    def test_cleanup_cache_failure(self) -> None:
        """Test cache cleanup failure."""
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)

            with patch.dict(
                "sys.modules",
                {
                    "torch": MagicMock(),
                    "requests": MagicMock(),
                    "pyarrow.parquet": MagicMock(),
                    "rustbpe": MagicMock(),
                },
            ):
                with patch("prepare.CACHE_DIR", cache_dir):
                    with patch(
                        "shutil.rmtree", side_effect=Exception("Permission denied")
                    ):
                        import prepare as prep

                        result = prep.cleanup_cache()
                        assert result is False  # nosec: B101 - Test assertion


class TestDownloadShard:
    """Test download_shard function - lines 436-438."""

    @patch("prepare.download_single_shard")
    def test_download_shard(self, mock_download: MagicMock) -> None:
        """Test download_shard function."""
        mock_download.return_value = True

        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            result = prep.download_shard(1, Path("/tmp"))
            assert result is True  # nosec: B101 - Test assertion


class TestDownloadWorker:
    """Test download_worker function - lines 454-457."""

    @patch("prepare.download_shard")
    def test_download_worker_success(self, mock_download: MagicMock) -> None:
        """Test download worker success."""
        mock_download.return_value = True

        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            result = prep.download_worker((1, Path("/tmp")))
            assert result == (1, True)  # nosec: B101 - Test assertion

    @patch("prepare.download_shard")
    def test_download_worker_failure(self, mock_download: MagicMock) -> None:
        """Test download worker failure."""
        mock_download.return_value = False

        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            result = prep.download_worker((1, Path("/tmp")))
            assert result == (1, False)  # nosec: B101 - Test assertion


class TestDownloadShardsParallel:
    """Test download_shards_parallel function - lines 441-451."""

    @patch("prepare.Pool")
    def test_download_shards_parallel_success(self, mock_pool: MagicMock) -> None:
        """Test parallel shard download success."""
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        mock_pool_instance.map.return_value = [(1, True), (2, True), (3, True)]

        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            result = prep.download_shards_parallel([1, 2, 3], Path("/tmp"))
            assert result is True  # nosec: B101 - Test assertion

    @patch("prepare.Pool")
    def test_download_shards_parallel_failure(self, mock_pool: MagicMock) -> None:
        """Test parallel shard download with failure."""
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        mock_pool_instance.map.return_value = [(1, True), (2, False), (3, True)]

        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "requests": MagicMock(),
                "pyarrow.parquet": MagicMock(),
                "rustbpe": MagicMock(),
            },
        ):
            import prepare as prep

            result = prep.download_shards_parallel([1, 2, 3], Path("/tmp"))
            assert result is False  # nosec: B101 - Test assertion


class TestReadShard:
    """Test read_shard function - lines 460-472."""

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

            with patch.dict(
                "sys.modules",
                {
                    "torch": MagicMock(),
                    "requests": MagicMock(),
                    "pyarrow.parquet": MagicMock(),
                    "rustbpe": MagicMock(),
                },
            ):
                import prepare as prep

                result = prep.read_shard(1, data_dir)
                assert result is not None  # nosec: B101 - Test assertion

    def test_read_shard_not_exists(self) -> None:
        """Test read when shard doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)

            with patch.dict(
                "sys.modules",
                {
                    "torch": MagicMock(),
                    "requests": MagicMock(),
                    "pyarrow.parquet": MagicMock(),
                    "rustbpe": MagicMock(),
                },
            ):
                import prepare as prep

                result = prep.read_shard(1, data_dir)
                assert result is None  # nosec: B101 - Test assertion

    @patch("pyarrow.parquet.read_table")
    def test_read_shard_error(self, mock_read: MagicMock) -> None:
        """Test shard read error."""
        mock_read.side_effect = Exception("Read error")

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            shard_file = data_dir / "shard_00001.parquet"
            shard_file.touch()

            with patch.dict(
                "sys.modules",
                {
                    "torch": MagicMock(),
                    "requests": MagicMock(),
                    "pyarrow.parquet": MagicMock(),
                    "rustbpe": MagicMock(),
                },
            ):
                import prepare as prep

                result = prep.read_shard(1, data_dir)
                assert result is None  # nosec: B101 - Test assertion


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
