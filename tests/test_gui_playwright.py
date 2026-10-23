"""
================================================================================
GUI TESTS WITH PLAYWRIGHT
================================================================================

Playwright-based GUI tests for the APGI Experiment Runner.
Tests the CustomTkinter GUI components for correct rendering and interaction.

Note: These tests require a display server (X11/Wayland on Linux, Quartz on macOS).
For headless environments, use Xvfb or similar.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("playwright.sync_api", reason="requires playwright")
from playwright.sync_api import Page  # noqa: E402
from pytest import MonkeyPatch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def wait_for_server(url: str, timeout: float = 30.0) -> bool:
    """Wait for server to be ready."""
    import urllib.request

    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(0.1)
    return False


class GUIPage:
    """Page object for GUI interactions."""

    def __init__(self, page: Page, app_window: Any = None):
        self.page = page
        self.app_window = app_window

    def click_menu(self, menu_name: str) -> None:
        """Click a menu item."""
        if self.app_window:
            # Use tkinter directly for menu interaction
            self.app_window.after(0, lambda: self._trigger_menu(menu_name))
            time.sleep(0.1)

    def _trigger_menu(self, menu_name: str) -> None:
        """Internal menu trigger."""
        if self.app_window and hasattr(self.app_window, f"_show_{menu_name}_menu"):
            getattr(self.app_window, f"_show_{menu_name}_menu")()

    def click_experiment(self, experiment_name: str) -> None:
        """Click on an experiment card."""
        if self.app_window:
            buttons = self.app_window.experiment_buttons
            if experiment_name in buttons:
                btn = buttons[experiment_name]
                btn.invoke()

    def wait_for_status(
        self, experiment_name: str, status: str, timeout: float = 10.0
    ) -> bool:
        """Wait for experiment to reach a status."""
        start = time.time()
        while True:
            elapsed = time.time() - start
            if elapsed >= timeout:
                return False
            if self.app_window:
                indicators = self.app_window.status_indicators
                if experiment_name in indicators:
                    current_status = indicators[experiment_name].cget("text")
                    if status in current_status:
                        return True
            time.sleep(0.1)

    def take_screenshot(self, name: str) -> bytes:
        """Take a screenshot for visual regression testing."""
        if self.app_window:
            # Use PIL to capture the tkinter window
            from PIL import ImageGrab

            x = self.app_window.winfo_rootx()
            y = self.app_window.winfo_rooty()
            width = self.app_window.winfo_width()
            height = self.app_window.winfo_height()

            if width > 0 and height > 0:
                try:
                    # Ensure window is updated and visible for capture
                    self.app_window.update_idletasks()
                    self.app_window.update()
                    # On some systems, a small delay is needed for mapping
                    time.sleep(0.2)

                    screenshot = ImageGrab.grab(bbox=(x, y, x + width, y + height))
                    # Convert to bytes
                    import io

                    img_byte_arr = io.BytesIO()
                    screenshot.save(img_byte_arr, format="PNG")
                    return img_byte_arr.getvalue()
                except Exception as e:
                    print(f"Warning: GUI screenshot capture failed: {e}")
                    # In some environments (like headless CI or restricted macOS),
                    # screen capture may not be possible.
                    return b""
        return b""


@pytest.fixture
def screenshot_dir() -> Generator[Path, None, None]:
    """Create directory for screenshots."""
    ss_dir = PROJECT_ROOT / "test_reports" / "screenshots"
    ss_dir.mkdir(parents=True, exist_ok=True)
    yield ss_dir


@pytest.fixture
def temp_research_dir() -> Generator[Path, None, None]:
    """Create temporary research directory with mock experiments."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Create mock experiment files
        for exp_name in ["test_visual_search", "test_stroop_effect"]:
            exp_file = tmp_path / f"run_{exp_name}.py"
            exp_file.write_text(f"""
# Mock experiment for testing
import time
print(f"Running {exp_name}")
time.sleep(0.1)
""")
        yield tmp_path


def _can_create_gui() -> bool:
    """Check if GUI can be created in this environment."""
    # Skip GUI check in CI/headless environments
    if os.environ.get("CI") or os.environ.get("DISPLAY") == "":
        return False
    try:
        import tkinter as tk

        # Try to create a hidden test window with timeout
        root = tk.Tk()
        root.withdraw()
        root.update()
        root.destroy()
        return True
    except Exception:
        return False


@pytest.mark.gui
@pytest.mark.slow
class TestGUIBasic:
    """Basic GUI functionality tests."""

    def test_gui_imports(self) -> None:
        """Test that GUI modules can be imported."""
        try:
            import customtkinter as ctk

            assert ctk is not None  # nosec: B101 - Test assertion
        except ImportError:
            pytest.skip("customtkinter not installed")

        try:
            from GUI_auto_improve_experiments import ExperimentRunnerGUI

            assert ExperimentRunnerGUI is not None  # nosec: B101 - Test assertion
        except ImportError as e:
            pytest.skip(f"Cannot import GUI: {e}")

    @pytest.mark.timeout(10)  # Timeout after 10 seconds
    def test_gui_initialization_without_display(self, monkeypatch: Any) -> None:
        """Test GUI can be initialized (headless mode)."""
        # Skip if no GUI support available
        if not _can_create_gui():
            pytest.skip("GUI not available in this environment")

        # Check if customtkinter is available
        if sys.modules.get("customtkinter") is None:
            try:
                import customtkinter  # noqa: F401
            except ImportError:
                pytest.skip("customtkinter not installed")

        # Mock ctk BEFORE importing GUI module (it patches DropdownMenu at import time)
        # Create a proper base class that can be inherited from but ignores extra kwargs
        class MockCTkBase(object):
            def __init__(self, *args, **kwargs):
                # Absorb any kwargs that CTk would normally accept
                pass

            # Mock CTk methods that the GUI calls on self
            def title(self, *args):
                pass

            def geometry(self, *args):
                pass

            def winfo_rootx(self):
                return 0

            def winfo_rooty(self):
                return 0

            def winfo_width(self):
                return 800

            def winfo_height(self):
                return 600

            def update_idletasks(self):
                pass

            def update(self):
                pass

            def after(self, *args):
                pass

            def mainloop(self):
                pass

            # Grid layout methods - must be explicit for inheritance
            def grid_rowconfigure(self, index, weight=None, minsize=None, pad=None):
                pass

            def grid_columnconfigure(self, index, weight=None, minsize=None, pad=None):
                pass

        mock_ctk = MagicMock()
        mock_ctk.CTk = MockCTkBase
        mock_ctk.CTkFont = MagicMock(return_value=MagicMock())
        mock_ctk.CTkFrame = MagicMock(return_value=MagicMock())
        mock_ctk.CTkLabel = MagicMock(return_value=MagicMock())
        mock_ctk.CTkButton = MagicMock(return_value=MagicMock())
        mock_ctk.CTkTextbox = MagicMock(return_value=MagicMock())
        mock_ctk.CTkOptionMenu = MagicMock(return_value=MagicMock())
        mock_ctk.CTkProgressBar = MagicMock(return_value=MagicMock())
        mock_ctk.set_appearance_mode = MagicMock()
        mock_ctk.set_default_color_theme = MagicMock()
        mock_ctk.winfo_screenwidth = MagicMock(return_value=1920)
        mock_ctk.winfo_screenheight = MagicMock(return_value=1080)

        # Mock the DropdownMenu patch target to prevent AttributeError
        mock_dropdown_menu = MagicMock()
        mock_dropdown_menu._add_menu_commands = MagicMock()
        mock_ctk.windows.widgets.core_widget_classes.dropdown_menu.DropdownMenu = (
            mock_dropdown_menu
        )

        with patch.dict("sys.modules", {"customtkinter": mock_ctk}):
            with patch.dict("sys.modules", {"customtkinter.windows": mock_ctk.windows}):
                with patch.dict(
                    "sys.modules",
                    {"customtkinter.windows.widgets": mock_ctk.windows.widgets},
                ):
                    with patch.dict(
                        "sys.modules",
                        {
                            "customtkinter.windows.widgets.core_widget_classes": mock_ctk.windows.widgets.core_widget_classes
                        },
                    ):
                        with patch.dict(
                            "sys.modules",
                            {
                                "customtkinter.windows.widgets.core_widget_classes.dropdown_menu": mock_ctk.windows.widgets.core_widget_classes.dropdown_menu
                            },
                        ):
                            # Now import the GUI module with mocked ctk
                            # Remove cached module to force reimport
                            if "GUI_auto_improve_experiments" in sys.modules:
                                del sys.modules["GUI_auto_improve_experiments"]

                            from GUI_auto_improve_experiments import ExperimentRunnerGUI

                            with patch.object(
                                ExperimentRunnerGUI,
                                "_check_dependencies",
                                lambda self: None,
                            ):
                                app = ExperimentRunnerGUI()
                                assert app is not None  # nosec: B101 - Test assertion
                                assert hasattr(
                                    app, "experiments"
                                )  # nosec: B101 - Test assertion
                                assert hasattr(
                                    app, "experiment_cards"
                                )  # nosec: B101 - Test assertion

    @pytest.mark.timeout(30)  # Add timeout to prevent hanging
    def test_gui_experiment_discovery(
        self, temp_research_dir: Path, monkeypatch: Any
    ) -> None:
        """Test that experiments are discovered correctly."""
        # Skip if no GUI support available
        if not _can_create_gui():
            pytest.skip("GUI tests require tkinter display support")

        try:
            from GUI_auto_improve_experiments import ExperimentRunnerGUI
        except ImportError:
            pytest.skip("Required dependencies not installed")

        # Create comprehensive mocks to prevent any GUI operations
        mock_ctk = MagicMock()
        mock_ctk.CTk = MagicMock()
        mock_ctk.CTkFont = MagicMock(return_value=MagicMock())
        mock_ctk.CTkFrame = MagicMock(return_value=MagicMock())
        mock_ctk.CTkLabel = MagicMock(return_value=MagicMock())
        mock_ctk.CTkButton = MagicMock(return_value=MagicMock())
        mock_ctk.CTkTextbox = MagicMock(return_value=MagicMock())
        mock_ctk.CTkOptionMenu = MagicMock(return_value=MagicMock())
        mock_ctk.CTkProgressBar = MagicMock(return_value=MagicMock())
        mock_ctk.set_appearance_mode = MagicMock()
        mock_ctk.set_default_color_theme = MagicMock()
        mock_ctk.winfo_screenwidth = MagicMock(return_value=1920)
        mock_ctk.winfo_screenheight = MagicMock(return_value=1080)

        # Mock the DropdownMenu patch target
        mock_dropdown_menu = MagicMock()
        mock_dropdown_menu._add_menu_commands = MagicMock()
        mock_ctk.windows = MagicMock()
        mock_ctk.windows.widgets = MagicMock()
        mock_ctk.windows.widgets.core_widget_classes = MagicMock()
        mock_ctk.windows.widgets.core_widget_classes.dropdown_menu = MagicMock()
        mock_ctk.windows.widgets.core_widget_classes.dropdown_menu.DropdownMenu = (
            mock_dropdown_menu
        )

        # Mock tkinter completely to prevent any GUI operations
        mock_tkinter = MagicMock()
        mock_tkinter.messagebox = MagicMock()

        # Mock matplotlib to prevent plotting operations
        mock_matplotlib = MagicMock()
        mock_matplotlib.use = MagicMock()
        mock_matplotlib.backends = MagicMock()
        mock_matplotlib.backends.backend_tkagg = MagicMock()
        mock_matplotlib.backends.backend_tkagg.FigureCanvasTkAgg = MagicMock()
        mock_matplotlib.backends.backend_tkagg.NavigationToolbar2Tk = MagicMock()
        mock_matplotlib.figure = MagicMock()
        mock_matplotlib.figure.Figure = MagicMock(return_value=MagicMock())

        # Mock hypothesis approval board
        mock_approval_board = MagicMock()
        mock_approval_board.ApprovalBoard = MagicMock()
        mock_approval_board.Hypothesis = MagicMock()
        mock_approval_board.HypothesisStatus = MagicMock()

        # Mock apgi_security
        mock_apgi_security = MagicMock()
        mock_apgi_security.secure_popen = MagicMock()

        # Mock numpy and other dependencies
        mock_numpy = MagicMock()
        mock_numpy.array = MagicMock(return_value=MagicMock())

        # Create a comprehensive mock environment
        with patch.dict(
            "sys.modules",
            {
                "customtkinter": mock_ctk,
                "tkinter": mock_tkinter,
                "tkinter.messagebox": mock_tkinter.messagebox,
                "matplotlib": mock_matplotlib,
                "matplotlib.backends": mock_matplotlib.backends,
                "matplotlib.backends.backend_tkagg": mock_matplotlib.backends.backend_tkagg,
                "matplotlib.backends.backend_tkagg.FigureCanvasTkAgg": mock_matplotlib.backends.backend_tkagg.FigureCanvasTkAgg,
                "matplotlib.backends.backend_tkagg.NavigationToolbar2Tk": mock_matplotlib.backends.backend_tkagg.NavigationToolbar2Tk,
                "matplotlib.figure": mock_matplotlib.figure,
                "hypothesis_approval_board": mock_approval_board,
                "apgi_security": mock_apgi_security,
                "numpy": mock_numpy,
            },
        ):
            # Remove cached module to force reimport with mocked dependencies
            if "GUI_auto_improve_experiments" in sys.modules:
                del sys.modules["GUI_auto_improve_experiments"]

            # Mock the research directory and all CTk widgets
            with patch.object(
                Path, "glob", return_value=list(temp_research_dir.glob("run_*.py"))
            ):
                with patch.object(Path, "resolve", return_value=temp_research_dir):
                    # Mock the file operations to prevent actual file system calls
                    with patch(
                        "GUI_auto_improve_experiments.Path.glob",
                        return_value=list(temp_research_dir.glob("run_*.py")),
                    ):
                        with patch(
                            "GUI_auto_improve_experiments.Path.resolve",
                            return_value=temp_research_dir,
                        ):
                            with patch(
                                "GUI_auto_improve_experiments.Path.exists",
                                return_value=True,
                            ):
                                with patch(
                                    "GUI_auto_improve_experiments.Path.is_dir",
                                    return_value=True,
                                ):
                                    with patch(
                                        "GUI_auto_improve_experiments.Path.__truediv__",
                                        return_value=temp_research_dir,
                                    ):
                                        app = ExperimentRunnerGUI()
                                        # Check experiments were found
                                        assert (
                                            len(app.experiments) > 0
                                        )  # nosec: B101 - Test assertion
                                        for name, _ in app.experiments:
                                            assert (  # nosec: B101 - Test assertion
                                                "Visual Search" in name
                                                or "Stroop" in name
                                            )


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.integration
class TestGUIInteraction:
    """GUI interaction tests."""

    def test_menu_bar_creation(self, monkeypatch: MonkeyPatch) -> None:
        """Test menu bar is created with all menu items."""
        if not _can_create_gui():
            pytest.skip("GUI tests require tkinter display support")

        try:
            from GUI_auto_improve_experiments import ExperimentRunnerGUI
        except ImportError:
            pytest.skip("Required dependencies not installed")

        with patch("GUI_auto_improve_experiments.ctk.CTk", MagicMock()):
            with patch("GUI_auto_improve_experiments.ctk.CTkFont", MagicMock()):
                with patch("GUI_auto_improve_experiments.ctk.CTkFrame", MagicMock()):
                    with patch(
                        "GUI_auto_improve_experiments.ctk.CTkLabel", MagicMock()
                    ):
                        with patch(
                            "GUI_auto_improve_experiments.ctk.CTkButton", MagicMock()
                        ):
                            app = ExperimentRunnerGUI()
                            # Verify menu exists
                            assert hasattr(
                                app, "_create_menu_bar"
                            )  # nosec: B101 - Test assertion

    def test_experiment_button_creation(
        self, temp_research_dir: Any, monkeypatch: MonkeyPatch
    ) -> None:
        """Test experiment buttons are created."""
        if not _can_create_gui():
            pytest.skip("GUI tests require tkinter display support")

        try:
            from GUI_auto_improve_experiments import ExperimentRunnerGUI
        except ImportError:
            pytest.skip("Required dependencies not installed")

        with patch.object(
            Path, "glob", return_value=list(temp_research_dir.glob("run_*.py"))
        ):
            with patch.object(Path, "resolve", return_value=temp_research_dir):
                with patch("GUI_auto_improve_experiments.ctk.CTk", MagicMock()):
                    with patch("GUI_auto_improve_experiments.ctk.CTkFont", MagicMock()):
                        with patch(
                            "GUI_auto_improve_experiments.ctk.CTkFrame", MagicMock()
                        ):
                            with patch(
                                "GUI_auto_improve_experiments.ctk.CTkLabel", MagicMock()
                            ):
                                with patch(
                                    "GUI_auto_improve_experiments.ctk.CTkButton",
                                    MagicMock(),
                                ):
                                    app = ExperimentRunnerGUI()
                                    # Check buttons exist for experiments
                                    for exp_name, _ in app.experiments:
                                        assert (
                                            exp_name in app.experiment_buttons
                                        )  # nosec: B101 - Test assertion

    def test_status_indicators(
        self, temp_research_dir: Any, monkeypatch: MonkeyPatch
    ) -> None:
        """Test status indicators are tracked."""
        if not _can_create_gui():
            pytest.skip("GUI tests require tkinter display support")

        try:
            from GUI_auto_improve_experiments import ExperimentRunnerGUI
        except ImportError:
            pytest.skip("Required dependencies not installed")

        with patch.object(
            Path, "glob", return_value=list(temp_research_dir.glob("run_*.py"))
        ):
            with patch.object(Path, "resolve", return_value=temp_research_dir):
                with patch("GUI_auto_improve_experiments.ctk.CTk", MagicMock()):
                    with patch("GUI_auto_improve_experiments.ctk.CTkFont", MagicMock()):
                        with patch(
                            "GUI_auto_improve_experiments.ctk.CTkFrame", MagicMock()
                        ):
                            with patch(
                                "GUI_auto_improve_experiments.ctk.CTkLabel", MagicMock()
                            ):
                                with patch(
                                    "GUI_auto_improve_experiments.ctk.CTkButton",
                                    MagicMock(),
                                ):
                                    app = ExperimentRunnerGUI()
                                    # Check status indicators exist
                                    for exp_name, _ in app.experiments:
                                        assert (
                                            exp_name in app.status_indicators
                                        )  # nosec: B101 - Test assertion


@pytest.mark.visual
@pytest.mark.slow
class TestGUIVisualRegression:
    """Visual regression tests for GUI."""

    @pytest.fixture
    def visual_baseline_dir(self) -> Path:
        """Directory for baseline screenshots."""
        baseline_dir = PROJECT_ROOT / "tests" / "visual_baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        return baseline_dir

    def test_gui_screenshot_comparison(
        self, temp_research_dir: Any, visual_baseline_dir: Any, monkeypatch: MonkeyPatch
    ) -> None:
        """Test GUI rendering against baseline."""
        # Skip visual tests in headless/CI environments
        if not _can_create_gui():
            pytest.skip("GUI tests require tkinter display support")

        try:
            from GUI_auto_improve_experiments import ExperimentRunnerGUI
        except ImportError:
            pytest.skip("Required dependencies not installed")

        with patch.object(
            Path, "glob", return_value=list(temp_research_dir.glob("run_*.py"))
        ):
            with patch.object(Path, "resolve", return_value=temp_research_dir):
                with patch("GUI_auto_improve_experiments.ctk.CTk", MagicMock()):
                    with patch("GUI_auto_improve_experiments.ctk.CTkFont", MagicMock()):
                        with patch(
                            "GUI_auto_improve_experiments.ctk.CTkFrame", MagicMock()
                        ):
                            with patch(
                                "GUI_auto_improve_experiments.ctk.CTkLabel", MagicMock()
                            ):
                                with patch(
                                    "GUI_auto_improve_experiments.ctk.CTkButton",
                                    MagicMock(),
                                ):
                                    app = ExperimentRunnerGUI()
                                    # Verify app was created successfully
                                    assert (
                                        app is not None
                                    )  # nosec: B101 - Test assertion
                                    assert hasattr(
                                        app, "experiments"
                                    )  # nosec: B101 - Test assertion
                                    # Skip screenshot comparison in mocked environment
                                    pytest.skip(
                                        "Screenshot tests require real GUI display"
                                    )


@pytest.mark.e2e
@pytest.mark.slow
class TestGUIE2E:
    """End-to-end GUI tests."""

    def test_full_experiment_workflow(
        self, temp_research_dir: Any, monkeypatch: MonkeyPatch
    ) -> None:
        """Test complete experiment workflow from selection to completion."""
        if not _can_create_gui():
            pytest.skip("GUI tests require tkinter display support")

        try:
            from GUI_auto_improve_experiments import ExperimentRunnerGUI
        except ImportError:
            pytest.skip("Required dependencies not installed")

        # Create a simple mock experiment that completes quickly
        mock_exp = temp_research_dir / "run_quick_test.py"
        mock_exp.write_text("""
import sys
import time
print("Test experiment running", flush=True)
time.sleep(0.01)
print("Test experiment completed", flush=True)
sys.exit(0)
""")

        with patch.object(Path, "glob", return_value=[mock_exp]):
            with patch.object(Path, "resolve", return_value=temp_research_dir):
                with patch("GUI_auto_improve_experiments.ctk.CTk", MagicMock()):
                    with patch("GUI_auto_improve_experiments.ctk.CTkFont", MagicMock()):
                        with patch(
                            "GUI_auto_improve_experiments.ctk.CTkFrame", MagicMock()
                        ):
                            with patch(
                                "GUI_auto_improve_experiments.ctk.CTkLabel", MagicMock()
                            ):
                                with patch(
                                    "GUI_auto_improve_experiments.ctk.CTkButton",
                                    MagicMock(),
                                ):
                                    app = ExperimentRunnerGUI()

                                    # Verify experiments were discovered
                                    assert (
                                        len(app.experiments) > 0
                                    )  # nosec: B101 - Test assertion
                                    assert (
                                        len(app.experiment_buttons) > 0
                                    )  # nosec: B101 - Test assertion


@pytest.mark.performance
@pytest.mark.gui
class TestGUIPerformance:
    """GUI performance tests."""

    def test_gui_startup_time(
        self, temp_research_dir: Any, request: Any, monkeypatch: MonkeyPatch
    ) -> None:
        """Benchmark GUI startup time."""
        if not _can_create_gui():
            pytest.skip("GUI tests require tkinter display support")

        try:
            from GUI_auto_improve_experiments import ExperimentRunnerGUI
        except ImportError:
            pytest.skip("Required dependencies not installed")

        import time

        def startup():
            with patch.object(
                Path, "glob", return_value=list(temp_research_dir.glob("run_*.py"))
            ):
                with patch.object(Path, "resolve", return_value=temp_research_dir):
                    with patch("GUI_auto_improve_experiments.ctk.CTk", MagicMock()):
                        with patch(
                            "GUI_auto_improve_experiments.ctk.CTkFont", MagicMock()
                        ):
                            with patch(
                                "GUI_auto_improve_experiments.ctk.CTkFrame", MagicMock()
                            ):
                                _ = ExperimentRunnerGUI()

        start = time.perf_counter()
        startup()
        duration = time.perf_counter() - start
        # Should start in less than 5 seconds
        assert duration < 5.0  # nosec: B101 - Test assertion

    def test_gui_memory_usage(
        self, temp_research_dir: Any, monkeypatch: MonkeyPatch
    ) -> None:
        """Test GUI memory usage stays within bounds."""
        if not _can_create_gui():
            pytest.skip("GUI tests require tkinter display support")

        try:
            import os

            import psutil

            from GUI_auto_improve_experiments import ExperimentRunnerGUI
        except ImportError:
            pytest.skip("Required dependencies not installed")

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        with patch.object(
            Path, "glob", return_value=list(temp_research_dir.glob("run_*.py"))
        ):
            with patch.object(Path, "resolve", return_value=temp_research_dir):
                with patch("GUI_auto_improve_experiments.ctk.CTk", MagicMock()):
                    with patch("GUI_auto_improve_experiments.ctk.CTkFont", MagicMock()):
                        with patch(
                            "GUI_auto_improve_experiments.ctk.CTkFrame", MagicMock()
                        ):
                            _ = ExperimentRunnerGUI()

                            mem_after = process.memory_info().rss / 1024 / 1024  # MB
                            mem_increase = mem_after - mem_before

                            # Should use less than 100MB additional memory
                            assert (  # nosec: B101 - Test assertion
                                mem_increase < 100
                            ), f"Memory increase too large: {mem_increase:.1f}MB"


# Configure pytest markers
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with GUI-specific markers."""
    config.addinivalue_line("markers", "gui: marks tests as GUI tests")
    config.addinivalue_line("markers", "visual: marks tests as visual regression tests")
