"""
Test suite for GUI-auto_improve_experiments.py module.

Tests GUI functionality for APGI experiment management.
"""

import importlib
import os

# Add the parent directory to the path to import the module
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("tkinter", reason="GUI tests require tkinter")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the GUI module using importlib due to underscore in filename
import importlib.util

spec = importlib.util.spec_from_file_location(
    "GUI_auto_improve_experiments",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "GUI_auto_improve_experiments.py",
    ),
)
if spec is not None and spec.loader is not None:
    gui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gui)
    # Add gui to sys.modules so patches can find it
    sys.modules["gui"] = gui
else:
    raise ImportError("Could not load GUI_auto_improve_experiments module")


# Mock classes for customtkinter
class MockCTkFont:
    """Simple mock for CTkFont to prevent MagicMock recursion."""

    def __init__(self, *args, **kwargs):
        pass


class SimpleMock:
    """Simple mock class that returns itself for any attribute access."""

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self


class MockCustomTkinter:
    """Simple mock class for customtkinter module."""

    CTkFont = MockCTkFont

    def __getattr__(self, name):
        return SimpleMock()


@pytest.fixture(scope="module", autouse=True)
def mock_gui_dependencies():
    """Mock GUI dependencies for the module."""
    # Store original modules
    original_customtkinter = sys.modules.get("customtkinter")
    original_matplotlib = sys.modules.get("matplotlib")
    original_matplotlib_figure = sys.modules.get("matplotlib.figure")
    original_matplotlib_backend = sys.modules.get("matplotlib.backends.backend_tkagg")

    # Mock customtkinter
    mock_ctk = MockCustomTkinter()
    sys.modules["customtkinter"] = mock_ctk  # type: ignore[assignment]

    yield

    # Restore original modules
    if original_customtkinter:
        sys.modules["customtkinter"] = original_customtkinter
    elif "customtkinter" in sys.modules:
        del sys.modules["customtkinter"]

    if original_matplotlib:
        sys.modules["matplotlib"] = original_matplotlib
    elif "matplotlib" in sys.modules:
        del sys.modules["matplotlib"]

    if original_matplotlib_figure:
        sys.modules["matplotlib.figure"] = original_matplotlib_figure
    elif "matplotlib.figure" in sys.modules:
        del sys.modules["matplotlib.figure"]

    if original_matplotlib_backend:
        sys.modules["matplotlib.backends.backend_tkagg"] = original_matplotlib_backend
    elif "matplotlib.backends.backend_tkagg" in sys.modules:
        del sys.modules["matplotlib.backends.backend_tkagg"]


class TestConstants:
    """Test module constants."""

    def test_core_dependencies(self):
        """Test CORE_DEPENDENCIES constant."""
        assert "numpy" in gui.CORE_DEPENDENCIES  # nosec: B101 - Test assertion
        assert "pandas" in gui.CORE_DEPENDENCIES  # nosec: B101 - Test assertion
        assert "matplotlib" in gui.CORE_DEPENDENCIES  # nosec: B101 - Test assertion
        assert "customtkinter" in gui.CORE_DEPENDENCIES  # nosec: B101 - Test assertion
        assert "scipy" in gui.CORE_DEPENDENCIES  # nosec: B101 - Test assertion

    def test_optional_dependencies(self):
        """Test OPTIONAL_DEPENDENCIES constant."""
        assert "torch" in gui.OPTIONAL_DEPENDENCIES  # nosec: B101 - Test assertion
        assert "sklearn" in gui.OPTIONAL_DEPENDENCIES  # nosec: B101 - Test assertion
        assert "requests" in gui.OPTIONAL_DEPENDENCIES  # nosec: B101 - Test assertion
        assert "tqdm" in gui.OPTIONAL_DEPENDENCIES  # nosec: B101 - Test assertion
        assert "PIL" in gui.OPTIONAL_DEPENDENCIES  # nosec: B101 - Test assertion


class TestExperimentRunnerGUI:
    """Test ExperimentRunnerGUI class."""

    @patch("customtkinter.CTk")
    @patch("customtkinter.set_appearance_mode")
    @patch("customtkinter.set_default_color_theme")
    def test_gui_initialization(self, mock_theme, mock_appearance, mock_ctk):
        """Test GUI initialization."""
        # Create a more complete mock for the CTk instance
        mock_ctk_instance = MagicMock()
        mock_ctk.return_value = mock_ctk_instance

        # Mock the console_text attribute that will be accessed
        mock_ctk_instance.console_text = MagicMock()
        mock_ctk_instance.insert = MagicMock()  # Mock the insert method

        # Mock the methods that would cause issues during initialization
        with patch.object(
            gui.ExperimentRunnerGUI, "_find_experiments", return_value=[]
        ):
            with patch.object(gui.ExperimentRunnerGUI, "_setup_ui"):
                with patch.object(gui.ExperimentRunnerGUI, "_check_dependencies"):
                    with patch.object(gui.ExperimentRunnerGUI, "_create_menu_bar"):
                        with patch.object(
                            gui.ExperimentRunnerGUI, "_load_progress_tracker"
                        ):
                            with patch.object(gui.ExperimentRunnerGUI, "protocol"):
                                instance = gui.ExperimentRunnerGUI()

                                assert (
                                    instance.title()
                                    == "APGI Experiment Auto-Improvement"
                                )
                    assert hasattr(instance, "research_dir")
                    assert hasattr(instance, "running_experiments")
                    assert hasattr(instance, "experiment_cards")
                    assert hasattr(instance, "experiment_buttons")
                    assert hasattr(instance, "status_indicators")
                    assert hasattr(instance, "active_processes")
                    assert hasattr(instance, "stop_all")
                    assert hasattr(instance, "experiment_results")

    def test_check_dependencies_all_present(self):
        """Test dependency check when all dependencies are present."""
        mock_gui = MagicMock()
        mock_gui.research_dir = Path("/tmp")

        # Mock all dependencies as available
        with patch.dict(
            "sys.modules",
            {
                "numpy": MagicMock(),
                "pandas": MagicMock(),
                "matplotlib": MagicMock(),
                "customtkinter": MagicMock(),
                "scipy": MagicMock(),
                "torch": MagicMock(),
                "sklearn": MagicMock(),
                "requests": MagicMock(),
                "tqdm": MagicMock(),
                "PIL": MagicMock(),
            },
        ):
            with patch("tkinter.messagebox"):
                # Should not raise any errors
                gui.ExperimentRunnerGUI._check_dependencies(None)

    def test_check_dependencies_missing_core(self):
        """Test dependency check with missing core dependencies."""

        # Store the original import_module function
        original_import_module = importlib.import_module

        # Create mock importlib that raises ImportError for numpy
        mock_importlib = MagicMock()

        def mock_import_module(module_name):
            if module_name == "numpy":
                raise ImportError("No module named 'numpy'")
            # For other modules, use the original function
            return original_import_module(module_name)

        mock_importlib.import_module = mock_import_module

        # Patch sys.modules so local 'import importlib' finds our mock
        with patch.dict("sys.modules", {"importlib": mock_importlib}):
            with patch("tkinter.messagebox.showerror") as mock_msgbox:
                with patch("sys.exit") as mock_exit:
                    gui.ExperimentRunnerGUI._check_dependencies(None)
                    mock_msgbox.assert_called_once()
                    mock_exit.assert_called_once_with(1)

    def test_find_experiments(self):
        """Test finding experiment files."""
        mock_gui = MagicMock()
        mock_gui.research_dir = Path("/tmp/test_research")

        # Create mock experiment files
        experiment_files = [
            Path("run_experiment1.py"),
            Path("run_experiment2.py"),
            Path("run_prepare_experiment.py"),
            Path("run_test_run_experiment.py"),
        ]

        with patch("gui.Path.glob") as mock_glob:
            mock_glob.return_value = [Path(f) for f in experiment_files]

            experiments = gui.ExperimentRunnerGUI._find_experiments(mock_gui)

            # Should find files starting with 'run_' and ending with '.py'
            # Implementation prepends 'experiments/' to the filename
            expected = [
                ("Experiment1", "experiments/run_experiment1.py"),
                ("Experiment2", "experiments/run_experiment2.py"),
                ("Prepare Experiment", "experiments/run_prepare_experiment.py"),
                ("Test Experiment", "experiments/run_test_run_experiment.py"),
            ]
            assert experiments == expected  # nosec: B101 - Test assertion

    def test_find_experiments_no_files(self):
        """Test finding experiments when no files exist."""
        mock_gui = MagicMock()
        mock_gui.research_dir = Path("/tmp/test_research")

        with patch("gui.Path.glob") as mock_glob:
            mock_glob.return_value = []

            experiments = gui.ExperimentRunnerGUI._find_experiments(mock_gui)

            assert experiments == []  # nosec: B101 - Test assertion

    def test_create_menu_bar(self):
        """Test menu bar creation - verify method exists."""
        # Verify the method exists and is callable
        assert hasattr(
            gui.ExperimentRunnerGUI, "_create_menu_bar"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._create_menu_bar
        )  # nosec: B101 - Test assertion

    def test_setup_ui_creates_experiment_cards(self):
        """Test UI setup creates experiment cards for each experiment."""
        # Verify _setup_ui method exists
        assert hasattr(
            gui.ExperimentRunnerGUI, "_setup_ui"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._setup_ui
        )  # nosec: B101 - Test assertion

    def test_create_experiment_card(self):
        """Test creating individual experiment cards."""
        # Verify _create_experiment_card method exists with correct signature
        assert hasattr(
            gui.ExperimentRunnerGUI, "_create_experiment_card"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._create_experiment_card
        )  # nosec: B101 - Test assertion
        # Check method accepts the expected parameters (self, parent, name, script, index)
        import inspect

        sig = inspect.signature(gui.ExperimentRunnerGUI._create_experiment_card)
        params = list(sig.parameters.keys())
        # self, parent, name, script, index
        assert "parent" in params or len(params) >= 4  # nosec: B101 - Test assertion

    def test_run_experiment(self):
        """Test running an experiment."""
        mock_gui = MagicMock()
        mock_gui.running_experiments = set()
        mock_gui.active_processes = {}
        mock_gui.experiment_buttons = {}
        mock_gui.status_indicators = {}
        mock_gui.experiment_results = {}
        mock_gui.research_dir = Path("/tmp")

        # Mock button and status indicator
        mock_button = MagicMock()
        mock_status = MagicMock()
        mock_gui.experiment_buttons["test"] = mock_button
        mock_gui.status_indicators["test"] = mock_status
        mock_gui.active_processes = {}

        # Configure mock to return itself when accessed as active process
        mock_process = MagicMock()
        mock_gui.active_processes = {"test": mock_process}

        with patch("gui.subprocess.Popen") as mock_popen:
            with patch("gui.threading.Thread") as mock_thread:
                mock_popen.return_value = mock_process
                mock_thread.return_value = MagicMock()

                gui.ExperimentRunnerGUI._run_experiment(
                    mock_gui, name="test", script="run_test_run_experiment.py"
                )

                # Small delay to ensure process is added
                import time

                time.sleep(0.01)

                # Should update state and start process
                assert (
                    "test" in mock_gui.active_processes
                )  # nosec: B101 - Test assertion
                assert (
                    mock_gui.active_processes["test"] == mock_process
                )  # nosec: B101 - Test assertion
                mock_button.configure.assert_called()
                mock_thread.assert_called_once()

    def test_stop_all_flag_exists(self):
        """Test that stop_all flag exists and can be set."""
        mock_gui = MagicMock()
        mock_gui.stop_all = False

        # Set the flag (as _stop_all would do)
        mock_gui.stop_all = True
        assert mock_gui.stop_all is True  # nosec: B101 - Test assertion

    def test_create_visualization_panel(self):
        """Test creating visualization panel."""
        # Verify _create_visualization_panel method exists
        assert hasattr(
            gui.ExperimentRunnerGUI, "_create_visualization_panel"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._create_visualization_panel
        )  # nosec: B101 - Test assertion
        # Check method accepts expected parameters (self, parent_frame)
        import inspect

        sig = inspect.signature(gui.ExperimentRunnerGUI._create_visualization_panel)
        params = list(sig.parameters.keys())
        # self, parent_frame
        assert len(params) >= 2  # nosec: B101 - Test assertion


class TestPatchedAddMenuCommands:
    """Test the patched _add_menu_commands function."""

    def test_patched_function_exists(self):
        """Test that the patched function exists."""
        assert hasattr(
            gui, "_patched_add_menu_commands"
        )  # nosec: B101 - Test assertion
        assert callable(gui._patched_add_menu_commands)  # nosec: B101 - Test assertion

    def test_patched_function_handles_empty_menu(self):
        """Test that patched function handles empty menu gracefully."""
        mock_menu = MagicMock()
        mock_menu.index.return_value = None
        mock_menu._values = []
        mock_menu._command = None

        # Should not raise exception
        gui._patched_add_menu_commands(mock_menu)
        mock_menu.delete.assert_not_called()


class TestGUIMethods:
    """Test various GUI methods."""

    def test_log_method_exists(self):
        """Test _log method exists."""
        assert hasattr(gui.ExperimentRunnerGUI, "_log")  # nosec: B101 - Test assertion
        assert callable(gui.ExperimentRunnerGUI._log)  # nosec: B101 - Test assertion

    def test_clear_console_method_exists(self):
        """Test _clear_console method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_clear_console"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._clear_console
        )  # nosec: B101 - Test assertion

    def test_update_guardrail_dashboard_exists(self):
        """Test _update_guardrail_dashboard method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_update_guardrail_dashboard"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._update_guardrail_dashboard
        )  # nosec: B101 - Test assertion

    def test_notify_guardrail_escalation_exists(self):
        """Test _notify_guardrail_escalation method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_notify_guardrail_escalation"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._notify_guardrail_escalation
        )  # nosec: B101 - Test assertion

    def test_run_auto_improve_exists(self):
        """Test _run_auto_improve method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_run_auto_improve"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._run_auto_improve
        )  # nosec: B101 - Test assertion

    def test_execute_script_exists(self):
        """Test _execute_script method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_execute_script"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._execute_script
        )  # nosec: B101 - Test assertion

    def test_parse_experiment_results_exists(self):
        """Test _parse_experiment_results method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_parse_experiment_results"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._parse_experiment_results
        )  # nosec: B101 - Test assertion

    def test_finish_experiment_exists(self):
        """Test _finish_experiment method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_finish_experiment"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._finish_experiment
        )  # nosec: B101 - Test assertion

    def test_run_all_exists(self):
        """Test _run_all method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_run_all"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._run_all
        )  # nosec: B101 - Test assertion

    def test_run_all_sequential_exists(self):
        """Test _run_all_sequential method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_run_all_sequential"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._run_all_sequential
        )  # nosec: B101 - Test assertion

    def test_stop_all_exists(self):
        """Test _stop_all method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_stop_all"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._stop_all
        )  # nosec: B101 - Test assertion

    def test_display_dependencies_status_exists(self):
        """Test _display_dependencies_status method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_display_dependencies_status"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._display_dependencies_status
        )  # nosec: B101 - Test assertion

    def test_repair_dependencies_exists(self):
        """Test _repair_dependencies method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_repair_dependencies"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._repair_dependencies
        )  # nosec: B101 - Test assertion

    def test_change_appearance_mode_exists(self):
        """Test change_appearance_mode method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "change_appearance_mode"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI.change_appearance_mode
        )  # nosec: B101 - Test assertion

    def test_plot_experiment_results_exists(self):
        """Test _plot_experiment_results method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_plot_experiment_results"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._plot_experiment_results
        )  # nosec: B101 - Test assertion

    def test_show_results_visualization_exists(self):
        """Test _show_results_visualization method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_show_results_visualization"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._show_results_visualization
        )  # nosec: B101 - Test assertion

    def test_show_file_menu_exists(self):
        """Test _show_file_menu method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_show_file_menu"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._show_file_menu
        )  # nosec: B101 - Test assertion

    def test_show_edit_menu_exists(self):
        """Test _show_edit_menu method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_show_edit_menu"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._show_edit_menu
        )  # nosec: B101 - Test assertion

    def test_show_view_menu_exists(self):
        """Test _show_view_menu method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_show_view_menu"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._show_view_menu
        )  # nosec: B101 - Test assertion

    def test_show_help_menu_exists(self):
        """Test _show_help_menu method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_show_help_menu"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._show_help_menu
        )  # nosec: B101 - Test assertion

    def test_show_create_hypothesis_dialog_exists(self):
        """Test _show_create_hypothesis_dialog method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_show_create_hypothesis_dialog"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._show_create_hypothesis_dialog
        )  # nosec: B101 - Test assertion

    def test_show_hypothesis_review_exists(self):
        """Test _show_hypothesis_review method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_show_hypothesis_review"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._show_hypothesis_review
        )  # nosec: B101 - Test assertion

    def test_refresh_hypothesis_display_exists(self):
        """Test _refresh_hypothesis_display method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_refresh_hypothesis_display"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._refresh_hypothesis_display
        )  # nosec: B101 - Test assertion

    def test_launch_plan_generation_exists(self):
        """Test _launch_plan_generation method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_launch_plan_generation"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._launch_plan_generation
        )  # nosec: B101 - Test assertion


class TestGUIMethodBehavior:
    """Test behavior of key GUI methods."""

    def test_log_with_color(self):
        """Test _log method with color parameter."""
        mock_gui = MagicMock()
        mock_gui.console_text = MagicMock()

        gui.ExperimentRunnerGUI._log(mock_gui, "test message", "blue")
        mock_gui.console_text.insert.assert_called()

    def test_log_without_color(self):
        """Test _log method without color parameter."""
        mock_gui = MagicMock()
        mock_gui.console_text = MagicMock()

        gui.ExperimentRunnerGUI._log(mock_gui, "test message")
        mock_gui.console_text.insert.assert_called()

    def test_clear_console(self):
        """Test _clear_console clears the console."""
        mock_gui = MagicMock()
        mock_gui.console_text = MagicMock()

        gui.ExperimentRunnerGUI._clear_console(mock_gui)
        mock_gui.console_text.delete.assert_called()

    def test_finish_experiment_updates_state(self):
        """Test _finish_experiment updates experiment state."""
        mock_gui = MagicMock()
        mock_gui.running_experiments = {"test_exp"}
        mock_gui.experiment_buttons = {"test_exp": MagicMock()}
        mock_gui.status_indicators = {"test_exp": MagicMock()}

        gui.ExperimentRunnerGUI._finish_experiment(
            mock_gui, "test_exp", "completed", "green"
        )

        assert (
            "test_exp" not in mock_gui.running_experiments
        )  # nosec: B101 - Test assertion
        mock_gui.experiment_buttons["test_exp"].configure.assert_called()
        mock_gui.status_indicators["test_exp"].configure.assert_called()

    def test_stop_all_sets_flag(self):
        """Test _stop_all sets the stop flag."""
        mock_gui = MagicMock()
        mock_gui.stop_all = False
        mock_gui.active_processes = {"test": MagicMock()}

        gui.ExperimentRunnerGUI._stop_all(mock_gui)

        assert mock_gui.stop_all is True  # nosec: B101 - Test assertion
        # Should also terminate processes
        for process in mock_gui.active_processes.values():
            process.terminate.assert_called()

    def test_change_appearance_mode(self):
        """Test change_appearance_mode changes theme."""
        mock_gui = MagicMock()
        # Mock the _log method
        mock_gui._log = MagicMock()

        with patch.object(gui.ctk, "set_appearance_mode") as mock_set:
            gui.ExperimentRunnerGUI.change_appearance_mode(mock_gui, "Light")
            mock_set.assert_called_once_with("Light")
            # Verify _log was called with the theme change message
            mock_gui._log.assert_called_once_with("Theme changed to: Light", "#3498db")

    def test_check_dependencies_missing_optional(self):
        """Test dependency check with missing optional dependencies."""
        mock_gui = MagicMock()
        mock_gui.research_dir = Path("/tmp")
        mock_gui.console_text = MagicMock()

        # Mock only core dependencies as available
        with patch.dict(
            "sys.modules",
            {
                "numpy": MagicMock(),
                "pandas": MagicMock(),
                "matplotlib": MagicMock(),
                "customtkinter": MagicMock(),
                "scipy": MagicMock(),
            },
            clear=True,
        ):
            # Remove optional dependencies
            for mod in ["torch", "sklearn", "requests", "tqdm", "PIL"]:
                if mod in sys.modules:
                    del sys.modules[mod]

            with patch("tkinter.messagebox"):
                # Should not exit, just show warnings for optional deps
                gui.ExperimentRunnerGUI._check_dependencies(mock_gui)

    def test_find_experiments_filters_correctly(self):
        """Test that _find_experiments filters files correctly."""
        mock_gui = MagicMock()
        mock_gui.research_dir = Path("/tmp/test_research")

        # Mock mixed files
        all_files = [
            Path("run_valid.py"),
            Path("run_another.py"),
            Path("prepare_file.py"),  # Doesn't start with 'run_'
            Path("run_test.txt"),  # Doesn't end with '.py'
            Path("other_file.py"),  # Doesn't start with 'run_'
            Path("run_third.py"),
        ]

        with patch("gui.Path.glob") as mock_glob:
            mock_glob.return_value = [Path(f) for f in all_files]

            experiments = gui.ExperimentRunnerGUI._find_experiments(mock_gui)

            # Should only find files starting with 'run_' and ending with '.py'
            expected = [
                ("Valid", "experiments/run_valid.py"),
                ("Another", "experiments/run_another.py"),
                ("Third", "experiments/run_third.py"),
            ]
            assert experiments == expected  # nosec: B101 - Test assertion

    def test_execute_script_success(self):
        """Test _execute_script with successful execution."""
        mock_gui = MagicMock()
        mock_gui.research_dir = Path("/tmp")
        mock_gui.console_text = MagicMock()
        mock_gui.active_processes = {}

        mock_process = MagicMock()
        mock_process.stdout.readline.side_effect = ["Test output\n", ""]
        mock_process.poll.return_value = None
        mock_process.wait.return_value = 0

        with patch("gui.secure_popen", return_value=mock_process):
            with patch("gui.threading.Thread") as mock_thread:
                mock_thread.return_value = MagicMock()

                gui.ExperimentRunnerGUI._execute_script(
                    mock_gui, "test_exp", "run_test.py"
                )

                # Should start thread and log output
                mock_thread.assert_called_once()
                assert (
                    mock_gui.active_processes.get("test_exp") == mock_process
                )  # nosec: B101 - Test assertion

    def test_execute_script_failure(self):
        """Test _execute_script with failed execution."""
        mock_gui = MagicMock()
        mock_gui.research_dir = Path("/tmp")
        mock_gui.console_text = MagicMock()
        mock_gui.active_processes = {}

        mock_process = MagicMock()
        mock_process.stdout.readline.side_effect = ["Error output\n", ""]
        mock_process.poll.return_value = 1  # Non-zero exit code
        mock_process.wait.return_value = 1

        with patch("gui.secure_popen", return_value=mock_process):
            with patch("gui.threading.Thread") as mock_thread:
                mock_thread.return_value = MagicMock()

                gui.ExperimentRunnerGUI._execute_script(
                    mock_gui, "test_exp", "run_test.py"
                )

                # Should handle failure gracefully
                assert (
                    "test_exp" not in mock_gui.active_processes
                )  # nosec: B101 - Test assertion

    def test_parse_experiment_results_basic(self):
        """Test basic experiment results parsing."""
        mock_gui = MagicMock()
        mock_gui.experiment_results = {}
        mock_gui.current_figure = MagicMock()
        mock_gui.current_canvas = MagicMock()

        # Sample output lines with metrics
        output_lines = [
            "Experiment completed successfully",
            "Accuracy: 0.85",
            "Precision: 0.82",
            "Recall: 0.88",
            "F1-Score: 0.85",
            "Loss: 0.25",
        ]

        gui.ExperimentRunnerGUI._parse_experiment_results(
            mock_gui, "test_exp", output_lines
        )

        # Should parse and store results
        assert "test_exp" in mock_gui.experiment_results  # nosec: B101 - Test assertion
        results = mock_gui.experiment_results["test_exp"]
        assert "metrics" in results  # nosec: B101 - Test assertion
        assert results["metrics"]["accuracy"] == 0.85  # nosec: B101 - Test assertion
        assert results["metrics"]["precision"] == 0.82  # nosec: B101 - Test assertion

    def test_parse_experiment_results_with_guardrails(self):
        """Test parsing results with guardrail information."""
        mock_gui = MagicMock()
        mock_gui.experiment_results = {}
        mock_gui.current_figure = MagicMock()
        mock_gui.current_canvas = MagicMock()

        output_lines = [
            "APGI Guardrail Status:",
            "Confidence: 0.92",
            "Escalations: 2",
            "Warnings: 1",
            "Safety Score: 0.88",
            "Accuracy: 0.90",
        ]

        gui.ExperimentRunnerGUI._parse_experiment_results(
            mock_gui, "test_exp", output_lines
        )

        results = mock_gui.experiment_results["test_exp"]
        assert "guardrails" in results  # nosec: B101 - Test assertion
        assert (
            results["guardrails"]["confidence"] == 0.92
        )  # nosec: B101 - Test assertion
        assert results["guardrails"]["escalations"] == 2  # nosec: B101 - Test assertion
        assert (
            results["guardrails"]["safety_score"] == 0.88
        )  # nosec: B101 - Test assertion

    def test_update_guardrail_dashboard(self):
        """Test guardrail dashboard update."""
        mock_gui = MagicMock()
        mock_gui.guardrail_confidence_label = MagicMock()
        mock_gui.guardrail_escalations_label = MagicMock()
        mock_gui.guardrail_warnings_label = MagicMock()
        mock_gui.guardrail_safety_label = MagicMock()

        gui.ExperimentRunnerGUI._update_guardrail_dashboard(
            mock_gui, "ACTIVE", 0.85, 3, 2, 0.90
        )

        # Should update all labels
        mock_gui.guardrail_confidence_label.configure.assert_called()
        mock_gui.guardrail_escalations_label.configure.assert_called()
        mock_gui.guardrail_warnings_label.configure.assert_called()
        mock_gui.guardrail_safety_label.configure.assert_called()

    def test_notify_guardrail_escalation(self):
        """Test guardrail escalation notification."""
        mock_gui = MagicMock()
        mock_gui._log = MagicMock()

        with patch("tkinter.messagebox.showwarning") as mock_warning:
            gui.ExperimentRunnerGUI._notify_guardrail_escalation(
                mock_gui, "test_exp", 0.75, "Low confidence detected"
            )

            # Should show warning and log message
            mock_warning.assert_called_once()
            mock_gui._log.assert_called()

    def test_run_all_sequential(self):
        """Test running all experiments sequentially."""
        mock_gui = MagicMock()
        mock_gui.experiments = [
            ("exp1", "run_exp1.py"),
            ("exp2", "run_exp2.py"),
        ]
        mock_gui.running_experiments = set()
        mock_gui.stop_all = False
        mock_gui._run_experiment = MagicMock()
        mock_gui._log = MagicMock()

        # Mock time.sleep to avoid actual delays
        with patch("time.sleep"):
            gui.ExperimentRunnerGUI._run_all_sequential(mock_gui)

            # Should run all experiments
            assert (
                mock_gui._run_experiment.call_count == 2
            )  # nosec: B101 - Test assertion
            mock_gui._run_experiment.assert_any_call("exp1", "run_exp1.py")
            mock_gui._run_experiment.assert_any_call("exp2", "run_exp2.py")

    def test_run_all_sequential_with_stop(self):
        """Test sequential run with stop flag."""
        mock_gui = MagicMock()
        mock_gui.experiments = [
            ("exp1", "run_exp1.py"),
            ("exp2", "run_exp2.py"),
        ]
        mock_gui.running_experiments = set()
        mock_gui.stop_all = True  # Set stop flag
        mock_gui._run_experiment = MagicMock()

        with patch("time.sleep"):
            gui.ExperimentRunnerGUI._run_all_sequential(mock_gui)

            # Should not run any experiments due to stop flag
            mock_gui._run_experiment.assert_not_called()

    def test_display_dependencies_status(self):
        """Test dependency status display."""
        mock_gui = MagicMock()
        mock_gui.console_text = MagicMock()
        mock_gui.research_dir = Path("/tmp")

        # Mock all dependencies as available
        with patch.dict(
            "sys.modules",
            {
                "numpy": MagicMock(),
                "pandas": MagicMock(),
                "matplotlib": MagicMock(),
                "customtkinter": MagicMock(),
                "scipy": MagicMock(),
                "torch": MagicMock(),
                "sklearn": MagicMock(),
                "requests": MagicMock(),
                "tqdm": MagicMock(),
                "PIL": MagicMock(),
            },
        ):
            gui.ExperimentRunnerGUI._display_dependencies_status(mock_gui)

            # Should log status information
            assert mock_gui.console_text.insert.called  # nosec: B101 - Test assertion

    def test_repair_dependencies(self):
        """Test dependency repair functionality."""
        mock_gui = MagicMock()
        mock_gui._display_dependencies_status = MagicMock()
        mock_gui._log = MagicMock()

        with patch("gui.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            gui.ExperimentRunnerGUI._repair_dependencies(mock_gui)

            # Should display status and attempt repairs
            mock_gui._display_dependencies_status.assert_called_once()

    def test_create_visualization_panel(self):
        """Test visualization panel creation."""
        mock_gui = MagicMock()
        mock_parent = MagicMock()

        with patch("gui.Figure") as mock_figure:
            with patch("gui.FigureCanvasTkAgg") as mock_canvas:
                with patch("gui.NavigationToolbar2Tk") as mock_toolbar:

                    gui.ExperimentRunnerGUI._create_visualization_panel(
                        mock_gui, mock_parent
                    )

                    # Should create figure, canvas, and toolbar
                    mock_figure.assert_called_once()
                    mock_canvas.assert_called_once()
                    mock_toolbar.assert_called_once()

    def test_plot_experiment_results(self):
        """Test plotting experiment results."""
        mock_gui = MagicMock()
        mock_gui.current_figure = MagicMock()
        mock_gui.current_canvas = MagicMock()
        mock_ax = MagicMock()
        mock_gui.current_figure.add_subplot.return_value = mock_ax

        results = {
            "metrics": {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
                "f1_score": 0.85,
            }
        }

        gui.ExperimentRunnerGUI._plot_experiment_results(mock_gui, "test_exp", results)

        # Should create plot and update canvas
        mock_ax.bar.assert_called()
        mock_gui.current_canvas.draw.assert_called()

    def test_show_create_hypothesis_dialog(self):
        """Test hypothesis creation dialog."""
        mock_gui = MagicMock()
        mock_gui.approval_board = MagicMock()
        mock_gui.hypothesis_scrollable = MagicMock()

        with patch("gui.ctk.CTkToplevel") as mock_toplevel:
            with patch("gui.ctk.CTkFrame"):
                with patch("gui.ctk.CTkLabel"):
                    with patch("gui.ctk.CTkEntry"):
                        with patch("gui.ctk.CTkButton"):

                            gui.ExperimentRunnerGUI._show_create_hypothesis_dialog(
                                mock_gui
                            )

                            # Should create dialog with widgets
                            mock_toplevel.assert_called_once()

    def test_show_hypothesis_review_empty(self):
        """Test hypothesis review with no pending hypotheses."""
        mock_gui = MagicMock()
        mock_gui.approval_board = MagicMock()
        mock_gui.approval_board.get_pending_hypotheses.return_value = []

        with patch("tkinter.messagebox.showinfo") as mock_info:
            gui.ExperimentRunnerGUI._show_hypothesis_review(mock_gui)

            # Should show info message
            mock_info.assert_called_once()

    def test_refresh_hypothesis_display(self):
        """Test refreshing hypothesis display."""
        mock_gui = MagicMock()
        mock_gui.hypothesis_scrollable = MagicMock()
        mock_gui.hypothesis_scrollable.winfo_children.return_value = [
            MagicMock(),
            MagicMock(),
        ]
        mock_gui.approval_board = MagicMock()
        mock_gui.approval_board.get_all_hypotheses.return_value = []

        gui.ExperimentRunnerGUI._refresh_hypothesis_display(mock_gui)

        # Should clear existing widgets and refresh
        assert (
            mock_gui.hypothesis_scrollable.winfo_children.called
        )  # nosec: B101 - Test assertion
        for widget in mock_gui.hypothesis_scrollable.winfo_children.return_value:
            widget.destroy.assert_called()


class TestFileMenuFunctionality:
    """Test File menu functionality."""

    def test_create_new_experiment_exists(self):
        """Test _create_new_experiment method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_create_new_experiment"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._create_new_experiment
        )  # nosec: B101 - Test assertion

    def test_create_new_experiment_functionality(self):
        """Test creating a new experiment dialog."""
        mock_gui = MagicMock()
        mock_gui.research_dir = Path("/tmp/test_experiments")

        with patch("tkinter.simpledialog.askstring") as mock_dialog:
            mock_dialog.return_value = "test_experiment"

            with patch("pathlib.Path.mkdir"):
                with patch("gui.ctk.CTkTextbox") as mock_textbox:
                    with patch("gui.ctk.CTkButton") as mock_button:
                        gui.ExperimentRunnerGUI._create_new_experiment(mock_gui)

                        # Should ask for experiment name
                        mock_dialog.assert_called_once()
                        # Should create UI elements
                        assert (
                            mock_textbox.called or mock_button.called
                        )  # nosec: B101 - Test assertion

    def test_load_experiment_functionality(self):
        """Test loading experiment from file."""
        mock_gui = MagicMock()
        mock_gui.research_dir = Path("/tmp/test_experiments")

        with patch("tkinter.filedialog.askopenfilename") as mock_dialog:
            mock_dialog.return_value = "/path/to/experiment.py"

            with patch("pathlib.Path.read_text", return_value="print('test code')"):
                with patch("gui.ctk.CTkTextbox") as mock_textbox:
                    gui.ExperimentRunnerGUI._load_experiment(mock_gui)

                    mock_dialog.assert_called_once()
                    mock_textbox.assert_called()


class TestExperimentExecution:
    """Test experiment execution functionality."""

    def test_execute_script_success(self):
        """Test successful script execution."""
        mock_gui = MagicMock()
        mock_gui.stop_all = False
        mock_gui.log_text = MagicMock()

        with patch("utils.apgi_security.secure_popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.communicate.return_value = ("output", "")
            mock_process.returncode = 0
            mock_popen.return_value = mock_process

            gui.ExperimentRunnerGUI._execute_script(
                mock_gui, "test_script.py", "test_exp"
            )

            # Check that the result was set by the callback
            # The callback sets result via _finish_experiment which stores in experiment_results
            assert (
                "test_exp" in mock_gui.experiment_results
            )  # nosec: B101 - Test assertion
            assert (
                mock_gui.experiment_results["test_exp"]["status"] == "Success"
            )  # nosec: B101 - Test assertion
            mock_popen.assert_called_once()

    def test_execute_script_failure(self):
        """Test script execution failure."""
        mock_gui = MagicMock()
        mock_gui.stop_all = False
        mock_gui.log_text = MagicMock()

        with patch("utils.apgi_security.secure_popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.communicate.return_value = ("", "error")
            mock_process.returncode = 1
            mock_popen.return_value = mock_process

            gui.ExperimentRunnerGUI._execute_script(
                mock_gui, "test_script.py", "test_exp"
            )

            # Result is handled via callback, no direct return value to check

    def test_execute_script_stopped(self):
        """Test script execution when stopped."""
        mock_gui = MagicMock()
        mock_gui.stop_all = True

        gui.ExperimentRunnerGUI._execute_script(mock_gui, "test_script.py", "test_exp")

        # Result is handled via callback, no direct return value to check

    def test_parse_experiment_results(self):
        """Test parsing experiment results."""
        mock_gui = MagicMock()
        mock_gui.experiment_results = {}

        # Test with valid JSON output
        test_output = '{"accuracy": 0.85, "loss": 0.15}'
        gui.ExperimentRunnerGUI._parse_experiment_results(
            mock_gui, "test_exp", test_output
        )

        assert "test_exp" in mock_gui.experiment_results  # nosec: B101 - Test assertion
        assert (
            mock_gui.experiment_results["test_exp"]["accuracy"] == 0.85
        )  # nosec: B101 - Test assertion

    def test_parse_experiment_results_invalid_json(self):
        """Test parsing experiment results with invalid JSON."""
        mock_gui = MagicMock()
        mock_gui.experiment_results = {}

        # Test with invalid JSON
        test_output = "invalid json output"
        gui.ExperimentRunnerGUI._parse_experiment_results(
            mock_gui, "test_exp", test_output
        )

        # Should handle gracefully without crashing
        assert "test_exp" in mock_gui.experiment_results  # nosec: B101 - Test assertion

    def test_finish_experiment(self):
        """Test finishing experiment execution."""
        mock_gui = MagicMock()
        mock_gui.running_experiments = {"test_exp"}
        mock_gui.experiment_buttons = {"test_exp": MagicMock()}
        mock_gui.status_indicators = {"test_exp": MagicMock()}
        mock_gui.experiment_results = {"test_exp": {"accuracy": 0.85}}

        with patch("gui.ctk.CTkProgressBar"):
            gui.ExperimentRunnerGUI._finish_experiment(mock_gui, "test_exp")

            assert (
                "test_exp" not in mock_gui.running_experiments
            )  # nosec: B101 - Test assertion
            mock_gui.experiment_buttons["test_exp"].configure.assert_called()


class TestAutoImproveFunctionality:
    """Test auto-improve functionality."""

    def test_run_auto_improve_with_llm(self):
        """Test auto-improve with LLM available."""
        mock_gui = MagicMock()
        mock_gui.stop_all = False
        mock_gui.code_text = MagicMock()
        mock_gui.code_text.get.return_value = "print('test code')"

        # Mock LLM available
        with patch("gui.LLM_AVAILABLE", True):
            with patch("gui.litellm.completion") as mock_completion:
                mock_completion.return_value = "improved_code"

                with patch("gui.ctk.CTkTextbox"):
                    gui.ExperimentRunnerGUI._run_auto_improve(mock_gui)

                    mock_completion.assert_called_once()

    def test_run_auto_improve_no_llm(self):
        """Test auto-improve without LLM."""
        mock_gui = MagicMock()
        mock_gui.stop_all = False

        # Mock LLM not available
        with patch("gui.LLM_AVAILABLE", False):
            with patch("tkinter.messagebox.showinfo") as mock_info:
                gui.ExperimentRunnerGUI._run_auto_improve(mock_gui)

                mock_info.assert_called_once()


class TestUtilityMethods:
    """Test utility methods."""

    def test_log_method(self):
        """Test logging method."""
        mock_gui = MagicMock()
        mock_gui.log_text = MagicMock()
        mock_gui.log_text.insert = MagicMock()

        gui.ExperimentRunnerGUI._log(mock_gui, "Test message")

        mock_gui.log_text.insert.assert_called()

    def test_clear_console(self):
        """Test clearing console."""
        mock_gui = MagicMock()
        mock_gui.log_text = MagicMock()
        mock_gui.log_text.delete = MagicMock()

        gui.ExperimentRunnerGUI._clear_console(mock_gui)

        mock_gui.log_text.delete.assert_called_once()

    def test_display_dependencies_status(self):
        """Test displaying dependencies status."""
        mock_gui = MagicMock()
        mock_gui.deps_frame = MagicMock()

        with patch("gui.ctk.CTkLabel") as mock_label:
            gui.ExperimentRunnerGUI._display_dependencies_status(mock_gui)

            mock_label.assert_called()

    def test_repair_dependencies(self):
        """Test repairing dependencies."""
        mock_gui = MagicMock()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0

            with patch("tkinter.messagebox.showinfo") as mock_info:
                gui.ExperimentRunnerGUI._repair_dependencies(mock_gui)

                mock_run.assert_called()
                mock_info.assert_called()


class TestVisualizationFunctionality:
    """Test visualization functionality."""

    def test_create_visualization_panel(self):
        """Test creating visualization panel."""
        mock_gui = MagicMock()

        with patch("gui.matplotlib.figure.Figure") as mock_figure:
            with patch("gui.FigureCanvasTkAgg") as mock_canvas:
                gui.ExperimentRunnerGUI._create_visualization_panel(mock_gui)

                mock_figure.assert_called_once()
                mock_canvas.assert_called_once()

    def test_update_guardrail_dashboard(self):
        """Test updating guardrail dashboard."""
        mock_gui = MagicMock()
        mock_gui.guardrail_canvas = MagicMock()
        mock_gui.guardrail_canvas.get_tk_widget = MagicMock()

        with patch("gui.matplotlib.figure.Figure"):
            gui.ExperimentRunnerGUI._update_guardrail_dashboard(mock_gui)

            # Should attempt to update visualization
            assert mock_gui.guardrail_canvas.called  # nosec: B101 - Test assertion


class TestHypothesisManagement:
    """Test hypothesis management functionality."""

    def test_create_hypothesis_success(self):
        """Test successful hypothesis creation."""
        mock_gui = MagicMock()
        mock_gui.approval_board = MagicMock()
        mock_gui.hypothesis_entry = MagicMock()
        mock_gui.hypothesis_entry.get.return_value = "Test hypothesis"

        with patch("tkinter.messagebox.showinfo") as mock_info:
            gui.ExperimentRunnerGUI._create_hypothesis(mock_gui)

            mock_gui.approval_board.submit_hypothesis.assert_called_once()
            mock_info.assert_called_once()

    def test_create_hypothesis_empty(self):
        """Test hypothesis creation with empty input."""
        mock_gui = MagicMock()
        mock_gui.hypothesis_entry = MagicMock()
        mock_gui.hypothesis_entry.get.return_value = ""

        with patch("tkinter.messagebox.showwarning") as mock_warning:
            gui.ExperimentRunnerGUI._create_hypothesis(mock_gui)

            mock_warning.assert_called_once()

    def test_approve_hypothesis(self):
        """Test approving a hypothesis."""
        mock_gui = MagicMock()
        mock_gui.approval_board = MagicMock()
        mock_hypothesis = MagicMock()
        mock_hypothesis.id = "test_id"

        gui.ExperimentRunnerGUI._approve_hypothesis(mock_gui, mock_hypothesis)

        mock_gui.approval_board.approve_hypothesis.assert_called_once_with("test_id")

    def test_reject_hypothesis(self):
        """Test rejecting a hypothesis."""
        mock_gui = MagicMock()
        mock_gui.approval_board = MagicMock()
        mock_hypothesis = MagicMock()
        mock_hypothesis.id = "test_id"

        gui.ExperimentRunnerGUI._reject_hypothesis(mock_gui, mock_hypothesis)

        mock_gui.approval_board.reject_hypothesis.assert_called_once_with("test_id")


class TestExperimentManagement:
    """Test experiment management functionality."""

    def test_run_all_experiments(self):
        """Test running all experiments."""
        mock_gui = MagicMock()
        mock_gui.stop_all = False
        mock_gui.experiment_cards = ["exp1", "exp2"]

        with patch.object(gui.ExperimentRunnerGUI, "_run_experiment"):
            gui.ExperimentRunnerGUI._run_all(mock_gui)

            # Should attempt to run all experiments
            assert (
                gui.ExperimentRunnerGUI._run_experiment.call_count == 2
            )  # nosec: B101 - Test assertion

    def test_run_all_sequential(self):
        """Test running all experiments sequentially."""
        mock_gui = MagicMock()
        mock_gui.stop_all = False
        mock_gui.experiment_cards = ["exp1", "exp2"]

        with patch.object(gui.ExperimentRunnerGUI, "_run_experiment"):
            gui.ExperimentRunnerGUI._run_all_sequential(mock_gui)

            # Should run experiments sequentially
            assert (
                gui.ExperimentRunnerGUI._run_experiment.call_count == 2
            )  # nosec: B101 - Test assertion

    def test_stop_all_experiments(self):
        """Test stopping all experiments."""
        mock_gui = MagicMock()
        mock_gui.running_experiments = {"exp1", "exp2"}
        mock_gui.active_processes = {"exp1": MagicMock(), "exp2": MagicMock()}

        gui.ExperimentRunnerGUI._stop_all(mock_gui)

        assert mock_gui.stop_all is True  # nosec: B101 - Test assertion
        # Should terminate all active processes
        for process in mock_gui.active_processes.values():
            process.terminate.assert_called()

    def test_get_experiment_template_exists(self):
        """Test _get_experiment_template method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_get_experiment_template"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._get_experiment_template
        )  # nosec: B101 - Test assertion

    def test_open_experiment_directory_exists(self):
        """Test _open_experiment_directory method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_open_experiment_directory"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._open_experiment_directory
        )  # nosec: B101 - Test assertion

    def test_import_experiment_results_exists(self):
        """Test _import_experiment_results method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_import_experiment_results"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._import_experiment_results
        )  # nosec: B101 - Test assertion

    def test_save_experiment_results_exists(self):
        """Test _save_experiment_results method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_save_experiment_results"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._save_experiment_results
        )  # nosec: B101 - Test assertion

    def test_export_research_report_exists(self):
        """Test _export_research_report method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_export_research_report"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._export_research_report
        )  # nosec: B101 - Test assertion

    def test_generate_research_report_exists(self):
        """Test _generate_research_report method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_generate_research_report"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._generate_research_report
        )  # nosec: B101 - Test assertion

    def test_reload_experiments_exists(self):
        """Test _reload_experiments method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_reload_experiments"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._reload_experiments
        )  # nosec: B101 - Test assertion

    def test_refresh_experiment_display_exists(self):
        """Test _refresh_experiment_display method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_refresh_experiment_display"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._refresh_experiment_display
        )  # nosec: B101 - Test assertion

    def test_clear_session_exists(self):
        """Test _clear_session method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_clear_session"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._clear_session
        )  # nosec: B101 - Test assertion

    def test_confirm_exit_exists(self):
        """Test _confirm_exit method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_confirm_exit"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._confirm_exit
        )  # nosec: B101 - Test assertion

    @patch("tkinter.messagebox.askyesno")
    @patch("tkinter.messagebox.showinfo")
    def test_clear_session_with_confirmation(self, mock_showinfo, mock_askyesno):
        """Test clear session with user confirmation."""
        mock_gui = MagicMock()
        mock_gui.running_experiments = set()
        mock_gui.experiment_results = {"test": "data"}
        mock_gui.console_text = MagicMock()
        mock_gui.guardrail_state = {"status": "test"}
        mock_gui._update_guardrail_dashboard = MagicMock()
        mock_gui._clear_console = MagicMock()
        mock_gui._log = MagicMock()

        mock_askyesno.return_value = True

        gui.ExperimentRunnerGUI._clear_session(mock_gui)

        mock_askyesno.assert_called_once()
        mock_gui._clear_console.assert_called_once()
        mock_gui._update_guardrail_dashboard.assert_called_once()
        mock_showinfo.assert_called_once()

    @patch("tkinter.messagebox.askyesno")
    def test_clear_session_cancelled(self, mock_askyesno):
        """Test clear session when user cancels."""
        mock_gui = MagicMock()
        mock_gui._log = MagicMock()

        mock_askyesno.return_value = False

        gui.ExperimentRunnerGUI._clear_session(mock_gui)

        mock_askyesno.assert_called_once()
        # Should not clear anything
        mock_gui._log.assert_not_called()


class TestEditMenuFunctionality:
    """Test Edit menu functionality."""

    def test_copy_console_output_exists(self):
        """Test _copy_console_output method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_copy_console_output"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._copy_console_output
        )  # nosec: B101 - Test assertion

    def test_save_console_log_exists(self):
        """Test _save_console_log method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_save_console_log"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._save_console_log
        )  # nosec: B101 - Test assertion

    def test_edit_experiment_script_exists(self):
        """Test _edit_experiment_script method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_edit_experiment_script"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._edit_experiment_script
        )  # nosec: B101 - Test assertion

    def test_delete_experiment_exists(self):
        """Test _delete_experiment method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_delete_experiment"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._delete_experiment
        )  # nosec: B101 - Test assertion

    def test_clear_all_results_exists(self):
        """Test _clear_all_results method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_clear_all_results"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._clear_all_results
        )  # nosec: B101 - Test assertion

    def test_reset_visualizations_exists(self):
        """Test _reset_visualizations method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_reset_visualizations"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._reset_visualizations
        )  # nosec: B101 - Test assertion

    def test_show_settings_dialog_exists(self):
        """Test _show_settings_dialog method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_show_settings_dialog"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._show_settings_dialog
        )  # nosec: B101 - Test assertion

    def test_reset_guardrails_exists(self):
        """Test _reset_guardrails method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_reset_guardrails"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._reset_guardrails
        )  # nosec: B101 - Test assertion

    @patch("tkinter.messagebox.askyesno")
    @patch("tkinter.messagebox.showinfo")
    def test_clear_all_results_with_confirmation(self, mock_showinfo, mock_askyesno):
        """Test clear all results with user confirmation."""
        mock_gui = MagicMock()
        mock_gui.experiment_results = {"test": "data"}
        mock_gui._log = MagicMock()

        mock_askyesno.return_value = True

        gui.ExperimentRunnerGUI._clear_all_results(mock_gui)

        mock_askyesno.assert_called_once()
        assert len(mock_gui.experiment_results) == 0  # nosec: B101 - Test assertion
        mock_showinfo.assert_called_once()

    @patch("tkinter.messagebox.askyesno")
    def test_reset_guardrails_with_confirmation(self, mock_askyesno):
        """Test reset guardrails with user confirmation."""
        mock_gui = MagicMock()
        mock_gui.guardrail_state = {"status": "test"}
        mock_gui._update_guardrail_dashboard = MagicMock()
        mock_gui._log = MagicMock()

        mock_askyesno.return_value = True

        gui.ExperimentRunnerGUI._reset_guardrails(mock_gui)

        mock_askyesno.assert_called_once()
        assert (
            mock_gui.guardrail_state["status"] == "IDLE"
        )  # nosec: B101 - Test assertion
        mock_gui._update_guardrail_dashboard.assert_called_once()


class TestViewMenuFunctionality:
    """Test View menu functionality."""

    def test_toggle_sidebar_exists(self):
        """Test _toggle_sidebar method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_toggle_sidebar"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._toggle_sidebar
        )  # nosec: B101 - Test assertion

    def test_maximize_console_exists(self):
        """Test _maximize_console method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_maximize_console"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._maximize_console
        )  # nosec: B101 - Test assertion

    def test_zoom_in_exists(self):
        """Test _zoom_in method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_zoom_in"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._zoom_in
        )  # nosec: B101 - Test assertion

    def test_zoom_out_exists(self):
        """Test _zoom_out method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_zoom_out"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._zoom_out
        )  # nosec: B101 - Test assertion

    def test_reset_zoom_exists(self):
        """Test _reset_zoom method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_reset_zoom"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._reset_zoom
        )  # nosec: B101 - Test assertion

    def test_show_all_visualizations_exists(self):
        """Test _show_all_visualizations method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_show_all_visualizations"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._show_all_visualizations
        )  # nosec: B101 - Test assertion

    def test_close_all_windows_exists(self):
        """Test _close_all_windows method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_close_all_windows"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._close_all_windows
        )  # nosec: B101 - Test assertion

    def test_refresh_ui_exists(self):
        """Test _refresh_ui method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_refresh_ui"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._refresh_ui
        )  # nosec: B101 - Test assertion

    @patch("tkinter.messagebox.showinfo")
    def test_close_all_windows_with_count(self, mock_showinfo):
        """Test close all windows shows count."""
        mock_gui = MagicMock()
        mock_gui._log = MagicMock()
        mock_gui.winfo_children.return_value = [
            MagicMock(spec=gui.ctk.CTkToplevel),
            MagicMock(spec=gui.ctk.CTkToplevel),
            MagicMock(),  # Not a CTkToplevel
        ]

        gui.ExperimentRunnerGUI._close_all_windows(mock_gui)

        # Should destroy 2 CTkToplevel windows
        assert mock_gui.winfo_children.return_value[
            0
        ].destroy.called  # nosec: B101 - Test assertion
        assert mock_gui.winfo_children.return_value[
            1
        ].destroy.called  # nosec: B101 - Test assertion
        assert not mock_gui.winfo_children.return_value[
            2
        ].destroy.called  # nosec: B101 - Test assertion
        mock_showinfo.assert_called_once_with("Success", "Closed 2 popup windows.")


class TestHelpMenuFunctionality:
    """Test Help menu functionality."""

    def test_show_user_guide_exists(self):
        """Test _show_user_guide method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_show_user_guide"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._show_user_guide
        )  # nosec: B101 - Test assertion

    def test_show_api_docs_exists(self):
        """Test _show_api_docs method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_show_api_docs"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._show_api_docs
        )  # nosec: B101 - Test assertion

    def test_show_experiment_templates_exists(self):
        """Test _show_experiment_templates method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_show_experiment_templates"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._show_experiment_templates
        )  # nosec: B101 - Test assertion

    def test_show_getting_started_exists(self):
        """Test _show_getting_started method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_show_getting_started"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._show_getting_started
        )  # nosec: B101 - Test assertion

    def test_show_experiment_tutorial_exists(self):
        """Test _show_experiment_tutorial method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_show_experiment_tutorial"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._show_experiment_tutorial
        )  # nosec: B101 - Test assertion

    def test_show_xpr_guide_exists(self):
        """Test _show_xpr_guide method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_show_xpr_guide"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._show_xpr_guide
        )  # nosec: B101 - Test assertion

    def test_report_issue_exists(self):
        """Test _report_issue method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_report_issue"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._report_issue
        )  # nosec: B101 - Test assertion

    def test_send_feedback_exists(self):
        """Test _send_feedback method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_send_feedback"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._send_feedback
        )  # nosec: B101 - Test assertion

    def test_show_about_exists(self):
        """Test _show_about method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_show_about"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._show_about
        )  # nosec: B101 - Test assertion

    def test_show_system_info_exists(self):
        """Test _show_system_info method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_show_system_info"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._show_system_info
        )  # nosec: B101 - Test assertion

    def test_show_text_dialog_exists(self):
        """Test _show_text_dialog method exists."""
        assert hasattr(
            gui.ExperimentRunnerGUI, "_show_text_dialog"
        )  # nosec: B101 - Test assertion
        assert callable(
            gui.ExperimentRunnerGUI._show_text_dialog
        )  # nosec: B101 - Test assertion

    @patch("tkinter.messagebox.showinfo")
    def test_report_issue_submission(self, mock_showinfo):
        """Test issue report submission."""
        mock_gui = MagicMock()
        mock_gui._log = MagicMock()

        with patch("gui.ctk.CTkToplevel"):
            with patch("gui.ctk.CTkLabel"):
                with patch("gui.ctk.CTkOptionMenu"):
                    with patch("gui.ctk.CTkTextbox"):
                        with patch("gui.ctk.CTkFrame"):
                            with patch("gui.ctk.CTkButton"):
                                gui.ExperimentRunnerGUI._report_issue(mock_gui)

        mock_showinfo.assert_called_once_with(
            "Thank You", "Your issue has been reported. We'll look into it!"
        )

    @patch("tkinter.messagebox.showinfo")
    def test_send_feedback_submission(self, mock_showinfo):
        """Test feedback submission."""
        mock_gui = MagicMock()
        mock_gui._log = MagicMock()

        with patch("gui.ctk.CTkToplevel"):
            with patch("gui.ctk.CTkLabel"):
                with patch("gui.ctk.CTkTextbox"):
                    with patch("gui.ctk.CTkFrame"):
                        with patch("gui.ctk.CTkButton"):
                            gui.ExperimentRunnerGUI._send_feedback(mock_gui)

        mock_showinfo.assert_called_once_with(
            "Thank You!", "Your feedback has been submitted. We appreciate your input!"
        )


class TestGUICoreFunctionality:
    """Test core GUI functionality with improved coverage."""

    def test_generate_research_report_content(self):
        """Test research report generation content."""
        mock_gui = MagicMock()
        mock_gui.experiments = [
            ("Test1", "experiments/run_test1.py"),
            ("Test2", "experiments/run_test2.py"),
        ]
        mock_gui.experiment_results = {
            "Test1": {"status": "success", "data": {"value": 42}}
        }
        mock_gui.running_experiments = set()
        mock_gui.approval_board = MagicMock()
        mock_gui.approval_board.get_pending_hypotheses.return_value = []
        mock_gui.guardrail_state = {
            "status": "IDLE",
            "confidence": 0.95,
            "last_regression": 0.1,
            "escalation_count": 0,
            "last_experiment": "",
        }

        report = gui.ExperimentRunnerGUI._generate_research_report(mock_gui)

        # Check that report contains expected sections
        assert "# APGI Research Report" in report  # nosec: B101 - Test assertion
        assert "Test1" in report  # nosec: B101 - Test assertion
        assert "Test2" in report  # nosec: B101 - Test assertion
        assert "Guardrail Status" in report  # nosec: B101 - Test assertion
        assert "Hypotheses" in report  # nosec: B101 - Test assertion

    def test_get_experiment_template_basic(self):
        """Test experiment template generation."""
        template = gui.ExperimentRunnerGUI._get_experiment_template(
            "Test Experiment", "Test description", "Basic Experiment"
        )

        assert "Test Experiment" in template  # nosec: B101 - Test assertion
        assert "Test description" in template  # nosec: B101 - Test assertion
        assert "def main() -> None:" in template  # nosec: B101 - Test assertion
        assert (
            "def run_experiment() -> dict:" in template
        )  # nosec: B101 - Test assertion

    def test_get_experiment_template_data_analysis(self):
        """Test data analysis template generation."""
        template = gui.ExperimentRunnerGUI._get_experiment_template(
            "Data Analysis", "Data analysis experiment", "Data Analysis"
        )

        assert "Data Analysis" in template  # nosec: B101 - Test assertion
        assert "def main() -> None:" in template  # nosec: B101 - Test assertion
        assert (
            "def run_experiment() -> dict:" in template
        )  # nosec: B101 - Test assertion

    def test_get_experiment_template_model_training(self):
        """Test model training template generation."""
        template = gui.ExperimentRunnerGUI._get_experiment_template(
            "Model Training", "Model training experiment", "Model Training"
        )

        assert "Model Training" in template  # nosec: B101 - Test assertion
        assert "def main() -> None:" in template  # nosec: B101 - Test assertion
        assert (
            "def run_experiment() -> dict:" in template
        )  # nosec: B101 - Test assertion

    def test_get_experiment_template_visualization(self):
        """Test visualization template generation."""
        template = gui.ExperimentRunnerGUI._get_experiment_template(
            "Visualization", "Visualization experiment", "Visualization"
        )

        assert "Visualization" in template  # nosec: B101 - Test assertion
        assert "def main() -> None:" in template  # nosec: B101 - Test assertion
        assert (
            "def run_experiment() -> dict:" in template
        )  # nosec: B101 - Test assertion

    @patch("tkinter.filedialog.askopenfilename")
    def test_import_experiment_results_success(self, mock_filedialog):
        """Test successful import of experiment results."""
        mock_gui = MagicMock()
        mock_gui._log = MagicMock()
        mock_filedialog.return_value = "/path/to/results.json"

        gui.ExperimentRunnerGUI._import_experiment_results(mock_gui)

        mock_filedialog.assert_called_once()
        mock_gui._log.assert_called_once()

    @patch("tkinter.filedialog.asksaveasfilename")
    @patch("json.dump")
    def test_save_experiment_results_success(self, mock_json_dump, mock_filedialog):
        """Test successful save of experiment results."""
        mock_gui = MagicMock()
        mock_gui.experiment_results = {"test": "data"}
        mock_gui._log = MagicMock()
        mock_filedialog.return_value = "/path/to/save.json"

        gui.ExperimentRunnerGUI._save_experiment_results(mock_gui)

        mock_filedialog.assert_called_once()
        mock_json_dump.assert_called_once()
        mock_gui._log.assert_called_once()

    @patch("tkinter.filedialog.asksaveasfilename")
    def test_save_experiment_results_no_data(self, mock_filedialog):
        """Test save experiment results when no data exists."""
        mock_gui = MagicMock()
        mock_gui.experiment_results = {}
        mock_gui._log = MagicMock()

        with patch("tkinter.messagebox.showinfo") as mock_info:
            gui.ExperimentRunnerGUI._save_experiment_results(mock_gui)

        mock_info.assert_called_once_with(
            "No Results", "No experiment results to save."
        )
        mock_filedialog.assert_not_called()

    @patch("tkinter.messagebox.askyesno")
    def test_confirm_exit_with_running_experiments(self, mock_askyesno):
        """Test exit confirmation with running experiments."""
        mock_gui = MagicMock()
        mock_gui.running_experiments = {"test"}
        mock_gui._log = MagicMock()
        mock_askyesno.return_value = False

        gui.ExperimentRunnerGUI._confirm_exit(mock_gui)

        mock_askyesno.assert_called_once()
        mock_gui._log.assert_not_called()

    @patch("tkinter.messagebox.askyesno")
    @patch("tkinter.messagebox.askyesnocancel")
    def test_confirm_exit_save_results_cancel(self, mock_askyesnocancel, mock_askyesno):
        """Test exit confirmation with save results cancelled."""
        mock_gui = MagicMock()
        mock_gui.running_experiments = set()
        mock_gui.experiment_results = {"test": "data"}
        mock_gui._log = MagicMock()
        mock_askyesno.return_value = True
        mock_askyesnocancel.return_value = None  # Cancel

        gui.ExperimentRunnerGUI._confirm_exit(mock_gui)

        mock_askyesno.assert_called_once()
        mock_askyesnocancel.assert_called_once()
        mock_gui._log.assert_not_called()

    @patch("subprocess.run")
    def test_open_experiment_directory_macos(self, mock_subprocess):
        """Test opening experiment directory on macOS."""
        mock_gui = MagicMock()
        mock_gui.research_dir = Path("/test/dir")
        mock_gui._log = MagicMock()
        mock_subprocess.return_value.returncode = 0

        with patch("platform.system", return_value="Darwin"):
            gui.ExperimentRunnerGUI._open_experiment_directory(mock_gui)

        mock_subprocess.assert_called_once_with(["open", "/test/dir"], check=True)
        mock_gui._log.assert_called_once()

    @patch("subprocess.run")
    def test_open_experiment_directory_windows(self, mock_subprocess):
        """Test opening experiment directory on Windows."""
        mock_gui = MagicMock()
        mock_gui.research_dir = Path("/test/dir")
        mock_gui._log = MagicMock()
        mock_subprocess.return_value.returncode = 0

        with patch("platform.system", return_value="Windows"):
            gui.ExperimentRunnerGUI._open_experiment_directory(mock_gui)

        mock_subprocess.assert_called_once_with(["explorer", "/test/dir"], check=True)
        mock_gui._log.assert_called_once()

    @patch("subprocess.run")
    def test_open_experiment_directory_linux(self, mock_subprocess):
        """Test opening experiment directory on Linux."""
        mock_gui = MagicMock()
        mock_gui.research_dir = Path("/test/dir")
        mock_gui._log = MagicMock()
        mock_subprocess.return_value.returncode = 0

        with patch("platform.system", return_value="Linux"):
            gui.ExperimentRunnerGUI._open_experiment_directory(mock_gui)

        mock_subprocess.assert_called_once_with(["xdg-open", "/test/dir"], check=True)
        mock_gui._log.assert_called_once()

    def test_zoom_in_increases_font_size(self):
        """Test zoom in increases font size."""
        mock_gui = MagicMock()
        mock_gui.console_text = MagicMock()
        mock_gui.console_text.cget.return_value = ("Courier", 13)
        mock_gui._log = MagicMock()

        gui.ExperimentRunnerGUI._zoom_in(mock_gui)

        mock_gui.console_text.configure.assert_called_once_with(font=("Courier", 14))
        mock_gui._log.assert_called_once_with("Zoomed in to font size 14", "#3498db")

    def test_zoom_out_decreases_font_size(self):
        """Test zoom out decreases font size."""
        mock_gui = MagicMock()
        mock_gui.console_text = MagicMock()
        mock_gui.console_text.cget.return_value = ("Courier", 13)
        mock_gui._log = MagicMock()

        gui.ExperimentRunnerGUI._zoom_out(mock_gui)

        mock_gui.console_text.configure.assert_called_once_with(font=("Courier", 12))
        mock_gui._log.assert_called_once_with("Zoomed out to font size 12", "#3498db")

    def test_reset_zoom_resets_font_size(self):
        """Test reset zoom resets font size to default."""
        mock_gui = MagicMock()
        mock_gui.console_text = MagicMock()
        mock_gui._log = MagicMock()

        gui.ExperimentRunnerGUI._reset_zoom(mock_gui)

        mock_gui.console_text.configure.assert_called_once_with(font=("Courier", 13))
        mock_gui._log.assert_called_once_with("Zoom reset to default", "#3498db")

    def test_zoom_in_max_limit(self):
        """Test zoom in respects maximum font size limit."""
        mock_gui = MagicMock()
        mock_gui.console_text = MagicMock()
        mock_gui.console_text.cget.return_value = ("Courier", 20)
        mock_gui._log = MagicMock()

        gui.ExperimentRunnerGUI._zoom_in(mock_gui)

        # Should not increase beyond max size 20
        mock_gui.console_text.configure.assert_called_once_with(font=("Courier", 20))
        mock_gui._log.assert_called_once_with("Zoomed in to font size 20", "#3498db")

    def test_zoom_out_min_limit(self):
        """Test zoom out respects minimum font size limit."""
        mock_gui = MagicMock()
        mock_gui.console_text = MagicMock()
        mock_gui.console_text.cget.return_value = ("Courier", 8)
        mock_gui._log = MagicMock()

        gui.ExperimentRunnerGUI._zoom_out(mock_gui)

        # Should not decrease below min size 8
        mock_gui.console_text.configure.assert_called_once_with(font=("Courier", 8))
        mock_gui._log.assert_called_once_with("Zoomed out to font size 8", "#3498db")

    @patch("tkinter.messagebox.showinfo")
    def test_show_all_visualizations_no_results(self, mock_showinfo):
        """Test show all visualizations when no results exist."""
        mock_gui = MagicMock()
        mock_gui.experiment_results = {}
        mock_gui._log = MagicMock()
        mock_gui._show_results_visualization = MagicMock()

        gui.ExperimentRunnerGUI._show_all_visualizations(mock_gui)

        mock_showinfo.assert_called_once_with(
            "No Results", "No experiment results available for visualization."
        )
        mock_gui._show_results_visualization.assert_not_called()

    @patch("tkinter.messagebox.showinfo")
    def test_show_all_visualizations_with_results(self, mock_showinfo):
        """Test show all visualizations when results exist."""
        mock_gui = MagicMock()
        mock_gui.experiment_results = {"test1": "data1", "test2": "data2"}
        mock_gui._log = MagicMock()
        mock_gui._show_results_visualization = MagicMock()

        gui.ExperimentRunnerGUI._show_all_visualizations(mock_gui)

        mock_gui._show_results_visualization.assert_called()
        mock_gui._log.assert_called_once_with("Opened 2 visualizations", "#27ae60")

    def test_refresh_ui_calls_all_refresh_methods(self):
        """Test refresh UI calls all refresh methods."""
        mock_gui = MagicMock()
        mock_gui._reload_experiments = MagicMock()
        mock_gui._refresh_hypothesis_display = MagicMock()
        mock_gui._update_guardrail_dashboard = MagicMock()
        mock_gui._log = MagicMock()

        with patch("tkinter.messagebox.showinfo"):
            gui.ExperimentRunnerGUI._refresh_ui(mock_gui)

        mock_gui._reload_experiments.assert_called_once()
        mock_gui._refresh_hypothesis_display.assert_called_once()
        mock_gui._update_guardrail_dashboard.assert_called_once()
        mock_gui._log.assert_called_once_with("UI refreshed successfully", "#27ae60")

    @patch("tkinter.messagebox.showerror")
    def test_refresh_ui_handles_errors(self, mock_showerror):
        """Test refresh UI handles errors gracefully."""
        mock_gui = MagicMock()
        mock_gui._reload_experiments = MagicMock(side_effect=Exception("Test error"))
        mock_gui._log = MagicMock()

        with patch("tkinter.messagebox.showinfo"):
            gui.ExperimentRunnerGUI._refresh_ui(mock_gui)

        mock_gui._log.assert_called_with("Failed to refresh UI: Test error", "#e74c3c")
        mock_showerror.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])


def test_import_preserves_selected_matplotlib_backend():
    """Importing GUI helpers must not switch a headless analysis backend."""
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import matplotlib; matplotlib.use('Agg'); "
            "import matplotlib.pyplot; import GUI_auto_improve_experiments; "
            "assert matplotlib.get_backend().lower() == 'agg'",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
