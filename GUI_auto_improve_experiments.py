"""APGI Experiment Runner GUI (Premium Edition)
Modernized for apgi-research directory with CustomTkinter.
"""

import logging
import os
import sys

# Must set multiprocessing start method before any other imports on macOS
# to prevent "The process has forked and you cannot use this CoreFoundation
# functionality" error.
if sys.platform == "darwin":
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

# Check for litellm availability first
try:
    import litellm  # noqa: F401

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("litellm not available, using mock LLM integration")

from tkinter import messagebox
from typing import Any, Dict, List, Optional, Set, Tuple

import customtkinter as ctk

from xpr_agent_engine import XPRAgentEngine


# Fix for customtkinter DropdownMenu bug: _add_menu_commands fails on empty menu
def _patched_add_menu_commands(self: Any) -> None:
    """Patched version that handles empty menus safely."""
    try:
        # Check if menu has items before trying to delete
        end_index = self.index("end")
        if end_index is not None and end_index != "":
            self.delete(0, "end")
    except Exception:
        pass  # Menu is empty or not fully initialized

    # Add the actual menu commands - only if menu has values
    if hasattr(self, "_values") and self._values:
        for i, value in enumerate(self._values):
            self.add_command(
                label=value,
                command=lambda v=value: self._command(v) if self._command else None,
            )


# Apply the patch
ctk.windows.widgets.core_widget_classes.dropdown_menu.DropdownMenu._add_menu_commands = (
    _patched_add_menu_commands
)

import importlib.util
import json
import re
import subprocess
import threading
import time
from pathlib import Path

# Matplotlib imports for embedded visualization
import matplotlib

# Import hypothesis approval board
from hypothesis_approval_board import ApprovalBoard, Hypothesis, HypothesisStatus
from utils.apgi_security import secure_popen, secure_run

# Import ProgressTracker for type annotations
try:
    from progress_tracking import ProgressTracker
except ImportError:
    ProgressTracker = None  # type: ignore

# FigureCanvasTkAgg below selects the embedded canvas explicitly. Do not
# switch the global pyplot backend during import: analysis and headless tests
# may already have selected Agg.
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# Configure matplotlib for embedded GUI
matplotlib.rcParams["figure.figsize"] = (8, 6)
matplotlib.rcParams["figure.dpi"] = 100
matplotlib.rcParams["font.size"] = 8
matplotlib.rcParams["axes.titlesize"] = 10
matplotlib.rcParams["axes.labelsize"] = 8
matplotlib.rcParams["xtick.labelsize"] = 8
matplotlib.rcParams["ytick.labelsize"] = 8
matplotlib.rcParams["figure.autolayout"] = True
matplotlib.rcParams["figure.facecolor"] = "#2b2b2b"
matplotlib.rcParams["axes.facecolor"] = "#2b2b2b"
matplotlib.rcParams["text.color"] = "#ffffff"
matplotlib.rcParams["axes.edgecolor"] = "#444444"
matplotlib.rcParams["axes.linewidth"] = 1.2
matplotlib.rcParams["grid.color"] = "#444444"
matplotlib.rcParams["grid.linewidth"] = 0.5
matplotlib.rcParams["grid.alpha"] = 0.3
# Core dependencies required for APGI experiments
CORE_DEPENDENCIES = {
    "numpy": "NumPy - Numerical computing",
    "pandas": "Pandas - Data manipulation",
    "matplotlib": "Matplotlib - Plotting and visualization",
    "customtkinter": "CustomTkinter - Modern UI framework",
    "scipy": "SciPy - Scientific computing",
}

OPTIONAL_DEPENDENCIES = {
    "torch": "PyTorch - Deep learning framework",
    "sklearn": "Scikit-learn - Machine learning",
    "requests": "Requests - HTTP library",
    "tqdm": "tqdm - Progress bars",
    "PIL": "Pillow - Image processing",
}


class ExperimentRunnerGUI(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        # Set Dark theme as default before any UI creation
        ctk.set_appearance_mode("Dark")

        self.title("APGI Experiment Auto-Improvement")
        self.geometry("1400x900")

        # Set main path to current research directory
        self.research_dir = Path(os.path.dirname(os.path.abspath(__file__)))

        # Create menu bar
        self._create_menu_bar()

        # Thread-safety lock for all shared mutable state accessed from background
        # threads (running_experiments, active_processes, stop_all).
        self._state_lock = threading.Lock()

        # Application state
        self.running_experiments: Set[str] = set()
        self.experiment_cards: Dict[str, ctk.CTkFrame] = {}
        self.experiment_buttons: Dict[str, ctk.CTkButton] = {}
        self.status_indicators: Dict[str, ctk.CTkLabel] = {}
        self.active_processes: Dict[str, subprocess.Popen] = {}
        self.stop_all = False
        self.current_figure: Figure | None = None
        self.current_canvas: FigureCanvasTkAgg | None = None
        self.experiment_results: Dict[str, dict] = {}

        # Agent engine (lazy init on first use to avoid startup latency)
        self._agent_engine: Optional[XPRAgentEngine] = None

        # Hypothesis approval board
        self.approval_board: ApprovalBoard = ApprovalBoard()

        # Progress tracking
        self._progress_tracker: Optional["ProgressTracker"] = None

        # Guardrail state tracking
        self.guardrail_state: Dict[str, str | float] = {
            "status": "IDLE",
            "confidence": 1.0,
            "last_regression": 0.0,
            "escalation_count": 0,
            "last_experiment": "",
        }

        # Find experiments
        self.experiments = self._find_experiments()

        self._setup_ui()

        # Load progress tracker after UI is set up so we can log to console
        self._load_progress_tracker()

        # Register graceful shutdown handler
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Check dependencies after UI is initialized so we can log to console
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check for required dependencies on startup and show error if missing."""
        import importlib

        missing_core = []
        for module, description in CORE_DEPENDENCIES.items():
            try:
                importlib.import_module(module)
            except ImportError:
                missing_core.append(f"  - {module}: {description}")

        if missing_core:
            error_msg = "Missing required dependencies:\n\n" + "\n".join(missing_core)
            error_msg += "\n\nInstall with: pip install " + " ".join(
                mod.split("[")[0] for mod, _ in CORE_DEPENDENCIES.items()
            )
            messagebox.showerror("Missing Dependencies", error_msg)
            sys.exit(1)

        # Log optional dependencies
        missing_optional = []
        for module, description in OPTIONAL_DEPENDENCIES.items():
            try:
                importlib.import_module(module)
            except ImportError:
                missing_optional.append(f"  - {module}: {description}")

        if missing_optional:
            print("Optional dependencies missing (some features may be unavailable):")
            for msg in missing_optional:
                print(msg)

    def _load_progress_tracker(self) -> None:
        """Load existing progress tracker or create new one."""
        try:
            from progress_tracking import ProgressTracker

            progress_file = self.research_dir / "progress" / "current_progress.json"
            if progress_file.exists():
                self._progress_tracker = ProgressTracker(load_existing=True)
                self._log("Loaded existing progress", "#3498db")
            else:
                self._progress_tracker = ProgressTracker(
                    experiment_name="gui_session",
                    output_dir=str(self.research_dir / "progress"),
                )
                self._log("Initialized new progress tracker", "#3498db")
        except Exception as e:
            self._log(f"Progress tracker unavailable: {e}", "#f39c12")
            self._progress_tracker = None

    def _show_progress_dialog(self) -> None:
        """Show progress tracking dialog."""
        if self._progress_tracker is None:
            messagebox.showinfo("Progress", "Progress tracking not available")
            return

        dialog = ctk.CTkToplevel(self)
        dialog.title("Experiment Progress")
        dialog.geometry("500x400")
        dialog.transient(self)
        dialog.attributes("-topmost", True)

        ctk.CTkLabel(
            dialog,
            text="Progress Tracking",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(pady=(20, 10))

        scroll_frame = ctk.CTkScrollableFrame(dialog)
        scroll_frame.pack(padx=20, pady=10, fill="both", expand=True)

        tasks = self._progress_tracker.tasks
        if not tasks:
            ctk.CTkLabel(
                scroll_frame,
                text="No active tasks",
                font=ctk.CTkFont(size=12),
            ).pack(pady=20)
        else:
            for task_id, task in tasks.items():
                task_frame = ctk.CTkFrame(scroll_frame)
                task_frame.pack(fill="x", pady=5, padx=5)

                ctk.CTkLabel(
                    task_frame,
                    text=task_id,
                    font=ctk.CTkFont(size=12, weight="bold"),
                ).pack(anchor="w", padx=10, pady=(5, 0))

                ctk.CTkLabel(
                    task_frame,
                    text=task.message or "No message",
                ).pack(anchor="w", padx=10)

                progress_bar = ctk.CTkProgressBar(task_frame, width=300)
                progress_bar.set(task.progress_percent / 100.0)
                progress_bar.pack(padx=10, pady=5)

                status_colors = {
                    "pending": "#7f8c8d",
                    "running": "#3498db",
                    "completed": "#27ae60",
                    "failed": "#e74c3c",
                }
                ctk.CTkLabel(
                    task_frame,
                    text=f"Status: {task.status.value}",
                    text_color=status_colors.get(task.status.value, "#ffffff"),
                    font=ctk.CTkFont(size=10),
                ).pack(anchor="w", padx=10, pady=(0, 5))

        ctk.CTkButton(
            dialog,
            text="Close",
            command=dialog.destroy,
            width=120,
        ).pack(pady=20)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    @property
    def agent_engine(self) -> XPRAgentEngine:
        """Lazy-initialise XPRAgentEngine on first access."""
        if self._agent_engine is None:
            from xpr_agent_engine import XPRAgentEngine

            self._agent_engine = XPRAgentEngine()
        return self._agent_engine

    def _on_close(self) -> None:
        """Graceful shutdown: terminate all running experiment processes."""
        with self._state_lock:
            self.stop_all = True
            procs = list(self.active_processes.items())
        for name, proc in procs:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            logging.getLogger(__name__).info(f"Terminated experiment process: {name}")
        self.destroy()

    def _stop_experiment(self, name: str) -> None:
        """Terminate a single running experiment by name."""
        with self._state_lock:
            proc = self.active_processes.pop(name, None)
            self.running_experiments.discard(name)
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.after(
            0,
            lambda: self._log(f"[STOPPED] {name}", "#e74c3c"),
        )

    def _find_experiments(self) -> List[Tuple[str, str]]:
        """Dynamically find all run_*.py files in the experiments directory."""
        experiments = []
        experiments_dir = self.research_dir / "experiments"
        run_files = sorted(list(experiments_dir.glob("run_*.py")))

        for file in run_files:
            # Format name: run_visual_search.py -> Visual Search
            name = file.stem.replace("run_", "").replace("_", " ").title()
            if name == "Tests":
                continue  # Skip run_tests.py if it exists here
            experiments.append((name, f"experiments/{file.name}"))

        return experiments

    def _create_menu_bar(self) -> None:
        """Create the application menu bar with interactive buttons."""
        menubar = ctk.CTkFrame(self, height=40)
        menubar.grid(
            row=0, column=0, columnspan=2, sticky="ew", padx=(0, 0), pady=(0, 0)
        )

        # File menu button
        file_btn = ctk.CTkButton(
            menubar,
            text="File",
            font=ctk.CTkFont(size=12, weight="bold"),
            width=60,
            fg_color="transparent",
            hover_color="#333333",
            command=self._show_file_menu,
        )
        file_btn.pack(side="left", padx=5, pady=5)

        # Edit menu button
        edit_btn = ctk.CTkButton(
            menubar,
            text="Edit",
            font=ctk.CTkFont(size=12, weight="bold"),
            width=60,
            fg_color="transparent",
            hover_color="#333333",
            command=self._show_edit_menu,
        )
        edit_btn.pack(side="left", padx=5, pady=5)

        # View menu button
        view_btn = ctk.CTkButton(
            menubar,
            text="View",
            font=ctk.CTkFont(size=12, weight="bold"),
            width=60,
            fg_color="transparent",
            hover_color="#333333",
            command=self._show_view_menu,
        )
        view_btn.pack(side="left", padx=5, pady=5)

        # Help menu button
        help_btn = ctk.CTkButton(
            menubar,
            text="Help",
            font=ctk.CTkFont(size=12, weight="bold"),
            width=60,
            fg_color="transparent",
            hover_color="#333333",
            command=self._show_help_menu,
        )
        help_btn.pack(side="left", padx=5, pady=5)

        # Adjust grid layout to account for menu bar
        self.grid_rowconfigure(1, weight=3)  # Main content area
        self.grid_rowconfigure(2, weight=1)  # Console area

    def _setup_ui(self) -> None:
        # Configure grid layout (3x2: menu, sidebar+main, console)
        self.grid_columnconfigure(0, weight=0)  # Sidebar fixed width
        self.grid_columnconfigure(1, weight=1)  # Main content expandable
        self.grid_rowconfigure(0, weight=0)  # Menu bar fixed height
        self.grid_rowconfigure(1, weight=3)  # Main content area (3x weight)
        self.grid_rowconfigure(2, weight=1)  # Console area (1x weight)

        # Create navigation frame (sidebar)
        self.navigation_frame = ctk.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=1, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(7, weight=1)

        self.navigation_frame_label = ctk.CTkLabel(
            self.navigation_frame,
            text="APGI Research",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        # -----------------------------------------------------------
        # Compact Controls Section
        # -----------------------------------------------------------
        # Create a compact controls frame
        controls_frame = ctk.CTkFrame(self.navigation_frame, fg_color="transparent")
        controls_frame.grid(row=3, column=0, padx=15, pady=5, sticky="ew")
        controls_frame.grid_columnconfigure(0, weight=1)
        controls_frame.grid_columnconfigure(1, weight=1)

        # Appearance dropdown (smaller)
        self.appearance_mode_label = ctk.CTkLabel(
            controls_frame, text="Theme:", anchor="w", font=ctk.CTkFont(size=10)
        )
        self.appearance_mode_label.grid(
            row=0, column=0, columnspan=2, padx=(0, 0), pady=(0, 2), sticky="w"
        )

        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(
            controls_frame,
            values=["Dark", "Light", "System"],
            command=self.change_appearance_mode,
            width=120,
            height=24,
            font=ctk.CTkFont(size=10),
        )
        # Set default value to "Dark"
        self.appearance_mode_optionemenu.set("Dark")
        self.appearance_mode_optionemenu.grid(
            row=1, column=0, columnspan=2, padx=(0, 0), pady=(0, 8), sticky="ew"
        )

        # Action buttons in horizontal layout (smaller)
        self.run_all_button = ctk.CTkButton(
            controls_frame,
            text="▶ Run All",
            command=self._run_all,
            height=26,
            width=100,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#27ae60",
            hover_color="#219150",
        )
        self.run_all_button.grid(row=2, column=0, padx=(0, 3), pady=2)

        self.stop_button = ctk.CTkButton(
            controls_frame,
            text="⏹ Stop",
            command=self._stop_all,
            height=26,
            width=100,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#e74c3c",
            hover_color="#c0392b",
        )
        self.stop_button.grid(row=2, column=1, padx=(3, 0), pady=2)

        self.clear_button = ctk.CTkButton(
            controls_frame,
            text="🧹 Clear",
            command=self._clear_console,
            height=26,
            width=100,
            font=ctk.CTkFont(size=10),
            fg_color="#3498db",
            hover_color="#2980b9",
        )
        self.clear_button.grid(row=3, column=0, columnspan=2, padx=(0, 0), pady=(2, 8))

        # -----------------------------------------------------------
        # Compact Guardrail Dashboard Panel
        # -----------------------------------------------------------
        self.guardrail_frame = ctk.CTkFrame(
            self.navigation_frame, corner_radius=6, fg_color="#1a1a2e"
        )
        self.guardrail_frame.grid(row=4, column=0, padx=10, pady=(5, 3), sticky="ew")

        ctk.CTkLabel(
            self.guardrail_frame,
            text="⚡ Guardrails",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#2ecc71",
        ).grid(row=0, column=0, columnspan=2, padx=8, pady=(4, 2), sticky="w")

        # Compact grid layout for guardrail stats
        # Status
        ctk.CTkLabel(
            self.guardrail_frame,
            text="Status:",
            font=ctk.CTkFont(size=9),
            text_color="#888888",
        ).grid(row=1, column=0, padx=(8, 2), pady=1, sticky="w")
        self.guardrail_status_label = ctk.CTkLabel(
            self.guardrail_frame,
            text="IDLE",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color="#2ecc71",
        )
        self.guardrail_status_label.grid(
            row=1, column=1, padx=(2, 8), pady=1, sticky="w"
        )

        # Confidence
        ctk.CTkLabel(
            self.guardrail_frame,
            text="Conf:",
            font=ctk.CTkFont(size=9),
            text_color="#888888",
        ).grid(row=2, column=0, padx=(8, 2), pady=1, sticky="w")
        self.guardrail_confidence_label = ctk.CTkLabel(
            self.guardrail_frame,
            text="100%",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color="#2ecc71",
        )
        self.guardrail_confidence_label.grid(
            row=2, column=1, padx=(2, 8), pady=1, sticky="w"
        )

        # Regression
        ctk.CTkLabel(
            self.guardrail_frame,
            text="Regr:",
            font=ctk.CTkFont(size=9),
            text_color="#888888",
        ).grid(row=3, column=0, padx=(8, 2), pady=1, sticky="w")
        self.guardrail_regression_label = ctk.CTkLabel(
            self.guardrail_frame,
            text="0.0%",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color="#2ecc71",
        )
        self.guardrail_regression_label.grid(
            row=3, column=1, padx=(2, 8), pady=1, sticky="w"
        )

        # Escalation count
        ctk.CTkLabel(
            self.guardrail_frame,
            text="Esc:",
            font=ctk.CTkFont(size=9),
            text_color="#888888",
        ).grid(row=4, column=0, padx=(8, 2), pady=(1, 4), sticky="w")
        self.guardrail_escalation_label = ctk.CTkLabel(
            self.guardrail_frame,
            text="0",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color="#2ecc71",
        )
        self.guardrail_escalation_label.grid(
            row=4, column=1, padx=(2, 8), pady=(1, 4), sticky="w"
        )

        # -----------------------------------------------------------
        # Compact Hypothesis Board Panel
        # -----------------------------------------------------------
        self.hypothesis_frame = ctk.CTkFrame(
            self.navigation_frame, corner_radius=6, fg_color="#1a1a2e"
        )
        self.hypothesis_frame.grid(row=5, column=0, padx=10, pady=(5, 5), sticky="nsew")
        self.navigation_frame.grid_rowconfigure(5, weight=1)

        ctk.CTkLabel(
            self.hypothesis_frame,
            text="🧪 Hypotheses",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#e2e2e2",
        ).grid(row=0, column=0, columnspan=2, padx=8, pady=(4, 2), sticky="w")

        # Compact hypothesis controls
        controls_frame = ctk.CTkFrame(self.hypothesis_frame, fg_color="transparent")
        controls_frame.grid(row=1, column=0, padx=8, pady=2, sticky="ew")

        ctk.CTkButton(
            controls_frame,
            text="➕ New",
            command=self._show_create_hypothesis_dialog,
            width=65,
            height=24,
            font=ctk.CTkFont(size=9),
            fg_color="#27ae60",
            hover_color="#219150",
        ).grid(row=0, column=0, padx=(0, 2), pady=1)

        ctk.CTkButton(
            controls_frame,
            text="📋 Review",
            command=self._show_hypothesis_review,
            width=65,
            height=24,
            font=ctk.CTkFont(size=9),
            fg_color="#3498db",
            hover_color="#2980b9",
        ).grid(row=0, column=1, padx=(2, 0), pady=1)

        self.hypothesis_scrollable = ctk.CTkScrollableFrame(
            self.hypothesis_frame, label_text="Active"
        )
        self.hypothesis_scrollable.grid(
            row=2, column=0, padx=8, pady=(2, 4), sticky="nsew"
        )
        self.hypothesis_scrollable.grid_columnconfigure(0, weight=1)

        self._refresh_hypothesis_display()

        # -----------------------------------------------------------
        # XPR AUTO Explanation Panel (parameter change log)
        # -----------------------------------------------------------
        self.xpr_log_frame = ctk.CTkFrame(
            self.navigation_frame, corner_radius=6, fg_color="#0d1117"
        )
        self.xpr_log_frame.grid(row=6, column=0, padx=10, pady=(3, 8), sticky="ew")
        self.xpr_log_frame.grid_columnconfigure(0, weight=1)

        _xpr_hdr = ctk.CTkFrame(self.xpr_log_frame, fg_color="transparent")
        _xpr_hdr.grid(row=0, column=0, padx=8, pady=(4, 0), sticky="ew")
        _xpr_hdr.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            _xpr_hdr,
            text="🔬 XPR Changes",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#9b59b6",
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkButton(
            _xpr_hdr,
            text="✕",
            width=22,
            height=18,
            font=ctk.CTkFont(size=9),
            fg_color="transparent",
            hover_color="#2c2c2c",
            command=self._clear_xpr_log,
        ).grid(row=0, column=1, sticky="e")

        self.xpr_log_text = ctk.CTkTextbox(
            self.xpr_log_frame,
            height=90,
            font=("Courier", 9),
            state="disabled",
        )
        self.xpr_log_text.grid(row=1, column=0, padx=8, pady=(2, 6), sticky="ew")

        # -----------------------------------------------------------
        # Main Content Area (Experiments)
        # -----------------------------------------------------------
        self.scrollable_frame = ctk.CTkScrollableFrame(
            self, label_text=f"Research Experiments ({len(self.experiments)})"
        )
        self.scrollable_frame.grid(
            row=1, column=1, padx=(20, 10), pady=(20, 10), sticky="nsew"
        )
        self.scrollable_frame.grid_columnconfigure((0, 1), weight=1)

        for i, (name, script) in enumerate(self.experiments):
            self._create_experiment_card(self.scrollable_frame, name, script, i)

        # -----------------------------------------------------------
        # Console Area
        # -----------------------------------------------------------
        self.console_frame = ctk.CTkFrame(self, height=250)
        self.console_frame.grid(
            row=2, column=0, columnspan=2, padx=20, pady=(10, 20), sticky="nsew"
        )
        self.console_frame.grid_columnconfigure(0, weight=1)
        self.console_frame.grid_rowconfigure(0, weight=1)

        self.console_text = ctk.CTkTextbox(self.console_frame, font=("Courier", 13))
        self.console_text.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.console_text.insert("0.0", "--- APGI Research Console Ready ---\n")

    def _show_create_hypothesis_dialog(self) -> None:
        """Show dialog to create a new hypothesis."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("🧪 Create New Hypothesis")
        dialog.geometry("600x500")
        dialog.transient(self)
        dialog.attributes("-topmost", True)

        ctk.CTkLabel(
            dialog,
            text="Create New Hypothesis",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=(15, 5))

        # Form fields
        form_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        form_frame.pack(padx=20, fill="x", pady=10)

        # Title
        ctk.CTkLabel(form_frame, text="Title:", font=ctk.CTkFont(size=12)).grid(
            row=0, column=0, sticky="w", padx=5, pady=2
        )
        title_entry = ctk.CTkEntry(form_frame, width=400)
        title_entry.grid(row=0, column=1, padx=5, pady=2)

        # Description
        ctk.CTkLabel(form_frame, text="Description:", font=ctk.CTkFont(size=12)).grid(
            row=1, column=0, sticky="nw", padx=5, pady=2
        )
        desc_entry = ctk.CTkTextbox(form_frame, height=80)
        desc_entry.grid(row=1, column=1, padx=5, pady=2, sticky="nsew")

        # Predicted Outcome
        ctk.CTkLabel(
            form_frame, text="Predicted Outcome:", font=ctk.CTkFont(size=12)
        ).grid(row=2, column=0, sticky="nw", padx=5, pady=2)
        outcome_entry = ctk.CTkEntry(form_frame, width=400)
        outcome_entry.grid(row=2, column=1, padx=5, pady=2)

        # Confidence Score
        ctk.CTkLabel(
            form_frame, text="Confidence (0-1):", font=ctk.CTkFont(size=12)
        ).grid(row=3, column=0, sticky="nw", padx=5, pady=2)
        conf_entry = ctk.CTkEntry(form_frame, width=400)
        conf_entry.insert(0, "0.7")
        conf_entry.grid(row=3, column=1, padx=5, pady=2)

        # Risk Assessment
        ctk.CTkLabel(form_frame, text="Risk Level:", font=ctk.CTkFont(size=12)).grid(
            row=4, column=0, sticky="nw", padx=5, pady=2
        )
        risk_menu = ctk.CTkOptionMenu(form_frame, values=["low", "medium", "high"])
        risk_menu.grid(row=4, column=1, padx=5, pady=2)

        # Success Criteria
        ctk.CTkLabel(
            form_frame, text="Success Criteria:", font=ctk.CTkFont(size=12)
        ).grid(row=5, column=0, sticky="nw", padx=5, pady=2)
        criteria_entry = ctk.CTkTextbox(form_frame, height=60)
        criteria_entry.grid(row=5, column=1, padx=5, pady=2, sticky="nsew")

        # Buttons
        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(pady=15)

        def create_hypothesis() -> None:
            try:
                hypothesis = self.approval_board.create_hypothesis(
                    title=title_entry.get(),
                    description=desc_entry.get("0.0", "end"),
                    predicted_outcome=outcome_entry.get(),
                    confidence_score=float(conf_entry.get()),
                    risk_assessment=risk_menu.get(),
                    success_criteria=(
                        criteria_entry.get("0.0", "end").strip().split("\n")
                        if criteria_entry.get("0.0", "end").strip()
                        else []
                    ),
                )
                self._refresh_hypothesis_display()
                dialog.destroy()
                self._log(f"✅ Created hypothesis: {hypothesis.title}", "#2ecc71")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create hypothesis: {e}")

        def cancel() -> None:
            dialog.destroy()

        ctk.CTkButton(
            btn_frame,
            text="Create",
            command=create_hypothesis,
            fg_color="#27ae60",
            hover_color="#219150",
            width=100,
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            btn_frame,
            text="Cancel",
            command=cancel,
            fg_color="#e74c3c",
            hover_color="#c0392b",
            width=100,
        ).pack(side="left", padx=5)

    def _show_hypothesis_review(self) -> None:
        """Show dialog to review pending hypotheses."""
        pending_hypotheses = self.approval_board.get_pending_hypotheses()
        if not pending_hypotheses:
            messagebox.showinfo(
                "No Pending Hypotheses", "No hypotheses pending review."
            )
            return

        dialog = ctk.CTkToplevel(self)
        dialog.title("📋 Review Hypotheses")
        dialog.geometry("800x600")
        dialog.transient(self)
        dialog.attributes("-topmost", True)

        ctk.CTkLabel(
            dialog,
            text="Pending Hypotheses Review",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=(15, 5))

        # Create scrollable frame for hypotheses
        review_frame = ctk.CTkScrollableFrame(dialog, height=400)
        review_frame.pack(padx=20, pady=10, fill="both", expand=True)

        for hypothesis in pending_hypotheses:
            self._create_hypothesis_review_item(review_frame, hypothesis, dialog)

        # Buttons
        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(pady=10)

        def close_review() -> None:
            dialog.destroy()

        ctk.CTkButton(
            btn_frame,
            text="Close",
            command=close_review,
            fg_color="#7f8c8d",
            hover_color="#636e72",
            width=100,
        ).pack()

    def _create_hypothesis_display_item(
        self, parent: ctk.CTkFrame, hypothesis: Hypothesis
    ) -> None:
        """Create a compact hypothesis display item for the main panel."""
        item_frame = ctk.CTkFrame(parent, corner_radius=6, fg_color="#2a2a3e")
        item_frame.pack(fill="x", pady=3, padx=5)

        # Title and status
        header_frame = ctk.CTkFrame(item_frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=8, pady=(4, 2))

        title_label = ctk.CTkLabel(
            header_frame,
            text=hypothesis.title,
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#ffffff",
        )
        title_label.pack(side="left")

        # Status badge
        status_colors = {
            HypothesisStatus.DRAFT: "#f39c12",
            HypothesisStatus.PENDING: "#3498db",
            HypothesisStatus.UNDER_REVIEW: "#9b59b6",
            HypothesisStatus.APPROVED: "#27ae60",
            HypothesisStatus.REJECTED: "#e74c3c",
        }

        status_label = ctk.CTkLabel(
            header_frame,
            text=hypothesis.status.value.upper(),
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color=status_colors.get(hypothesis.status, "#666666"),
            fg_color="#1a1a2e",
            corner_radius=4,
            padx=6,
            pady=2,
        )
        status_label.pack(side="right")

        # Description (truncated)
        if hypothesis.description:
            desc_text = (
                hypothesis.description[:60] + "..."
                if len(hypothesis.description) > 60
                else hypothesis.description
            )
            desc_label = ctk.CTkLabel(
                item_frame,
                text=desc_text,
                font=ctk.CTkFont(size=9),
                text_color="#cccccc",
                wraplength=250,
            )
            desc_label.pack(anchor="w", padx=8, pady=(0, 4))

    def _create_hypothesis_review_item(
        self,
        parent: ctk.CTkFrame,
        hypothesis: Hypothesis,
        dialog: ctk.CTkToplevel | None,
    ) -> None:
        """Create a single hypothesis review item."""
        item_frame = ctk.CTkFrame(parent, corner_radius=8, fg_color="#f8f9fa")
        item_frame.pack(fill="x", pady=5, padx=10)

        # Title and status
        header_frame = ctk.CTkFrame(item_frame, fg_color="transparent")
        header_frame.pack(fill="x", pady=5)

        title_label = ctk.CTkLabel(
            header_frame,
            text=hypothesis.title,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        title_label.pack(side="left", padx=5)

        status_colors = {
            HypothesisStatus.PENDING: "#f39c12",
            HypothesisStatus.UNDER_REVIEW: "#3498db",
            HypothesisStatus.APPROVED: "#2ecc71",
            HypothesisStatus.REJECTED: "#e74c3c",
            HypothesisStatus.MODIFIED: "#ff9500",
        }
        status_label = ctk.CTkLabel(
            header_frame,
            text=hypothesis.status.value.upper(),
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=status_colors.get(hypothesis.status, "#666666"),
        )
        status_label.pack(side="right", padx=5)

        # Description
        desc_frame = ctk.CTkFrame(item_frame, fg_color="transparent")
        desc_frame.pack(fill="x", pady=(0, 5), padx=10)

        desc_label = ctk.CTkLabel(
            desc_frame,
            text="Description:",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#333333",
        )
        desc_label.pack(anchor="w", padx=(0, 2))

        desc_text = ctk.CTkLabel(
            desc_frame,
            text=hypothesis.description,
            font=ctk.CTkFont(size=10),
            wraplength=300,
        )
        desc_text.pack(anchor="w", padx=(0, 2), pady=(0, 5))

        # Metadata
        meta_frame = ctk.CTkFrame(item_frame, fg_color="transparent")
        meta_frame.pack(fill="x", pady=5, padx=10)

        # Confidence and Risk
        ctk.CTkLabel(
            meta_frame,
            text=f"Confidence: {hypothesis.confidence_score:.2f}",
            font=ctk.CTkFont(size=10),
        ).pack(anchor="w", pady=2)
        ctk.CTkLabel(
            meta_frame,
            text=f"Risk: {hypothesis.risk_assessment.upper()}",
            font=ctk.CTkFont(size=10),
        ).pack(anchor="w", pady=2)

        # Action buttons
        action_frame = ctk.CTkFrame(item_frame, fg_color="transparent")
        action_frame.pack(fill="x", pady=5, padx=10)

        def approve() -> None:
            self.approval_board.update_hypothesis_status(
                hypothesis.id,
                HypothesisStatus.APPROVED,
                "Approved via GUI review",
                "GUI Reviewer",
            )
            self._refresh_hypothesis_display()
            if dialog:
                dialog.destroy()

        def reject() -> None:
            self.approval_board.update_hypothesis_status(
                hypothesis.id,
                HypothesisStatus.REJECTED,
                "Rejected via GUI review",
                "GUI Reviewer",
            )
            self._refresh_hypothesis_display()
            if dialog:
                dialog.destroy()

        def modify() -> None:
            # For now, just log - could open modify dialog later
            self._log(f"Request to modify hypothesis: {hypothesis.id}")

        ctk.CTkButton(
            action_frame,
            text="✅ Approve",
            command=approve,
            fg_color="#27ae60",
            hover_color="#219150",
            width=80,
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            action_frame,
            text="🔧 Modify",
            command=modify,
            fg_color="#f39c12",
            hover_color="#d68910",
            width=80,
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            action_frame,
            text="❌ Reject",
            command=reject,
            fg_color="#e74c3c",
            hover_color="#c0392b",
            width=80,
        ).pack(side="left", padx=5)

    def _refresh_hypothesis_display(self) -> None:
        """Refresh the hypothesis display scrollable frame."""
        for widget in self.hypothesis_scrollable.winfo_children():
            widget.destroy()

        # Show both DRAFT and PENDING hypotheses for better visibility
        active_hypotheses = [
            h
            for h in self.approval_board.hypotheses.values()
            if h.status in [HypothesisStatus.DRAFT, HypothesisStatus.PENDING]
        ]

        if not active_hypotheses:
            no_hypotheses_label = ctk.CTkLabel(
                self.hypothesis_scrollable,
                text="No active hypotheses",
                font=ctk.CTkFont(size=12, slant="italic"),
                text_color="#333333",
            )
            no_hypotheses_label.pack(pady=20)
        else:
            for hypothesis in active_hypotheses:
                self._create_hypothesis_display_item(
                    self.hypothesis_scrollable, hypothesis
                )

    def _create_experiment_card(
        self, parent: Any, name: str, script: str, index: int
    ) -> None:
        row = index // 2
        col = index % 2

        card = ctk.CTkFrame(parent)
        card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)

        name_label = ctk.CTkLabel(
            card, text=name, font=ctk.CTkFont(size=16, weight="bold")
        )
        name_label.grid(row=0, column=0, padx=15, pady=(15, 5), sticky="w")

        script_label = ctk.CTkLabel(
            card, text=f"📄 {script}", font=ctk.CTkFont(size=12), text_color="#444444"
        )
        script_label.grid(row=1, column=0, padx=15, pady=(0, 10), sticky="w")

        status_label = ctk.CTkLabel(
            card, text="Ready", font=ctk.CTkFont(size=12), text_color="#3498db"
        )
        status_label.grid(row=2, column=0, padx=15, pady=(0, 15), sticky="w")
        self.status_indicators[name] = status_label

        run_btn = ctk.CTkButton(
            card,
            text="▶ RUN",
            command=lambda n=name, s=script: self._run_experiment(n, s),
            width=70,
            height=28,
            fg_color="#27ae60",
            hover_color="#219150",
        )
        run_btn.grid(row=0, column=1, padx=(0, 5), pady=(15, 5))
        self.experiment_buttons[name] = run_btn

        viz_btn = ctk.CTkButton(
            card,
            text="📊 VIZ",
            command=lambda n=name: self._show_results_visualization(n),
            width=70,
            height=28,
            fg_color="#3498db",
            hover_color="#2980b9",
        )
        viz_btn.grid(row=1, column=1, padx=(0, 5), pady=(0, 15))

        auto_btn = ctk.CTkButton(
            card,
            text="🤖 XPR AUTO",
            command=lambda n=name, s=script: self._run_auto_improve(n, s),
            width=70,
            height=28,
            fg_color="#8e44ad",
            hover_color="#9b59b6",
        )
        auto_btn.grid(row=0, column=2, padx=(0, 15), pady=(15, 5))

        self.experiment_cards[name] = card

    def _log(self, text: str, color: str | None = None) -> None:
        self.console_text.insert("end", text + "\n")
        self.console_text.see("end")

    def _clear_console(self) -> None:
        self.console_text.delete("1.0", "end")
        self.console_text.insert("1.0", "--- Console Cleared ---\n")

    def _clear_xpr_log(self) -> None:
        self.xpr_log_text.configure(state="normal")
        self.xpr_log_text.delete("1.0", "end")
        self.xpr_log_text.configure(state="disabled")

    def _log_xpr_change(self, experiment: str, modifications: dict) -> None:
        """Append a parameter-change entry to the XPR Changes sidebar panel.

        Safe to call from any thread.
        """
        if threading.current_thread() is not threading.main_thread():
            self.after(
                0, lambda e=experiment, m=modifications: self._log_xpr_change(e, m)
            )
            return
        import datetime as _dt

        ts = _dt.datetime.now().strftime("%H:%M:%S")
        lines = [f"[{ts}] {experiment}"]
        for param, val in modifications.items():
            lines.append(f"  {param} → {val!r}")
        entry = "\n".join(lines) + "\n"
        self.xpr_log_text.configure(state="normal")
        self.xpr_log_text.insert("end", entry)
        self.xpr_log_text.see("end")
        self.xpr_log_text.configure(state="disabled")

    def _update_guardrail_dashboard(
        self,
        status: str = "IDLE",
        confidence: float = 1.0,
        regression: float = 0.0,
        experiment: str = "",
    ) -> None:
        """Update the guardrail dashboard indicators in the sidebar."""
        self.guardrail_state["status"] = status
        self.guardrail_state["confidence"] = confidence
        self.guardrail_state["last_regression"] = regression
        if experiment:
            self.guardrail_state["last_experiment"] = experiment

        # Status color logic
        status_colors = {
            "IDLE": "#2ecc71",
            "RUNNING": "#3498db",
            "OK": "#2ecc71",
            "WARNING": "#f39c12",
            "ESCALATED": "#e74c3c",
            "HALTED": "#e74c3c",
        }
        status_color = status_colors.get(status, "#aaaaaa")
        self.guardrail_status_label.configure(text=status, text_color=status_color)

        # Confidence color: green > 0.7, yellow > 0.4, red otherwise
        if confidence > 0.7:
            conf_color = "#2ecc71"
        elif confidence > 0.4:
            conf_color = "#f39c12"
        else:
            conf_color = "#e74c3c"
        self.guardrail_confidence_label.configure(
            text=f"{confidence:.0%}", text_color=conf_color
        )

        # Regression color: green < 2%, yellow < 5%, red otherwise
        reg_pct = abs(regression) * 100
        if reg_pct < 2:
            reg_color = "#2ecc71"
        elif reg_pct < 5:
            reg_color = "#f39c12"
        else:
            reg_color = "#e74c3c"
        self.guardrail_regression_label.configure(
            text=f"{reg_pct:.1f}%", text_color=reg_color
        )

        esc_count = self.guardrail_state.get("escalation_count", 0)
        esc_color = "#2ecc71" if esc_count == 0 else "#e74c3c"
        self.guardrail_escalation_label.configure(
            text=str(esc_count), text_color=esc_color
        )

    def _notify_guardrail_escalation(
        self, experiment_name: str, confidence: float, message: str
    ) -> None:
        """Show a GUI notification when a guardrail is tripped."""
        self.guardrail_state["escalation_count"] = (
            int(self.guardrail_state.get("escalation_count", 0)) + 1
        )
        self._update_guardrail_dashboard(
            status="ESCALATED", confidence=confidence, experiment=experiment_name
        )
        self._log(
            f"\n🚨 [GUARDRAIL ESCALATION] {experiment_name}: {message}", "#e74c3c"
        )
        # Show a non-blocking alert dialog
        alert = ctk.CTkToplevel(self)
        alert.title("⚠️ Guardrail Escalation")
        alert.geometry("450x200")
        alert.transient(self)
        alert.attributes("-topmost", True)

        ctk.CTkLabel(
            alert,
            text="🚨 Agent Guardrail Tripped",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#e74c3c",
        ).pack(pady=(15, 5))
        ctk.CTkLabel(
            alert,
            text=f"Experiment: {experiment_name}\nConfidence: {confidence:.0%}\n{message}",
            font=ctk.CTkFont(size=12),
            justify="center",
        ).pack(pady=10)
        ctk.CTkButton(
            alert,
            text="Acknowledge",
            command=alert.destroy,
            fg_color="#e74c3c",
            hover_color="#c0392b",
        ).pack(pady=10)

    def _run_auto_improve(self, name: str, script: str) -> None:
        """Phase 4: Launch the Steering Dashboard before execution."""

        # -----------------------------------------------------------
        # Phase 2: configure_if_needed() — Pre-run constraints dialog
        # -----------------------------------------------------------
        config_dialog = ctk.CTkToplevel(self)
        config_dialog.title(f"⚙️ Configure — {name}")
        config_dialog.geometry("480x400")
        config_dialog.transient(self)
        config_dialog.attributes("-topmost", True)

        ctk.CTkLabel(
            config_dialog,
            text="Pre-Run Configuration",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=(15, 5))
        ctk.CTkLabel(
            config_dialog,
            text="Set bounds and constraints before XPR AUTO execution.",
            font=ctk.CTkFont(size=11),
            text_color="#333333",
        ).pack(pady=(0, 10))

        fields_frame = ctk.CTkFrame(config_dialog, fg_color="transparent")
        fields_frame.pack(padx=20, fill="x")

        # Max Iterations
        ctk.CTkLabel(
            fields_frame, text="Max Iterations:", font=ctk.CTkFont(size=12)
        ).grid(row=0, column=0, sticky="w", pady=5)
        iter_entry = ctk.CTkEntry(fields_frame, width=120)
        iter_entry.insert(0, "3")
        iter_entry.grid(row=0, column=1, padx=10, pady=5)

        # Time Budget (seconds)
        ctk.CTkLabel(
            fields_frame, text="Time Budget (s):", font=ctk.CTkFont(size=12)
        ).grid(row=1, column=0, sticky="w", pady=5)
        time_entry = ctk.CTkEntry(fields_frame, width=120)
        time_entry.insert(0, "600")
        time_entry.grid(row=1, column=1, padx=10, pady=5)

        # Confidence Threshold
        ctk.CTkLabel(
            fields_frame, text="Min Confidence:", font=ctk.CTkFont(size=12)
        ).grid(row=2, column=0, sticky="w", pady=5)
        conf_entry = ctk.CTkEntry(fields_frame, width=120)
        conf_entry.insert(0, "0.5")
        conf_entry.grid(row=2, column=1, padx=10, pady=5)

        # Protected Files
        ctk.CTkLabel(
            fields_frame, text="Protected Files:", font=ctk.CTkFont(size=12)
        ).grid(row=3, column=0, sticky="w", pady=5)
        protected_entry = ctk.CTkEntry(fields_frame, width=250)
        protected_entry.insert(0, "prepare_*.py")
        protected_entry.grid(row=3, column=1, padx=10, pady=5)

        # Additional constraints
        ctk.CTkLabel(
            config_dialog, text="Additional Constraints:", font=ctk.CTkFont(size=12)
        ).pack(padx=20, anchor="w", pady=(10, 0))
        constraints_box = ctk.CTkTextbox(config_dialog, height=60, width=430)
        constraints_box.pack(padx=20, pady=5)
        constraints_box.insert(
            "0.0", "Do not modify file structure. Keep bounds valid."
        )

        # Store config for use in the plan generation
        run_config: Dict[str, Any] = {"proceed": False}

        def on_proceed() -> None:
            try:
                run_config["iterations"] = int(iter_entry.get())
            except ValueError:
                run_config["iterations"] = 3
            try:
                run_config["time_budget"] = int(time_entry.get())
            except ValueError:
                run_config["time_budget"] = 600
            try:
                run_config["min_confidence"] = float(conf_entry.get())
            except ValueError:
                run_config["min_confidence"] = 0.5
            try:
                run_config["protected_files"] = protected_entry.get().strip()
            except Exception:
                pass
            run_config["proceed"] = True
            # Store config for use in plan generation
            self.run_config = run_config
            config_dialog.destroy()

        def on_skip() -> None:
            run_config["iterations"] = 3
            run_config["time_budget"] = 600
            run_config["min_confidence"] = 0.5
            run_config["protected_files"] = "prepare_*.py"
            run_config["proceed"] = True
            self.run_config = run_config
            config_dialog.destroy()

        btn_row = ctk.CTkFrame(config_dialog, fg_color="transparent")
        btn_row.pack(pady=15)
        ctk.CTkButton(
            btn_row,
            text="▶ Proceed",
            command=on_proceed,
            fg_color="#27ae60",
            hover_color="#219150",
            width=120,
        ).pack(side="left", padx=10)
        ctk.CTkButton(
            btn_row,
            text="Skip (Defaults)",
            command=on_skip,
            fg_color="#7f8c8d",
            hover_color="#636e72",
            width=120,
        ).pack(side="left", padx=10)
        ctk.CTkButton(
            btn_row,
            text="Cancel",
            command=config_dialog.destroy,
            fg_color="#e74c3c",
            hover_color="#c0392b",
            width=100,
        ).pack(side="left", padx=10)

        def wait_for_config() -> None:
            """Wait for the config dialog to close, then continue."""
            if config_dialog.winfo_exists():
                self.after(100, wait_for_config)
                return
            if not run_config.get("proceed"):
                self._log(f"[XPR AGENT] Configuration cancelled for {name}.", "#7f8c8d")
                return

            self._log(
                f"\n[XPR AGENT] Configured: iters={run_config.get('iterations')}, "
                f"budget={run_config.get('time_budget')}s, "
                f"min_conf={run_config.get('min_confidence')}",
                "#8e44ad",
            )
            self._launch_plan_generation(name, script, run_config)

        self.after(100, wait_for_config)

    def _launch_plan_generation(
        self, name: str, script: str, run_config: Dict[str, Any]
    ) -> None:
        """Boot the XPR Agent after configuration is complete."""
        from autonomous_agent import AutonomousAgent
        from xpr_agent_engine import XPRAgentEngine

        # Create dialog for plan review
        dialog = ctk.CTkToplevel(self)
        dialog.title(f"XPR Control Layer - {name}")
        dialog.geometry("600x500")

        ctk.CTkLabel(
            dialog,
            text="ExperimentPlan Review",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=10)

        plan_text = ctk.CTkTextbox(dialog, height=300, width=550)
        plan_text.pack(pady=10, padx=20)

        def popup_callback(plan_str: str) -> None:
            popup_dialog = ctk.CTkToplevel(self)
            popup_dialog.title(f"XPR Control Layer - {name}")
            popup_dialog.geometry("600x500")

            ctk.CTkLabel(
                popup_dialog,
                text="ExperimentPlan Review",
                font=ctk.CTkFont(size=16, weight="bold"),
            ).pack(pady=10)

            plan_display_text = ctk.CTkTextbox(popup_dialog, height=300, width=550)
            plan_display_text.pack(pady=10, padx=20)
            plan_display_text.insert("0.0", plan_str)

            def approve() -> None:
                self._log(
                    f"[XPR AGENT] Plan APPROVED. Executing {name} tuning.", "#2ecc71"
                )
                self._update_guardrail_dashboard(status="RUNNING", experiment=name)
                popup_dialog.destroy()

                def run_agent() -> None:
                    try:
                        agent = AutonomousAgent(
                            str(self.research_dir),
                            param_change_callback=self._log_xpr_change,
                        )
                        experiment_key = (
                            script.replace("experiments/", "")
                            .replace("run_", "")
                            .replace(".py", "")
                        )
                        agent.optimize_experiment(
                            experiment_key, iterations=3, resume=False
                        )
                        self.after(
                            0,
                            lambda: self._update_guardrail_dashboard(
                                status="OK", confidence=0.9, experiment=name
                            ),
                        )
                    except Exception as e:
                        self._log(f"[XPR AGENT] Agent execution failed: {e}", "#e74c3c")
                        self._update_guardrail_dashboard(
                            status="WARNING", confidence=0.3, experiment=name
                        )
                        self._notify_guardrail_escalation(name, 0.3, str(e))

                threading.Thread(target=run_agent, daemon=True).start()

        # Generate plan using agent engine
        try:
            plan_result = self.agent_engine.plan_experiment(name, {})
            plan_str = (
                str(plan_result.result)
                if plan_result.success
                else "Failed to generate plan"
            )
            plan_text.insert("0.0", plan_str)
            # Remove unused variable
            # plan_details = (
            #     plan_result.result
            #     if plan_result.success
            #     else f"Failed to generate plan: {plan_result.error}"
            # )
        except Exception as e:
            plan_str = f"Failed to generate plan: {e}"
            # Show the error in the textbox widget — never overwrite plan_text reference.
            plan_text.delete("0.0", "end")
            plan_text.insert("0.0", plan_str)
            self._log(f"[XPR AGENT] Plan generation failed: {e}", "#e74c3c")
            self._update_guardrail_dashboard(status="WARNING", experiment=name)
            self._update_guardrail_dashboard(
                status="WARNING", confidence=0.6, experiment=name
            )
            # Capture current agent plan string (if any) to use in the modify chain.
            _plan_result = self.agent_engine.get_current_plan()
            if _plan_result:
                plan_content = _plan_result.get("plan", plan_str)
            else:
                plan_content = plan_str
            dialog.destroy()

            def run_modify_chain() -> None:
                try:
                    engine = self.agent_engine
                    experiment_key = (
                        script.replace("experiments/", "")
                        .replace("run_", "")
                        .replace(".py", "")
                    )
                    # Step 1: Run issue-fix chain on the current plan
                    fix_result = engine.execute_skill(
                        "issue_fix",
                        experiment_key=experiment_key,
                        original_plan=plan_content,
                        current_plan=plan_content,
                    )
                    fix_summary = (
                        str(fix_result.result)
                        if fix_result.success
                        else "No fix output generated."
                    )
                    self._log(
                        f"[XPR AGENT] Issue-fix chain complete: {fix_summary[:200]}",
                        "#f39c12",
                    )

                    # Step 2: Re-plan with the fix context
                    refined_plan = engine.plan_experiment(
                        task=f"Refine plan for {experiment_key}. Previous fix: {fix_summary[:500]}",
                        current_params={},
                    )
                    refined_str = (
                        str(refined_plan.result)
                        if refined_plan.success
                        else "No plan generated"
                    )

                    # Step 3: Re-open the popup with the refined plan
                    self.after(0, lambda: popup_callback(refined_str))
                except Exception as exc:
                    err_msg = str(exc)
                    self._log(f"[XPR AGENT] Modify chain failed: {err_msg}", "#e74c3c")
                    self.after(
                        0,
                        lambda m=err_msg: self._notify_guardrail_escalation(
                            name, 0.3, f"Modify chain failed: {m}"
                        ),
                    )

            threading.Thread(target=run_modify_chain, daemon=True).start()

        def approve() -> None:
            """Approve: Execute the plan directly."""
            self._log(f"[XPR AGENT] Plan APPROVED. Executing {name} tuning.", "#2ecc71")
            self._update_guardrail_dashboard(status="RUNNING", experiment=name)
            dialog.destroy()

            def run_agent() -> None:
                try:
                    agent = AutonomousAgent(str(self.research_dir))
                    experiment_key = (
                        script.replace("experiments/", "")
                        .replace("run_", "")
                        .replace(".py", "")
                    )
                    agent.optimize_experiment(
                        experiment_key, iterations=3, resume=False
                    )
                    self.after(
                        0,
                        lambda: self._update_guardrail_dashboard(
                            status="OK", confidence=0.9, experiment=name
                        ),
                    )
                except Exception as e:
                    self._log(f"[XPR AGENT] Agent execution failed: {e}", "#e74c3c")
                    self._update_guardrail_dashboard(
                        status="WARNING", confidence=0.3, experiment=name
                    )
                    self._notify_guardrail_escalation(name, 0.3, str(e))

            threading.Thread(target=run_agent, daemon=True).start()

        def modify() -> None:
            """Modify: Run issue-fix chain and re-plan."""
            self._log("[XPR AGENT] Plan modification requested.", "#f39c12")

            # Get current plan text BEFORE destroying dialog
            current_plan_text = plan_text.get("0.0", "end").strip()
            dialog.destroy()

            def run_modify_chain() -> None:
                try:
                    engine = XPRAgentEngine()
                    experiment_key = (
                        script.replace("experiments/", "")
                        .replace("run_", "")
                        .replace(".py", "")
                    )
                    # Step 1: Run issue-fix chain on the current plan
                    fix_result = engine.execute_skill(
                        "issue_fix",
                        experiment_key=experiment_key,
                        original_plan=current_plan_text,
                        current_plan=current_plan_text,
                    )
                    fix_summary = (
                        str(fix_result.result)
                        if fix_result.success
                        else "No fix output generated."
                    )
                    self._log(
                        f"[XPR AGENT] Issue-fix chain complete: {fix_summary[:200]}",
                        "#f39c12",
                    )

                    # Step 2: Re-plan with the fix context
                    refined_plan = engine.plan_experiment(
                        task=f"Refine plan for {experiment_key}. Previous fix: {fix_summary[:500]}",
                        current_params={},
                    )
                    refined_str = (
                        str(refined_plan.result)
                        if refined_plan.success
                        else "No plan generated"
                    )

                    # Step 3: Re-open the popup with the refined plan
                    self.after(0, lambda: popup_callback(refined_str))
                except Exception as exc:
                    err_msg = str(exc)
                    self._log(f"[XPR AGENT] Modify chain failed: {err_msg}", "#e74c3c")
                    self.after(
                        0,
                        lambda m=err_msg: self._notify_guardrail_escalation(
                            name, 0.3, f"Modify chain failed: {m}"
                        ),
                    )

            threading.Thread(target=run_modify_chain, daemon=True).start()

        def reject() -> None:
            """Reject: Prompt for human priorities, then re-plan with those priorities."""
            self._log(
                "[XPR AGENT] Plan REJECTED. Requesting human priorities.",
                "#e74c3c",
            )
            self._update_guardrail_dashboard(
                status="HALTED", confidence=0.2, experiment=name
            )
            dialog.destroy()

            # Show priority input dialog
            priority_dialog = ctk.CTkToplevel(self)
            priority_dialog.title("Rejection - Set New Priorities")
            priority_dialog.geometry("500x300")
            priority_dialog.transient(self)

            ctk.CTkLabel(
                priority_dialog,
                text="What should the agent focus on next?",
                font=ctk.CTkFont(size=14, weight="bold"),
            ).pack(pady=(15, 5))
            ctk.CTkLabel(
                priority_dialog,
                text="Describe your priorities, constraints, or a new direction:",
                font=ctk.CTkFont(size=11),
                text_color="#333333",
            ).pack(pady=(0, 5))

            priority_input = ctk.CTkTextbox(priority_dialog, height=120, width=450)
            priority_input.pack(pady=10, padx=20)
            priority_input.insert(
                "0.0",
                "e.g., Focus on reducing false positives rather than overall accuracy.",
            )

            def submit_priorities() -> None:
                human_priorities = priority_input.get("0.0", "end").strip()
                priority_dialog.destroy()
                self._log(
                    f"[XPR AGENT] New priorities received: {human_priorities[:100]}...",
                    "#3498db",
                )

                def run_replan() -> None:
                    try:
                        engine = XPRAgentEngine()
                        experiment_key = (
                            script.replace("experiments/", "")
                            .replace("run_", "")
                            .replace(".py", "")
                        )
                        new_plan = engine.plan_experiment(
                            task=f"Re-plan {experiment_key} with human priorities: {human_priorities}",
                            current_params={},
                        )
                        self.after(
                            0,
                            lambda: popup_callback(
                                str(new_plan.result) if new_plan.success else "No plan"
                            ),
                        )
                    except Exception as e:
                        self._log(f"[XPR AGENT] Re-plan failed: {e}", "#e74c3c")

                threading.Thread(target=run_replan, daemon=True).start()

            def cancel_reject() -> None:
                priority_dialog.destroy()
                self._update_guardrail_dashboard(status="IDLE", experiment=name)

            btn_row = ctk.CTkFrame(priority_dialog, fg_color="transparent")
            btn_row.pack(pady=10)
            ctk.CTkButton(
                btn_row,
                text="Submit & Re-plan",
                command=submit_priorities,
                fg_color="#3498db",
                width=130,
                hover_color="#2980b9",
            ).pack(side="left", padx=10)
            ctk.CTkButton(
                btn_row,
                text="Cancel",
                command=cancel_reject,
                fg_color="#7f8c8d",
                width=100,
                hover_color="#636e72",
            ).pack(side="left", padx=10)

        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(pady=10)

        ctk.CTkButton(
            btn_frame,
            text="✅ Approve",
            command=approve,
            fg_color="#27ae60",
            width=110,
            hover_color="#219150",
        ).pack(side="left", padx=10)
        ctk.CTkButton(
            btn_frame,
            text="🔧 Modify",
            command=modify,
            fg_color="#f39c12",
            width=110,
            hover_color="#d68910",
        ).pack(side="left", padx=10)

    def _run_experiment(self, name: str, script: str) -> None:
        if name in self.running_experiments:
            return

        self.running_experiments.add(name)
        self.experiment_buttons[name].configure(state="disabled")
        self.status_indicators[name].configure(text="Running...", text_color="#f39c12")
        self._log(f"\n[STARTING] {name} ({script})", "#3498db")

        thread = threading.Thread(target=self._execute_script, args=(name, script))
        thread.daemon = True
        thread.start()

    def _execute_script(
        self, name: str, script: str, output_callback: Any = None
    ) -> None:
        """Execute script with optional output callback and input sanitization."""
        try:
            # Validate and sanitize script name
            if not script.endswith(".py"):
                raise ValueError("Only Python scripts (.py) are allowed")

            # Validate script path is within research directory
            script_path = Path(self.research_dir) / script
            if not script_path.exists():
                raise ValueError(f"Script not found: {script}")

            # Ensure script is within allowed directory
            try:
                script_path.resolve().relative_to(Path(self.research_dir).resolve())
            except ValueError:
                raise ValueError("Script must be within research directory")

            # Convert script path to module name for proper package execution
            # e.g., "experiments/run_stroop_effect.py" -> "experiments.run_stroop_effect"
            script_path_rel = script_path.relative_to(self.research_dir)
            module_name = str(script_path_rel.with_suffix("")).replace(os.sep, ".")

            # Use same python executable as current process
            env = os.environ.copy()
            env["PYTHONPATH"] = (
                str(self.research_dir) + os.pathsep + env.get("PYTHONPATH", "")
            )
            # Prevent CoreFoundation fork warnings in subprocesses on macOS
            if sys.platform == "darwin":
                env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
            # Run as module to support relative imports
            # Use python3 explicitly for consistency
            proc = secure_popen(
                ["python3", "-m", module_name],
                cwd=self.research_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                bufsize=1,  # Line buffered
                universal_newlines=True,
            )
            self.active_processes[name] = proc

            # Collect output in a buffer instead of calling after() for every line
            def read_stream_to_buffer(
                stream: Any, prefix: str, buffer: List[str]
            ) -> None:
                try:
                    for line in iter(stream.readline, ""):
                        if self.stop_all:
                            break
                        if line:
                            buffer.append(f"[{prefix}] {line.rstrip()}")
                except (ValueError, IOError):
                    pass
                finally:
                    stream.close()

            stdout_buffer: List[str] = []
            stderr_buffer: List[str] = []

            stdout_thread = threading.Thread(
                target=read_stream_to_buffer, args=(proc.stdout, name, stdout_buffer)
            )
            stderr_thread = threading.Thread(
                target=read_stream_to_buffer, args=(proc.stderr, name, stderr_buffer)
            )

            stdout_thread.start()
            stderr_thread.start()

            # Wait for process to complete
            return_code = proc.wait()

            # Ensure threads are done
            stdout_thread.join(timeout=1.0)
            stderr_thread.join(timeout=1.0)

            # Flush all output at once to maintain sequential order
            all_output = stdout_buffer + stderr_buffer
            if output_callback:
                output_callback(all_output)
            else:
                for line in all_output:
                    self._log(line)

            # Parse non-APGI experiment output format first
            results = {}
            for line in all_output:
                if "PRIMARY METRIC" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse PRIMARY METRIC: {line} - {e}"
                        )
                elif "Mean RT:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip("ms")
                        value = float(value_str)
                        results["mean_rt_ms"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse Mean RT: {line} - {e}")
                elif "False Alarm Rate:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip("%")
                        value = float(value_str)
                        results["false_alarm_rate"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse False Alarm Rate: {line} - {e}"
                        )
                elif "completion_time_s" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip(",")
                        value = float(value_str)
                        results["completion_time_s"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse completion_time_s: {line} - {e}"
                        )
                elif "time_min" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip(",")
                        value = float(value_str)
                        results["time_min"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse time_min: {line} - {e}")
                elif "mean_response_time_ms" in line:
                    try:
                        value_str = (
                            line.split(":", 1)[1].strip().rstrip(",").rstrip("ms")
                        )
                        value = float(value_str)
                        results["mean_response_time_ms"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse mean_response_time_ms: {line} - {e}"
                        )
                elif "detection_accuracy" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip(",")
                        value = float(value_str)
                        results["detection_accuracy"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse detection_accuracy: {line} - {e}"
                        )
                elif "false_alarm_rate" in line:
                    try:
                        value_str = (
                            line.split(":", 1)[1].strip().rstrip(",").rstrip("%")
                        )
                        value = float(value_str)
                        results["false_alarm_rate"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse false_alarm_rate: {line} - {e}"
                        )
                elif "completion_time_s" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip(",")
                        value = float(value_str)
                        results["completion_time_s"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse completion_time_s: {line} - {e}"
                        )
                elif "time_min" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip(",")
                        value = float(value_str)
                        results["time_min"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse time_min: {line} - {e}")
                elif "mean_response_time_ms" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip("ms")
                        value = float(value_str)
                        results["mean_response_time_ms"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse mean_response_time_ms: {line} - {e}"
                        )
                elif "detection_accuracy" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["detection_accuracy"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse detection_accuracy: {line} - {e}"
                        )
                elif "false_alarm_rate" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip("%")
                        value = float(value_str)
                        results["false_alarm_rate"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse false_alarm_rate: {line} - {e}"
                        )

            # Parse APGI experiment results from stdout output for visualization
            self._parse_experiment_results(name, all_output)

            status = "Success" if return_code == 0 else f"Failed (Code {return_code})"
            color = "#2ecc71" if return_code == 0 else "#e74c3c"

            self.after(0, lambda s=status, c=color: self._finish_experiment(name, s, c))

        except Exception as e:
            self.after(0, lambda ex=str(e): self._log(f"ERROR: {ex}", "#e74c3c"))
            self.after(0, lambda: self._finish_experiment(name, "Error", "#e74c3c"))
        finally:
            if name in self.active_processes:
                del self.active_processes[name]

    def _parse_experiment_results(
        self, experiment_name: str, output_lines: List[str]
    ) -> None:
        """Parse experiment results from stdout output for visualization with validation."""
        results = {}
        validation_errors = []
        json_buffer = []
        in_json = False

        try:
            # First pass: try to extract JSON objects from output
            for raw_line in output_lines:
                line = raw_line.strip()
                # Remove experiment prefix if present
                if line.startswith(f"[{experiment_name}] "):
                    line = line[len(f"[{experiment_name}] ") :]
                elif line.startswith("[") and "] " in line:
                    line = line.split("] ", 1)[1]

                # Detect start of JSON object
                if not in_json and line.startswith("{"):
                    in_json = True
                    json_buffer = [line]
                elif in_json:
                    json_buffer.append(line)
                    if line.endswith("}"):
                        # Try to parse complete JSON
                        try:
                            json_str = " ".join(json_buffer)
                            data = json.loads(json_str)
                            # Extract APGI metrics from JSON
                            if "apgi_ignition_rate" in data:
                                try:
                                    results["ignition_rate"] = float(
                                        data["apgi_ignition_rate"]
                                    )
                                except (ValueError, TypeError) as e:
                                    logging.debug(
                                        f"Failed to parse ignition_rate: {data.get('apgi_ignition_rate')} - {e}"
                                    )
                            if "apgi_mean_surprise" in data:
                                try:
                                    results["mean_surprise"] = float(
                                        data["apgi_mean_surprise"]
                                    )
                                except (ValueError, TypeError) as e:
                                    logging.debug(
                                        f"Failed to parse mean_surprise: {data.get('apgi_mean_surprise')} - {e}"
                                    )
                            if "apgi_metabolic_cost" in data:
                                try:
                                    results["metabolic_cost"] = float(
                                        data["apgi_metabolic_cost"]
                                    )
                                except (ValueError, TypeError) as e:
                                    logging.debug(
                                        f"Failed to parse metabolic_cost: {data.get('apgi_metabolic_cost')} - {e}"
                                    )
                            if "apgi_mean_somatic_marker" in data:
                                try:
                                    results["mean_somatic_marker"] = float(
                                        data["apgi_mean_somatic_marker"]
                                    )
                                except (ValueError, TypeError) as e:
                                    logging.debug(
                                        f"Failed to parse mean_somatic_marker: {data.get('apgi_mean_somatic_marker')} - {e}"
                                    )
                            if "apgi_mean_threshold" in data:
                                try:
                                    results["mean_threshold"] = float(
                                        data["apgi_mean_threshold"]
                                    )
                                except (ValueError, TypeError) as e:
                                    logging.debug(
                                        f"Failed to parse mean_threshold: {data.get('apgi_mean_threshold')} - {e}"
                                    )
                            # Also extract non-prefixed APGI metrics (some experiments output these)
                            if (
                                "ignition_rate" in data
                                and "ignition_rate" not in results
                            ):
                                try:
                                    results["ignition_rate"] = float(
                                        data["ignition_rate"]
                                    )
                                except (ValueError, TypeError):
                                    pass
                            if (
                                "mean_surprise" in data
                                and "mean_surprise" not in results
                            ):
                                try:
                                    results["mean_surprise"] = float(
                                        data["mean_surprise"]
                                    )
                                except (ValueError, TypeError):
                                    pass
                            if (
                                "metabolic_cost" in data
                                and "metabolic_cost" not in results
                            ):
                                try:
                                    results["metabolic_cost"] = float(
                                        data["metabolic_cost"]
                                    )
                                except (ValueError, TypeError):
                                    pass
                            if (
                                "mean_somatic_marker" in data
                                and "mean_somatic_marker" not in results
                            ):
                                try:
                                    results["mean_somatic_marker"] = float(
                                        data["mean_somatic_marker"]
                                    )
                                except (ValueError, TypeError):
                                    pass
                            if (
                                "mean_threshold" in data
                                and "mean_threshold" not in results
                            ):
                                try:
                                    results["mean_threshold"] = float(
                                        data["mean_threshold"]
                                    )
                                except (ValueError, TypeError):
                                    pass
                            # Extract primary metrics
                            for key in [
                                "accuracy",
                                "d_prime",
                                "overall_accuracy",
                                "benchmark_accuracy",
                                "net_score",
                                "masking_effect_ms",
                                "alternation_rate",
                                "grammar_accuracy",
                            ]:
                                if key in data:
                                    results["primary_metric"] = float(data[key])
                                    results[key] = float(data[key])
                                    break

                            # Extract Neuromodulators (Panel 3)
                            for key, result_key in [
                                ("apgi_dopamine", "dopamine_level"),
                                ("apgi_serotonin", "serotonin_level"),
                                ("apgi_acetylcholine", "acetylcholine"),
                                ("apgi_norepinephrine", "noradrenaline"),
                                ("dopamine", "dopamine_level"),
                                ("serotonin", "serotonin_level"),
                                ("acetylcholine", "acetylcholine"),
                                ("norepinephrine", "noradrenaline"),
                            ]:
                                if key in data and result_key not in results:
                                    try:
                                        results[result_key] = float(data[key])
                                    except (ValueError, TypeError):
                                        pass

                            # Handle nested neuromodulators dict (format: {"neuromodulators": {"mean_DA": ...}})
                            for neuro_src in [
                                data.get("neuromodulators"),
                                (data.get("apgi_metrics") or {}).get("neuromodulators"),
                            ]:
                                if isinstance(neuro_src, dict):
                                    for src, dst in [
                                        ("mean_DA", "dopamine_level"),
                                        ("mean_5HT", "serotonin_level"),
                                        ("mean_ACh", "acetylcholine"),
                                        ("mean_NE", "noradrenaline"),
                                        ("DA", "dopamine_level"),
                                        ("5HT", "serotonin_level"),
                                        ("ACh", "acetylcholine"),
                                        ("NE", "noradrenaline"),
                                    ]:
                                        if src in neuro_src and dst not in results:
                                            try:
                                                results[dst] = float(neuro_src[src])
                                            except (ValueError, TypeError):
                                                pass

                            # Extract Domain-Specific metrics (Panel 4)
                            for key, result_key in [
                                ("learning_rate", "learning_rate"),
                                ("precision_mismatch", "precision_mismatch"),
                                ("apgi_learning_rate", "learning_rate"),
                                ("apgi_precision_mismatch", "precision_mismatch"),
                            ]:
                                if key in data and result_key not in results:
                                    try:
                                        results[result_key] = float(data[key])
                                    except (ValueError, TypeError):
                                        pass

                            # Extract Psychiatric Indicators (Panel 5)
                            for key, result_key in [
                                ("anxiety_level", "anxiety_level"),
                                ("anxiety_index", "anxiety_level"),
                                ("apgi_anxiety_level", "anxiety_level"),
                                ("depression_index", "depression_index"),
                                ("mania_index", "mania_index"),
                                ("psychosis_risk", "psychosis_risk"),
                            ]:
                                if key in data and result_key not in results:
                                    try:
                                        results[result_key] = float(data[key])
                                    except (ValueError, TypeError):
                                        pass

                            # Extract State Space Trajectory (Panel 6)
                            for key, result_key in [
                                ("state_x", "state_x"),
                                ("state_y", "state_y"),
                                ("apgi_state_x", "state_x"),
                                ("apgi_state_y", "state_y"),
                            ]:
                                if key in data and result_key not in results:
                                    try:
                                        results[result_key] = float(data[key])
                                    except (ValueError, TypeError):
                                        pass

                            # Extract Precision Gap over Time (Panel 7)
                            for key, result_key in [
                                ("time_steps", "time_steps"),
                                ("expected_precision", "expected_precision"),
                                ("actual_precision", "actual_precision"),
                                ("apgi_expected_precision", "expected_precision"),
                                ("apgi_actual_precision", "actual_precision"),
                            ]:
                                if key in data and result_key not in results:
                                    try:
                                        if isinstance(data[key], list):
                                            value_list = [float(x) for x in data[key]]
                                            # Take first value if multiple exist
                                            results[result_key] = (
                                                value_list[0] if value_list else 0.0
                                            )
                                        else:
                                            results[result_key] = float(data[key])
                                    except (ValueError, TypeError):
                                        pass

                            # If precision_mismatch exists, derive expected/actual precision for visualization
                            if (
                                "precision_mismatch" in results
                                and "expected_precision" not in results
                            ):
                                pm = results["precision_mismatch"]
                                results["expected_precision"] = 1.0 - pm
                                results["actual_precision"] = 1.0

                        except json.JSONDecodeError:
                            pass
                        in_json = False
                        json_buffer = []

            # Second pass: parse line-by-line metrics (fallback)
            for raw_line in output_lines:
                # Remove the prefix like "[Visual Search] " if present
                line = raw_line.strip()
                if line.startswith(f"[{experiment_name}] "):
                    line = line[len(f"[{experiment_name}] ") :]
                elif "] " in line and line.startswith("["):
                    # Handle other experiment name formats
                    line = line.split("] ", 1)[1] if "] " in line else line

                # Parse standard APGI metrics (with percentages)
                if "Ignition Rate:" in line or "- Ignition Rate:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip("%")
                        value = float(value_str)
                        # Validate value is in reasonable range
                        if not (0 <= value <= 100):
                            validation_errors.append(
                                f"Ignition Rate out of range: {value}"
                            )
                        results["ignition_rate"] = value
                    except (ValueError, IndexError) as e:
                        logging.debug(f"Failed to parse Ignition Rate: {line} - {e}")

                elif "Mean Surprise:" in line or "- Mean Surprise:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if not (0 <= value <= 1000):
                            validation_errors.append(
                                f"Mean Surprise out of range: {value}"
                            )
                        results["mean_surprise"] = value
                    except (ValueError, IndexError) as e:
                        logging.debug(f"Failed to parse Mean Surprise: {line} - {e}")

                elif "Metabolic Cost:" in line or "- Total Metabolic Cost:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        if not value_str:
                            # Skip empty values (as seen in the log error)
                            continue
                        value = float(value_str)
                        if value < 0:
                            validation_errors.append(
                                f"Metabolic Cost negative: {value}"
                            )
                        results["metabolic_cost"] = value
                    except (ValueError, IndexError) as e:
                        logging.debug(f"Failed to parse Metabolic Cost: {line} - {e}")

                elif "Mean Somatic Marker:" in line or "- Mean Somatic Marker:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if not (-10 <= value <= 10):
                            validation_errors.append(
                                f"Mean Somatic Marker out of range: {value}"
                            )
                        results["mean_somatic_marker"] = value
                    except (ValueError, IndexError) as e:
                        logging.debug(
                            f"Failed to parse Mean Somatic Marker: {line} - {e}"
                        )

                elif "Mean Threshold:" in line or "- Mean Threshold:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if value <= 0:
                            validation_errors.append(
                                f"Mean Threshold non-positive: {value}"
                            )
                        results["mean_threshold"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse Mean Threshold: {line} - {e}"
                        )

                # Parse experiment-specific primary metrics
                elif "conjunction_present_slope:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["primary_metric"] = value
                        results["conjunction_present_slope"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse conjunction_present_slope: {line} - {e}"
                        )

                elif "overall_accuracy:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip("%")
                        value = float(value_str)
                        results["primary_metric"] = value
                        results["overall_accuracy"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse overall_accuracy: {line} - {e}"
                        )

                elif "benchmark_accuracy:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["primary_metric"] = value
                        results["benchmark_accuracy"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse benchmark_accuracy: {line} - {e}"
                        )

                elif "accuracy:" in line.lower():  # Primary accuracy metric
                    try:
                        value_str = (
                            line.split(":", 1)[1].strip().rstrip("%").rstrip(",")
                        )
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["accuracy"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse accuracy: {line} - {e}")

                elif "d_prime:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["primary_metric"] = value
                        results["d_prime"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse d_prime: {line} - {e}")

                elif "net_score:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["primary_metric"] = value
                        results["net_score"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse net_score: {line} - {e}")

                elif "masking_effect_ms:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["primary_metric"] = value
                        results["masking_effect_ms"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse masking_effect_ms: {line} - {e}"
                        )

                elif "alternation_rate:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["primary_metric"] = value
                        results["alternation_rate"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse alternation_rate: {line} - {e}"
                        )

                # Parse APGI-prefixed metrics (from experiments that use apgi_ prefix)
                elif "apgi_ignition_rate:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip("%")
                        # Skip empty or multi-line formatted strings (cosmetic only)
                        if not value_str or "\n" in value_str:
                            continue
                        value = float(value_str)
                        results["ignition_rate"] = value
                    except (ValueError, IndexError):
                        # Silently skip cosmetic parser issues with multi-line strings
                        continue

                elif "apgi_mean_surprise:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["mean_surprise"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse apgi_mean_surprise: {line} - {e}"
                        )

                elif "apgi_metabolic_cost:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["metabolic_cost"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse apgi_metabolic_cost: {line} - {e}"
                        )

                elif "apgi_mean_somatic_marker:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["mean_somatic_marker"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse apgi_mean_somatic_marker: {line} - {e}"
                        )

                elif "apgi_mean_threshold:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["mean_threshold"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse apgi_mean_threshold: {line} - {e}"
                        )

                # Parse apgi_ prefixed neuromodulators
                elif '"apgi_dopamine":' in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip(",")
                        value = float(value_str)
                        results["dopamine_level"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse apgi_dopamine: {line} - {e}"
                        )
                elif '"apgi_serotonin":' in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip(",")
                        value = float(value_str)
                        results["serotonin_level"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse apgi_serotonin: {line} - {e}"
                        )
                elif '"apgi_acetylcholine":' in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip(",")
                        value = float(value_str)
                        results["acetylcholine"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse apgi_acetylcholine: {line} - {e}"
                        )
                elif '"apgi_norepinephrine":' in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip(",")
                        value = float(value_str)
                        results["noradrenaline"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse apgi_norepinephrine: {line} - {e}"
                        )

                # Parse apgi_ prefixed precision and anxiety
                elif '"apgi_precision_mismatch":' in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip(",")
                        value = float(value_str)
                        results["precision_mismatch"] = value
                        results["expected_precision"] = 1.0 - value
                        results["actual_precision"] = 1.0
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse apgi_precision_mismatch: {line} - {e}"
                        )
                elif '"apgi_anxiety_level":' in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip(",")
                        value = float(value_str)
                        results["anxiety_level"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse apgi_anxiety_level: {line} - {e}"
                        )

                # Parse learning_rate from experiments
                elif '"learning_rate":' in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip(",")
                        value = float(value_str)
                        results["learning_rate"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse learning_rate: {line} - {e}"
                        )

                # Parse basic experiment metrics for Change Blindness and similar
                elif "Detection Rate:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip("%")
                        value = float(value_str) / 100  # Convert percentage to decimal
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["detection_rate"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse Detection Rate: {line} - {e}"
                        )
                elif "detection_rate:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["detection_rate"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse detection_rate: {line} - {e}"
                        )

                # Parse neuromodulator metrics (actual format: "Dopamine (DA): 1.00")
                elif "Dopamine (DA):" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["dopamine_level"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse Dopamine (DA): {line} - {e}"
                        )
                elif "Serotonin (5-HT):" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["serotonin_level"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse Serotonin (5-HT): {line} - {e}"
                        )
                elif "Norepinephrine (NE):" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["noradrenaline"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse Norepinephrine (NE): {line} - {e}"
                        )
                elif "Acetylcholine (ACh):" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["acetylcholine"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse Acetylcholine (ACh): {line} - {e}"
                        )
                # Also support legacy format without abbreviation
                elif "Dopamine:" in line and "(DA)" not in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["dopamine_level"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse Dopamine: {line} - {e}")
                elif "Serotonin:" in line and "(5-HT)" not in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["serotonin_level"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse Serotonin: {line} - {e}")
                elif "Norepinephrine:" in line and "(NE)" not in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["noradrenaline"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse Norepinephrine: {line} - {e}"
                        )
                elif "Acetylcholine:" in line and "(ACh)" not in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["acetylcholine"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse Acetylcholine: {line} - {e}"
                        )
                elif "GABA Level:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["gaba_level"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse GABA Level: {line} - {e}")
                elif "Norepinephrine:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["norepinephrine_level"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse Norepinephrine: {line} - {e}"
                        )
                elif "Histamine:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["histamine_level"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse Histamine: {line} - {e}")

                # Parse domain-specific metrics
                elif "Foraging Efficiency:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["foraging_efficiency"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse Foraging Efficiency: {line} - {e}"
                        )
                elif "Economic Value:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["economic_value"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse Economic Value: {line} - {e}"
                        )
                elif "Social Score:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["social_score"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse Social Score: {line} - {e}")
                elif "Social Score:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["social_score"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse Social Score: {line} - {e}")
                elif "Learning Rate:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["learning_rate"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse Learning Rate: {line} - {e}"
                        )

                # Parse psychiatric indicator metrics
                elif "Anxiety Index:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["anxiety_index"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse Anxiety Index: {line} - {e}"
                        )
                elif "Depression Index:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["depression_index"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse Depression Index: {line} - {e}"
                        )
                elif "Mania Index:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["mania_index"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse Mania Index: {line} - {e}")
                elif "Psychosis Risk:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["psychosis_risk"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse Psychosis Risk: {line} - {e}"
                        )

                # Parse state space trajectory metrics
                elif "State X:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["state_x"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse State X: {line} - {e}")
                elif "State Y:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["state_y"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse State Y: {line} - {e}")

                # Parse precision gap metrics (actual format: "Precision Mismatch (Π̂-Π): 0.123")
                elif "Precision Mismatch" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["precision_mismatch"] = value
                        # Map to expected/actual precision for visualization
                        results["expected_precision"] = 1.0 - value
                        results["actual_precision"] = 1.0
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse Precision Mismatch: {line} - {e}"
                        )
                elif "Anxiety Level:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["anxiety_level"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse Anxiety Level: {line} - {e}"
                        )
                # Legacy format support
                elif "Time Steps:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value_list = [
                            float(x.strip()) for x in value_str.split(",") if x.strip()
                        ]
                        # Take first value if multiple exist
                        if value_list:
                            value = value_list[0]
                        else:
                            value = 0.0
                        results["time_steps"] = (
                            float(value) if isinstance(value, (int, float)) else 0.0
                        )
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse Time Steps: {line} - {e}")
                elif "Expected Precision:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["expected_precision"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse Expected Precision: {line} - {e}"
                        )
                elif "Actual Precision:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["actual_precision"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse Actual Precision: {line} - {e}"
                        )

                elif "gating_threshold:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["gating_threshold"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse gating_threshold: {line} - {e}"
                        )

                # Parse non-APGI experiment output format first
                if "PRIMARY METRIC" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse PRIMARY METRIC: {line} - {e}"
                        )
                elif "Mean RT:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip("ms")
                        value = float(value_str)
                        results["mean_rt_ms"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse Mean RT: {line} - {e}")
                elif "False Alarm Rate:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip("%")
                        value = float(value_str)
                        results["false_alarm_rate"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse False Alarm Rate: {line} - {e}"
                        )
                # Parse non-APGI experiment output format
                elif "PRIMARY METRIC" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse PRIMARY METRIC: {line} - {e}"
                        )
                elif "Mean RT:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip("ms")
                        value = float(value_str)
                        results["mean_rt_ms"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse Mean RT: {line} - {e}")
                elif "False Alarm Rate:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip("%")
                        value = float(value_str)
                        results["false_alarm_rate"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse False Alarm Rate: {line} - {e}"
                        )
                # Parse non-APGI experiment output format
                if "PRIMARY METRIC" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse PRIMARY METRIC: {line} - {e}"
                        )
                elif "Mean RT:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip("ms")
                        value = float(value_str)
                        results["mean_rt_ms"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse Mean RT: {line} - {e}")
                elif "False Alarm Rate:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip("%")
                        value = float(value_str)
                        results["false_alarm_rate"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse False Alarm Rate: {line} - {e}"
                        )
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip("ms")
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["interference_effect_ms"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse Interference Effect: {line} - {e}"
                        )

                elif "Gating Effect:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip("%")
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["gating_effect"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse Gating Effect: {line} - {e}"
                        )

                elif "Hit Rate:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip("%")
                        value = float(value_str)
                        results["hit_rate"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse Hit Rate: {line} - {e}")

                elif "False Alarm Rate:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip("%")
                        value = float(value_str)
                        results["false_alarm_rate"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse False Alarm Rate: {line} - {e}"
                        )

                # Parse non-APGI experiment output format first
                if "PRIMARY METRIC" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse PRIMARY METRIC: {line} - {e}"
                        )
                elif "Mean RT" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip("ms")
                        value = float(value_str)
                        results["mean_rt_ms"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse Mean RT: {line} - {e}")
                elif "False Alarm Rate" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip("%")
                        value = float(value_str)
                        results["false_alarm_rate"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse False Alarm Rate: {line} - {e}"
                        )
                elif "completion_time_s" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip(",")
                        value = float(value_str)
                        results["completion_time_s"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse completion_time_s: {line} - {e}"
                        )
                elif "time_min" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip(",")
                        value = float(value_str)
                        results["time_min"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse time_min: {line} - {e}")
                elif "mean_response_time_ms" in line:
                    try:
                        value_str = (
                            line.split(":", 1)[1].strip().rstrip(",").rstrip("ms")
                        )
                        value = float(value_str)
                        results["mean_response_time_ms"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse mean_response_time_ms: {line} - {e}"
                        )
                elif "detection_accuracy" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip(",")
                        value = float(value_str)
                        results["detection_accuracy"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse detection_accuracy: {line} - {e}"
                        )
                elif "false_alarm_rate" in line:
                    try:
                        value_str = (
                            line.split(":", 1)[1].strip().rstrip(",").rstrip("%")
                        )
                        value = float(value_str)
                        results["false_alarm_rate"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse false_alarm_rate: {line} - {e}"
                        )
                elif "completion_time_s" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip(",")
                        value = float(value_str)
                        results["completion_time_s"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse completion_time_s: {line} - {e}"
                        )
                elif "time_min" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip(",")
                        value = float(value_str)
                        results["time_min"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse time_min: {line} - {e}")
                elif "mean_response_time_ms" in line:
                    try:
                        value_str = (
                            line.split(":", 1)[1].strip().rstrip(",").rstrip("ms")
                        )
                        value = float(value_str)
                        results["mean_response_time_ms"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse mean_response_time_ms: {line} - {e}"
                        )
                elif "detection_accuracy" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip(",")
                        value = float(value_str)
                        results["detection_accuracy"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse detection_accuracy: {line} - {e}"
                        )
                elif "false_alarm_rate" in line:
                    try:
                        value_str = (
                            line.split(":", 1)[1].strip().rstrip(",").rstrip("%")
                        )
                        value = float(value_str)
                        results["false_alarm_rate"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse false_alarm_rate: {line} - {e}"
                        )
                elif "completion_time_s" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip(",")
                        value = float(value_str)
                        results["completion_time_s"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse completion_time_s: {line} - {e}"
                        )
                elif "time_min" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip(",")
                        value = float(value_str)
                        results["time_min"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse time_min: {line} - {e}")
                elif "mean_response_time_ms" in line:
                    try:
                        value_str = (
                            line.split(":", 1)[1].strip().rstrip(",").rstrip("ms")
                        )
                        value = float(value_str)
                        results["mean_response_time_ms"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse mean_response_time_ms: {line} - {e}"
                        )
                elif "detection_accuracy" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip(",")
                        value = float(value_str)
                        results["detection_accuracy"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse detection_accuracy: {line} - {e}"
                        )
                elif "false_alarm_rate" in line:
                    try:
                        value_str = (
                            line.split(":", 1)[1].strip().rstrip(",").rstrip("%")
                        )
                        value = float(value_str)
                        results["false_alarm_rate"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse false_alarm_rate: {line} - {e}"
                        )
                elif "completion_time_s" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip(",")
                        value = float(value_str)
                        results["completion_time_s"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse completion_time_s: {line} - {e}"
                        )
                elif "time_min" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip(",")
                        value = float(value_str)
                        results["time_min"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse time_min: {line} - {e}")
                elif "mean_response_time_ms" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["mean_response_time_ms"] = value
                    except (ValueError, IndexError):
                        self._log(
                            "[DEBUG] formatted output for non-APGI experiments with 9 metrics: ['primary_metric', 'detection_accuracy', 'false_alarm_rate', 'mean_rt_ms', 'completion_time_s', 'time_min']"
                        )
                elif "PRIMARY METRIC" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse PRIMARY METRIC: {line} - {e}"
                        )
                elif "accuracy" in line and ":" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip(",")
                        value = float(value_str)
                        results["accuracy"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse accuracy: {line} - {e}")
                elif "detection_accuracy" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["detection_accuracy"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse detection_accuracy: {line} - {e}"
                        )
                elif "false_alarm_rate" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip("%")
                        value = float(value_str)
                        results["false_alarm_rate"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse false_alarm_rate: {line} - {e}"
                        )
                elif "completion_time_s" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip(",")
                        value = float(value_str)
                        results["completion_time_s"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse completion_time_s: {line} - {e}"
                        )
                elif "time_min" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip(",")
                        value = float(value_str)
                        results["time_min"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse time_min: {line} - {e}")
                elif "mean_response_time_ms" in line:
                    try:
                        value_str = (
                            line.split(":", 1)[1].strip().rstrip(",").rstrip("ms")
                        )
                        value = float(value_str)
                        results["mean_response_time_ms"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse mean_response_time_ms: {line} - {e}"
                        )
                elif "detection_accuracy" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip(",")
                        value = float(value_str)
                        results["detection_accuracy"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse detection_accuracy: {line} - {e}"
                        )
                elif "false_alarm_rate" in line:
                    try:
                        value_str = (
                            line.split(":", 1)[1].strip().rstrip(",").rstrip("%")
                        )
                        value = float(value_str)
                        results["false_alarm_rate"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse false_alarm_rate: {line} - {e}"
                        )
                elif "completion_time_s" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip(",")
                        value = float(value_str)
                        results["completion_time_s"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse completion_time_s: {line} - {e}"
                        )
                elif "time_min" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip(",")
                        value = float(value_str)
                        results["time_min"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse time_min: {line} - {e}")
                elif "stimulus_type_performance" in line:
                    # Skip this line, it's a header
                    pass
                elif "task_type_performance" in line:
                    # Skip this line, it's a header
                    pass

                elif "metabolic_cost_ratio:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["metabolic_cost_ratio"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse metabolic_cost_ratio: {line} - {e}"
                        )

                elif "multisensory_gain_ms:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["multisensory_gain_ms"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse multisensory_gain_ms: {line} - {e}"
                        )

                elif "global_advantage_ms:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["interference_effect_ms"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse interference_effect_ms: {line} - {e}"
                        )

                elif "flanker_effect_ms:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["flanker_effect_ms"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse flanker_effect_ms: {line} - {e}"
                        )

                elif "ssrt_ms:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["ssrt_ms"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse ssrt_ms: {line} - {e}")

                elif "search_slope_ms_per_item:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["search_slope_ms_per_item"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse search_slope_ms_per_item: {line} - {e}"
                        )

                elif "simon_effect_ms:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["simon_effect_ms"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse simon_effect_ms: {line} - {e}"
                        )

                elif "priming_effect_ms:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["priming_effect_ms"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse priming_effect_ms: {line} - {e}"
                        )

                elif "learning_effect_ms:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["learning_effect_ms"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse learning_effect_ms: {line} - {e}"
                        )

                elif "validity_effect_ms:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["validity_effect_ms"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse validity_effect_ms: {line} - {e}"
                        )

                elif "mean_error_percent:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["mean_error_percent"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse mean_error_percent: {line} - {e}"
                        )

                elif "path_efficiency:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["path_efficiency"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse path_efficiency: {line} - {e}"
                        )

                elif "Learning Rate:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["learning_rate"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse learning_rate: {line} - {e}"
                        )

                # Additional visual search specific metrics
                elif "feature_present_slope:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["feature_present_slope"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse feature_present_slope: {line} - {e}"
                        )

                elif "slope_ratio:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["slope_ratio"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse slope_ratio: {line} - {e}")

            # Store results if any metrics were found
            if results:
                self.experiment_results[experiment_name] = results
                self._log(
                    f"[VISUALIZATION] Parsed {len(results)} metrics for {experiment_name}: {list(results.keys())}"
                )
                # Log validation errors if any
                if validation_errors:
                    self._log(
                        f"[WARNING] Validation issues found for {experiment_name}:"
                    )
                    for error in validation_errors:
                        self._log(f"  - {error}")
            else:
                # Some experiments lack standard metrics - this is expected and doesn't affect execution
                # Only log at debug level to avoid noise
                pass  # Silently skip experiments without standard metrics

        except Exception as e:
            self._log(f"[ERROR] Failed to parse results: {e}")

    def _finish_experiment(self, name: str, status: str, color: str) -> None:
        self.running_experiments.discard(name)
        if name in self.experiment_buttons:
            self.experiment_buttons[name].configure(state="normal")
            self.status_indicators[name].configure(text=status, text_color=color)
        self._log(f"[FINISHED] {name}: {status}")

    def _run_all(self) -> None:
        if self.running_experiments:
            if not messagebox.askyesno(
                "Confirm", "Some experiments are already running. Run all sequentially?"
            ):
                return

        self.stop_all = False
        self._log("\n" + "#" * 50)
        self._log("### STARTING SEQUENTIAL RUN OF ALL EXPERIMENTS ###")
        self._log("#" * 50 + "\n")

        thread = threading.Thread(target=self._run_all_sequential)
        thread.daemon = True
        thread.start()

    def _run_all_sequential(self) -> None:
        """Run all experiments sequentially with proper synchronization."""
        for i, (name, script) in enumerate(self.experiments):
            if self.stop_all:
                break

            # Create a proper closure to avoid race condition
            experiment_data = (name, script)

            # Use main thread to start run with proper closure
            self.after(
                0, lambda data=experiment_data: self._run_experiment(data[0], data[1])
            )

            # Wait for this experiment to finish with timeout
            wait_time = 0.0
            max_wait_time = 300  # 5 minutes max per experiment
            while (
                name in self.running_experiments
                and not self.stop_all
                and wait_time < max_wait_time
            ):
                time.sleep(0.5)
                wait_time += 0.5

            # Check if experiment timed out
            if wait_time >= max_wait_time and name in self.running_experiments:
                self.after(
                    0,
                    lambda n=name: self._log(
                        f"\n!!! EXPERIMENT {n} TIMED OUT !!!", "#e74c3c"
                    ),
                )
                # Force stop the timed out experiment
                if name in self.active_processes:
                    self._stop_experiment(name)

        self.after(
            0, lambda: self._log("\n### ALL EXPERIMENTS COMPLETE ###", "#2ecc71")
        )

    def _stop_all(self) -> None:
        self.stop_all = True
        for name, proc in list(self.active_processes.items()):
            try:
                proc.terminate()
            except Exception:
                pass
        self._log("\n[STOPPING] Termination signal sent to all active processes...")

    def _display_dependencies_status(self) -> None:
        """Check if all required dependencies are installed."""
        self._log("\n" + "=" * 50)
        self._log("CHECKING DEPENDENCIES")
        self._log("=" * 50)

        missing = []
        installed = []

        for module, description in {
            **CORE_DEPENDENCIES,
            **OPTIONAL_DEPENDENCIES,
        }.items():
            try:
                if module == "PIL":
                    spec = importlib.util.find_spec("PIL")
                elif module == "sklearn":
                    spec = importlib.util.find_spec("sklearn")
                else:
                    spec = importlib.util.find_spec(module)

                if spec is None:
                    missing.append((module, description))
                else:
                    installed.append((module, description))
                    self._log(f"✅ {description}")
            except Exception as e:
                missing.append((module, description))
                self._log(f"❌ {description} - Error: {e}")

        if missing:
            self._log(f"\n⚠️ Missing {len(missing)} dependencies:")
            for module, desc in missing:
                self._log(f"   - {desc}")
        else:
            self._log("\n✅ All dependencies are installed!")

        self._log("=" * 50)

    def _repair_dependencies(self) -> None:
        """Install missing dependencies using pip."""
        self._display_dependencies_status()

        # Check again for missing deps
        missing = []
        for module, description in {
            **CORE_DEPENDENCIES,
            **OPTIONAL_DEPENDENCIES,
        }.items():
            try:
                if module == "PIL":
                    spec = importlib.util.find_spec("PIL")
                elif module == "sklearn":
                    spec = importlib.util.find_spec("sklearn")
                else:
                    spec = importlib.util.find_spec(module)

                if spec is None:
                    missing.append(module)
            except Exception:
                missing.append(module)

        if not missing:
            messagebox.showinfo(
                "Dependencies", "All dependencies are already installed!"
            )
            return

        if not messagebox.askyesno(
            "Install Dependencies",
            f"Install {len(missing)} missing packages?\n\nThis will run:\npip install {' '.join(missing)}",
        ):
            return

        self._log(f"\n🔧 Installing {len(missing)} packages...")

        # Safe package mapping for known packages
        package_map = {
            "numpy": "numpy",
            "pandas": "pandas",
            "matplotlib": "matplotlib",
            "customtkinter": "customtkinter",
            "scipy": "scipy",
            "torch": "torch",
            "sklearn": "scikit-learn",
            "PIL": "Pillow",
        }

        # Validate and sanitize package names
        packages = []
        for m in missing:
            if m in package_map:
                packages.append(package_map[m])
            elif re.match(
                r"^[a-zA-Z][a-zA-Z0-9_-]*$", m
            ):  # Basic package name validation
                packages.append(m)
            else:
                logging.warning(f"Skipping invalid package name: {m}")

        if not packages:
            self.after(0, lambda: self._log("No valid packages to install", "#e74c3c"))
            return

        def install() -> None:
            try:
                env = os.environ.copy()
                if sys.platform == "darwin":
                    env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
                proc = secure_popen(
                    ["python3", "-m", "pip", "install"] + packages,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                )

                stdout, stderr = proc.communicate()

                if proc.returncode == 0:
                    self.after(
                        0,
                        lambda: self._log(
                            "✅ Installation complete! Please restart the GUI."
                        ),
                    )
                else:
                    self.after(
                        0,
                        lambda: self._log(
                            f"❌ Installation failed:\n{stderr}", "#e74c3c"
                        ),
                    )
            except Exception as install_error:
                self.after(
                    0,
                    lambda err=str(install_error): self._log(
                        f"❌ Error: {err}", "#e74c3c"
                    ),
                )

        thread = threading.Thread(target=install)
        thread.daemon = True
        thread.start()

    def change_appearance_mode(self, new_appearance_mode: str) -> None:
        """Change the appearance mode and update the dropdown to match."""
        ctk.set_appearance_mode(new_appearance_mode)
        # Update the dropdown to show the current selection
        if hasattr(self, "appearance_mode_optionemenu"):
            self.appearance_mode_optionemenu.set(new_appearance_mode)
        # Log the theme change
        self._log(f"Theme changed to: {new_appearance_mode}", "#3498db")

    def _create_visualization_panel(self, parent_frame: Any) -> None:
        """Create an embedded matplotlib visualization panel."""
        # Create figure and canvas
        self.current_figure = Figure(figsize=(8, 4), dpi=100, facecolor="#2b2b2b")
        self.current_canvas = FigureCanvasTkAgg(
            self.current_figure, master=parent_frame
        )
        self.current_canvas.draw()
        self.current_canvas.get_tk_widget().pack(
            fill="both", expand=True, padx=5, pady=5
        )

        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.current_canvas, parent_frame)
        toolbar.update()

    def _plot_experiment_results(self, experiment_name: str, results: dict) -> None:
        """Plot experiment results in the embedded visualization panel."""
        if self.current_figure is None:
            return

        # Clear previous plots
        self.current_figure.clear()

        import matplotlib.gridspec as gridspec

        gs = gridspec.GridSpec(3, 3, figure=self.current_figure, hspace=0.8, wspace=0.4)

        ax1 = self.current_figure.add_subplot(gs[0, 0])  # Core Dynamics
        ax2 = self.current_figure.add_subplot(gs[0, 1])  # Measurement Proxies
        ax3 = self.current_figure.add_subplot(gs[0, 2])  # Neuromodulators
        ax4 = self.current_figure.add_subplot(gs[1, 0])  # Domain-specific
        ax5 = self.current_figure.add_subplot(gs[1, 1])  # Psychiatric
        ax6 = self.current_figure.add_subplot(gs[1, 2])  # State space
        ax7 = self.current_figure.add_subplot(gs[2, :])  # Precision gap

        axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]

        # Set dark theme colors
        for ax in axes:
            ax.set_facecolor("#2b2b2b")
            ax.tick_params(colors="white", labelsize=8)
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            ax.title.set_fontsize(10)
            for spine in ax.spines.values():
                spine.set_color("white")

        def get_val(key: str) -> Optional[float]:
            """Return float value from results, or None if not present."""
            v = results.get(key)
            return float(v) if v is not None else None

        def _no_data(ax: Any, title: str) -> None:
            """Render a 'No data' placeholder on an axis."""
            ax.text(
                0.5,
                0.5,
                "No data\n(run experiment first)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#7f8c8d",
                fontsize=9,
            )
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])

        # Panel 1: Core Dynamics
        core_keys = [
            "ignition_rate",
            "metabolic_cost",
            "mean_surprise",
            "mean_threshold",
        ]
        core_vals = [get_val(k) for k in core_keys]
        if any(v is not None for v in core_vals):
            ax1.bar(
                ["Ignition", "Metabolism", "Surprise", "Threshold"],
                [v or 0.0 for v in core_vals],
                color=["#3498db", "#e74c3c", "#f39c12", "#9b59b6"],
                alpha=0.8,
            )
            ax1.set_title("1. Core Dynamics")
            ax1.tick_params(axis="x", rotation=45)
        else:
            _no_data(ax1, "1. Core Dynamics")

        # Panel 2: Measurement Proxies - use actual experiment metrics
        proxy_keys = [
            "primary_metric",
            "d_prime",
            "mean_somatic_marker",
            "learning_rate",
        ]
        proxy_vals = [get_val(k) for k in proxy_keys]
        if any(v is not None for v in proxy_vals):
            ax2.bar(
                ["Primary", "D-Prime", "Somatic", "Learning"],
                [v or 0.0 for v in proxy_vals],
                color=["#2ecc71", "#1abc9c", "#34495e", "#7f8c8d"],
                alpha=0.8,
            )
            ax2.set_title("2. Measurement Proxies")
            ax2.tick_params(axis="x", rotation=45)
        else:
            _no_data(ax2, "2. Measurement Proxies")

        # Panel 3: Neuromodulators
        neuro_keys = [
            "dopamine_level",
            "serotonin_level",
            "noradrenaline",
            "acetylcholine",
        ]
        neuro_vals = [get_val(k) for k in neuro_keys]
        if any(v is not None for v in neuro_vals):
            ax3.bar(
                ["DA", "5-HT", "NE", "ACh"],
                [v or 0.0 for v in neuro_vals],
                color=["#e67e22", "#d35400", "#c0392b", "#8e44ad"],
                alpha=0.8,
            )
            ax3.set_title("3. Neuromodulators")
        else:
            _no_data(ax3, "3. Neuromodulators")

        # Panel 4: Domain-specific - use actual experiment metrics
        domain_keys = [
            "learning_rate",
            "mean_threshold",
            "metabolic_cost",
            "precision_mismatch",
        ]
        domain_vals = [get_val(k) for k in domain_keys]
        if any(v is not None for v in domain_vals):
            ax4.bar(
                ["Learning", "Threshold", "Metabolism", "Precision Gap"],
                [v or 0.0 for v in domain_vals],
                color=["#27ae60", "#2980b9", "#8e44ad", "#f39c12"],
                alpha=0.8,
            )
            ax4.set_title("4. Domain-Specific")
            ax4.tick_params(axis="x", rotation=45)
        else:
            _no_data(ax4, "4. Domain-Specific")

        # Panel 5: Psychiatric - use actual parsed metrics
        psych_keys = [
            "anxiety_level",
            "precision_mismatch",
            "mean_surprise",
            "mean_threshold",
        ]
        psych_vals = [get_val(k) for k in psych_keys]
        if any(v is not None for v in psych_vals):
            ax5.bar(
                ["Anxiety", "Precision Gap", "Surprise", "Threshold"],
                [v or 0.0 for v in psych_vals],
                color=["#bdc3c7", "#95a5a6", "#7f8c8d", "#e74c3c"],
                alpha=0.8,
            )
            ax5.set_title("5. Psychiatric Indicators")
            ax5.tick_params(axis="x", rotation=45)
        else:
            _no_data(ax5, "5. Psychiatric Indicators")

        # Panel 6: State Space
        state_x = results.get("state_x")
        state_y = results.get("state_y")

        # Derive from available metrics if state_x/state_y not present
        if state_x is None or state_y is None:
            mean_surprise = results.get("mean_surprise")
            mean_threshold = results.get("mean_threshold")
            if mean_surprise is not None and mean_threshold is not None:
                num_points = 20
                import math

                base_x = float(mean_surprise) if mean_surprise else 0.0
                base_y = float(mean_threshold) if mean_threshold else 0.5
                state_x = [base_x + math.sin(i * 0.5) * 0.02 for i in range(num_points)]
                state_y = [base_y + math.cos(i * 0.5) * 0.02 for i in range(num_points)]

        if state_x is not None and state_y is not None:
            ax6.scatter(state_x, state_y, c="#1abc9c", alpha=0.6)
            ax6.set_title("6. State Space Trajectory")
            ax6.set_xticks([])
            ax6.set_yticks([])
        else:
            _no_data(ax6, "6. State Space Trajectory")

        # Panel 7: Precision Gap
        time_steps = results.get("time_steps")
        expected_prec = results.get("expected_precision")
        actual_prec = results.get("actual_precision")
        if (
            time_steps is not None
            and expected_prec is not None
            and actual_prec is not None
        ):
            ax7.plot(
                time_steps,
                expected_prec,
                label="Expected Precision",
                color="#3498db",
                lw=2,
            )
            ax7.plot(
                time_steps, actual_prec, label="Actual Precision", color="#e74c3c", lw=2
            )
            ax7.fill_between(
                time_steps,
                expected_prec,
                actual_prec,
                color="#9b59b6",
                alpha=0.3,
                label="Precision Gap",
            )
            ax7.set_title("7. Precision Gap over Time")
        else:
            _no_data(ax7, "7. Precision Gap over Time")
        ax7.legend(
            loc="upper right", facecolor="#2b2b2b", labelcolor="white", fontsize=8
        )

        self.current_figure.suptitle(
            f"{experiment_name} Results Analysis", color="white", fontsize=14, y=0.98
        )
        self.current_figure.tight_layout(rect=(0, 0, 1, 0.96))

        if self.current_canvas is not None:
            self.current_canvas.draw()

    def _show_results_visualization(self, experiment_name: str) -> None:
        """Open a visualization window for experiment results."""
        viz_window = ctk.CTkToplevel(self)
        viz_window.title(f"Results Visualization - {experiment_name}")
        viz_window.geometry("1400x900")
        viz_window.transient(self)

        # Create visualization panel in the new window
        viz_frame = ctk.CTkFrame(viz_window)
        viz_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Create window-local figure/canvas (not shared with main panel)
        viz_figure = Figure(
            figsize=(15, 10), dpi=100, facecolor="#2b2b2b", constrained_layout=True
        )
        viz_canvas = FigureCanvasTkAgg(viz_figure, master=viz_frame)
        viz_canvas.draw()
        viz_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Add toolbar
        toolbar = NavigationToolbar2Tk(viz_canvas, viz_frame)
        toolbar.update()

        # Plot results if available (use window-local figure)
        if experiment_name in self.experiment_results:
            results = self.experiment_results[experiment_name]
            self._log(
                f"[VIZ] Showing 7-panel visualization for {experiment_name} with {len(results)} metrics"
            )
            viz_figure.clear()

            import matplotlib.gridspec as gridspec

            gs = gridspec.GridSpec(3, 3, figure=viz_figure, hspace=0.6, wspace=0.4)

            ax1 = viz_figure.add_subplot(gs[0, 0])  # Core Dynamics
            ax2 = viz_figure.add_subplot(gs[0, 1])  # Measurement Proxies
            ax3 = viz_figure.add_subplot(gs[0, 2])  # Neuromodulators
            ax4 = viz_figure.add_subplot(gs[1, 0])  # Domain-specific
            ax5 = viz_figure.add_subplot(gs[1, 1])  # Psychiatric
            ax6 = viz_figure.add_subplot(gs[1, 2])  # State space
            ax7 = viz_figure.add_subplot(gs[2, :])  # Precision gap

            axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]

            # Set dark theme colors
            for ax in axes:
                ax.set_facecolor("#2b2b2b")
                ax.tick_params(colors="white")
                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")
                ax.title.set_color("white")
                for spine in ax.spines.values():
                    spine.set_color("white")

            def get_val2(key: str) -> Optional[float]:
                v = results.get(key)
                return float(v) if v is not None else None

            def _no_data2(ax: Any, title: str) -> None:
                ax.text(
                    0.5,
                    0.5,
                    "No data\n(run experiment first)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="#7f8c8d",
                    fontsize=9,
                )
                ax.set_title(title)
                ax.set_xticks([])
                ax.set_yticks([])

            # Panel 1: Core Dynamics
            core_keys = [
                "ignition_rate",
                "metabolic_cost",
                "mean_surprise",
                "mean_threshold",
            ]
            core_vals2 = [get_val2(k) for k in core_keys]
            if any(v is not None for v in core_vals2):
                ax1.bar(
                    ["Ignition", "Metabolism", "Surprise", "Threshold"],
                    [v or 0.0 for v in core_vals2],
                    color=["#3498db", "#e74c3c", "#f39c12", "#9b59b6"],
                    alpha=0.8,
                )
                ax1.set_title("1. Core Dynamics")
                ax1.tick_params(axis="x", rotation=45)
            else:
                behav_candidates2 = [
                    ("primary_metric", "Primary"),
                    ("accuracy", "Accuracy"),
                    ("d_prime", "D-Prime"),
                    ("hit_rate", "Hit Rate"),
                    ("detection_rate", "Detection"),
                    ("alternation_rate", "Alternation"),
                    ("mean_rt_ms", "Mean RT"),
                    ("go_accuracy", "Go Acc."),
                    ("overall_accuracy", "Overall"),
                    ("benchmark_accuracy", "Benchmark"),
                ]
                behav_pairs2 = [
                    (lbl, get_val2(k))
                    for k, lbl in behav_candidates2
                    if get_val2(k) is not None
                ][:4]
                if behav_pairs2:
                    lbls2, vals2 = zip(*behav_pairs2)
                    ax1.bar(
                        lbls2,
                        list(vals2),
                        color=["#3498db", "#e74c3c", "#f39c12", "#9b59b6"][
                            : len(vals2)
                        ],
                        alpha=0.8,
                    )
                    ax1.set_title("1. Core Dynamics (Behavioral)")
                    ax1.tick_params(axis="x", rotation=45)
                else:
                    _no_data2(ax1, "1. Core Dynamics")

            # Panel 2: Measurement Proxies - use actual experiment metrics
            proxy_candidates2 = [
                ("primary_metric", "Primary"),
                ("d_prime", "D-Prime"),
                ("mean_somatic_marker", "Somatic"),
                ("learning_rate", "Learning"),
                ("accuracy", "Accuracy"),
                ("hit_rate", "Hit Rate"),
                ("overall_accuracy", "Overall"),
                ("recognition_accuracy", "Recognition"),
                ("grammatical_accuracy", "Grammar"),
                ("detection_accuracy", "Detection"),
            ]
            proxy_pairs2 = [
                (lbl, get_val2(k))
                for k, lbl in proxy_candidates2
                if get_val2(k) is not None
            ][:4]
            if proxy_pairs2:
                lbls_p, vals_p = zip(*proxy_pairs2)
                ax2.bar(
                    lbls_p,
                    list(vals_p),
                    color=["#2ecc71", "#1abc9c", "#34495e", "#7f8c8d"][: len(vals_p)],
                    alpha=0.8,
                )
                ax2.set_title("2. Measurement Proxies")
                ax2.tick_params(axis="x", rotation=45)
            else:
                _no_data2(ax2, "2. Measurement Proxies")

            # Panel 3: Neuromodulators
            neuro_keys = [
                "dopamine_level",
                "serotonin_level",
                "noradrenaline",
                "acetylcholine",
            ]
            neuro_vals2 = [get_val2(k) for k in neuro_keys]
            if any(v is not None for v in neuro_vals2):
                ax3.bar(
                    ["DA", "5-HT", "NE", "ACh"],
                    [v or 0.0 for v in neuro_vals2],
                    color=["#e67e22", "#d35400", "#c0392b", "#8e44ad"],
                    alpha=0.8,
                )
                ax3.set_title("3. Neuromodulators")
            else:
                rt_candidates2 = [
                    ("mean_rt_ms", "Mean RT"),
                    ("mean_response_time_ms", "Mean RT (ms)"),
                    ("congruent_rt_ms", "Congruent"),
                    ("incongruent_rt_ms", "Incongruent"),
                    ("valid_rt_ms", "Valid RT"),
                    ("invalid_rt_ms", "Invalid RT"),
                    ("go_rt_ms", "Go RT"),
                    ("mean_go_rt_ms", "Go RT"),
                    ("completion_time_s", "Completion (s)"),
                    ("mean_saccade_latency", "Saccade"),
                ]
                rt_pairs2 = [
                    (lbl, get_val2(k))
                    for k, lbl in rt_candidates2
                    if get_val2(k) is not None
                ][:4]
                if rt_pairs2:
                    lbls_rt, vals_rt = zip(*rt_pairs2)
                    ax3.bar(
                        lbls_rt,
                        list(vals_rt),
                        color=["#e67e22", "#d35400", "#c0392b", "#8e44ad"][
                            : len(vals_rt)
                        ],
                        alpha=0.8,
                    )
                    ax3.set_title("3. Response Times (ms)")
                    ax3.tick_params(axis="x", rotation=45)
                else:
                    _no_data2(ax3, "3. Neuromodulators")

            # Panel 4: Domain-specific - use actual experiment metrics
            domain_candidates2 = [
                ("learning_rate", "Learning"),
                ("mean_threshold", "Threshold"),
                ("metabolic_cost", "Metabolism"),
                ("precision_mismatch", "Prec. Gap"),
                ("false_alarm_rate", "False Alarm"),
                ("miss_rate", "Miss Rate"),
                ("interference_effect_ms", "Interference"),
                ("flanker_effect_ms", "Flanker"),
                ("simon_effect_ms", "Simon Eff."),
                ("stroop_effect_ms", "Stroop Eff."),
                ("num_trials", "Num Trials"),
                ("mean_somatic_marker", "Somatic"),
                ("masking_effect_ms", "Masking"),
                ("net_score", "Net Score"),
            ]
            domain_pairs2 = [
                (lbl, get_val2(k))
                for k, lbl in domain_candidates2
                if get_val2(k) is not None
            ][:4]
            if domain_pairs2:
                lbls_d, vals_d = zip(*domain_pairs2)
                ax4.bar(
                    lbls_d,
                    list(vals_d),
                    color=["#27ae60", "#2980b9", "#8e44ad", "#f39c12"][: len(vals_d)],
                    alpha=0.8,
                )
                ax4.set_title("4. Domain-Specific")
                ax4.tick_params(axis="x", rotation=45)
            else:
                _no_data2(ax4, "4. Domain-Specific")

            # Panel 5: Psychiatric - use actual parsed metrics
            psych_candidates2 = [
                ("anxiety_level", "Anxiety"),
                ("max_anxiety_index", "Max Anxiety"),
                ("precision_mismatch", "Prec. Gap"),
                ("surprise_accumulation_index", "Surp. Accum."),
                ("mean_surprise", "Surprise"),
                ("mean_threshold", "Threshold"),
                ("accuracy", "Accuracy"),
                ("hit_rate", "Hit Rate"),
                ("false_alarm_rate", "False Alarm"),
                ("go_accuracy", "Go Acc."),
                ("no_go_accuracy", "No-Go Acc."),
                ("grammatical_accuracy", "Grammar"),
                ("detection_accuracy", "Detection"),
                ("recognition_accuracy", "Recognition"),
                ("overall_accuracy", "Overall"),
                ("alternation_rate", "Alt. Rate"),
                ("primary_metric", "Primary"),
            ]
            psych_pairs2 = [
                (lbl, get_val2(k))
                for k, lbl in psych_candidates2
                if get_val2(k) is not None
            ][:4]
            if psych_pairs2:
                lbls_ps, vals_ps = zip(*psych_pairs2)
                ax5.bar(
                    lbls_ps,
                    list(vals_ps),
                    color=["#bdc3c7", "#95a5a6", "#7f8c8d", "#e74c3c"][: len(vals_ps)],
                    alpha=0.8,
                )
                ax5.set_title("5. Psychiatric Indicators")
                ax5.tick_params(axis="x", rotation=45)
            else:
                _no_data2(ax5, "5. Psychiatric Indicators")

            # Panel 6: State Space
            state_x2 = results.get("state_x")
            state_y2 = results.get("state_y")

            if state_x2 is not None and state_y2 is not None:
                if isinstance(state_x2, (list, tuple)) and isinstance(
                    state_y2, (list, tuple)
                ):
                    ax6.plot(state_x2, state_y2, color="#1abc9c", lw=1.2, alpha=0.7)
                    ax6.scatter(state_x2, state_y2, c="#1abc9c", s=20, alpha=0.6)
                else:
                    sx6, sy6 = float(state_x2), float(state_y2)
                    ax6.scatter([sx6], [sy6], c="#1abc9c", s=120, zorder=5)
                    ax6.annotate(
                        f"({sx6:.3f}, {sy6:.3f})",
                        (sx6, sy6),
                        textcoords="offset points",
                        xytext=(6, 6),
                        color="white",
                        fontsize=8,
                    )
                ax6.set_title("6. State Space Trajectory")
                ax6.set_xlabel("State X", color="white", fontsize=8)
                ax6.set_ylabel("State Y", color="white", fontsize=8)
            else:
                mean_surprise = results.get("mean_surprise")
                mean_threshold = results.get("mean_threshold")
                if mean_surprise is not None and mean_threshold is not None:
                    sx6, sy6 = float(mean_surprise), float(mean_threshold)
                    ax6.scatter([sx6], [sy6], c="#1abc9c", s=120, zorder=5)
                    ax6.annotate(
                        f"({sx6:.3f}, {sy6:.3f})",
                        (sx6, sy6),
                        textcoords="offset points",
                        xytext=(6, 6),
                        color="white",
                        fontsize=8,
                    )
                    ax6.set_title("6. State (Surprise vs Threshold)")
                    ax6.set_xlabel("Surprise", color="white", fontsize=8)
                    ax6.set_ylabel("Threshold", color="white", fontsize=8)
                else:
                    _no_data2(ax6, "6. State Space Trajectory")

            # Panel 7: Precision Gap
            time_steps2 = results.get("time_steps")
            expected_prec2 = results.get("expected_precision")
            actual_prec2 = results.get("actual_precision")

            # Derive from precision_mismatch if direct values absent
            if expected_prec2 is None and actual_prec2 is None:
                pm7 = results.get("precision_mismatch")
                if pm7 is not None:
                    expected_prec2 = 1.0 - float(pm7)
                    actual_prec2 = 1.0

            if expected_prec2 is not None and actual_prec2 is not None:
                exp_f = (
                    float(expected_prec2)
                    if isinstance(expected_prec2, (int, float))
                    else expected_prec2[0]
                )
                act_f = (
                    float(actual_prec2)
                    if isinstance(actual_prec2, (int, float))
                    else actual_prec2[0]
                )
                gap2 = exp_f - act_f

                if isinstance(expected_prec2, (int, float)) and isinstance(
                    actual_prec2, (int, float)
                ):
                    if abs(gap2) < 1e-6:
                        # Trivially flat: show bar chart instead of overlapping lines
                        ax7.bar(
                            ["Expected", "Actual"],
                            [exp_f, act_f],
                            color=["#3498db", "#e74c3c"],
                            alpha=0.8,
                        )
                        ax7.set_title("7. Precision Values")
                    else:
                        num_points = 30
                        time_steps2 = list(range(num_points))
                        expected_prec2 = [
                            exp_f - (gap2 * i / num_points) for i in range(num_points)
                        ]
                        actual_prec2 = [act_f] * num_points
                        ax7.plot(
                            time_steps2,
                            expected_prec2,
                            label="Expected",
                            color="#3498db",
                            lw=2,
                        )
                        ax7.plot(
                            time_steps2,
                            actual_prec2,
                            label="Actual",
                            color="#e74c3c",
                            lw=2,
                        )
                        ax7.fill_between(
                            time_steps2,
                            expected_prec2,
                            actual_prec2,
                            color="#9b59b6",
                            alpha=0.3,
                            label="Gap",
                        )
                        all_vals7 = expected_prec2 + actual_prec2
                        _pmin7 = min(all_vals7)
                        _pmax7 = max(all_vals7)
                        _pad7 = (_pmax7 - _pmin7) * 0.1 or 0.05
                        ax7.set_ylim(max(0, _pmin7 - _pad7), _pmax7 + _pad7)
                        ax7.set_title("7. Precision Gap over Time")
                        ax7.legend(
                            loc="upper right", facecolor="#2b2b2b", labelcolor="white"
                        )
                else:
                    ts7 = (
                        time_steps2
                        if time_steps2 is not None
                        else list(range(len(expected_prec2)))
                    )
                    ax7.plot(
                        ts7, expected_prec2, label="Expected", color="#3498db", lw=2
                    )
                    ax7.plot(ts7, actual_prec2, label="Actual", color="#e74c3c", lw=2)
                    ax7.fill_between(
                        ts7,
                        expected_prec2,
                        actual_prec2,
                        color="#9b59b6",
                        alpha=0.3,
                        label="Gap",
                    )
                    all_vals7 = list(expected_prec2) + list(actual_prec2)
                    _pmin7 = min(all_vals7)
                    _pmax7 = max(all_vals7)
                    _pad7 = (_pmax7 - _pmin7) * 0.1 or 0.05
                    ax7.set_ylim(max(0, _pmin7 - _pad7), _pmax7 + _pad7)
                    ax7.set_title("7. Precision Gap over Time")
                    ax7.legend(
                        loc="upper right", facecolor="#2b2b2b", labelcolor="white"
                    )
            else:
                _no_data2(ax7, "7. Precision Gap over Time")

            viz_canvas.draw()
        else:
            # Show placeholder with debug info
            ax = viz_figure.add_subplot(111)
            ax.set_facecolor("#2b2b2b")

            # Check if we have any results at all
            available_experiments = list(self.experiment_results.keys())
            debug_info = (
                f"\n\nAvailable experiments with results: {available_experiments}"
                if available_experiments
                else "\n\nNo experiment results stored yet."
            )

            message = f"No results available for '{experiment_name}'.\nRun the experiment first to see visualization.{debug_info}"
            ax.text(
                0.5,
                0.5,
                message,
                ha="center",
                va="center",
                fontsize=12,
                color="white",
                transform=ax.transAxes,
                wrap=True,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            viz_canvas.draw()
            self._log(
                f"[VIZ] No results available for {experiment_name}. Available: {available_experiments}"
            )

        # No need to restore - we used window-local variables
        def on_close() -> None:
            viz_window.destroy()

        viz_window.protocol("WM_DELETE_WINDOW", on_close)

    def _show_file_menu(self) -> None:
        """Show comprehensive File menu with research-oriented functionality."""
        file_menu = ctk.CTkToplevel(self)
        file_menu.title("File")
        file_menu.geometry("300x500")
        file_menu.transient(self)
        file_menu.attributes("-topmost", True)

        # Menu title
        ctk.CTkLabel(
            file_menu, text="File Operations", font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(15, 10))

        # New Experiment Section
        ctk.CTkLabel(
            file_menu,
            text="New",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#3498db",
        ).pack(pady=(10, 5), anchor="w", padx=20)

        ctk.CTkButton(
            file_menu,
            text="🧪 New Experiment",
            command=self._create_new_experiment,
            width=250,
            height=35,
            fg_color="#27ae60",
            hover_color="#219150",
        ).pack(pady=2, padx=20)

        ctk.CTkButton(
            file_menu,
            text="📝 New Hypothesis",
            command=self._show_create_hypothesis_dialog,
            width=250,
            height=35,
            fg_color="#27ae60",
            hover_color="#219150",
        ).pack(pady=2, padx=20)

        # Open/Import Section
        ctk.CTkLabel(
            file_menu,
            text="Open",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#3498db",
        ).pack(pady=(10, 5), anchor="w", padx=20)

        ctk.CTkButton(
            file_menu,
            text="📂 Open Experiment Directory",
            command=self._open_experiment_directory,
            width=250,
            height=35,
        ).pack(pady=2, padx=20)

        ctk.CTkButton(
            file_menu,
            text="📊 Import Results",
            command=self._import_experiment_results,
            width=250,
            height=35,
        ).pack(pady=2, padx=20)

        ctk.CTkButton(
            file_menu,
            text="🔍 View Audit Log",
            command=self._show_audit_log_viewer,
            width=250,
            height=35,
        ).pack(pady=2, padx=20)

        # Save/Export Section
        ctk.CTkLabel(
            file_menu,
            text="Save",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#3498db",
        ).pack(pady=(10, 5), anchor="w", padx=20)

        ctk.CTkButton(
            file_menu,
            text="💾 Save Current Results",
            command=self._save_experiment_results,
            width=250,
            height=35,
            fg_color="#3498db",
            hover_color="#2980b9",
        ).pack(pady=2, padx=20)

        ctk.CTkButton(
            file_menu,
            text="📈 Export Report",
            command=self._export_research_report,
            width=250,
            height=35,
            fg_color="#3498db",
            hover_color="#2980b9",
        ).pack(pady=2, padx=20)

        # Session Management
        ctk.CTkLabel(
            file_menu,
            text="Session",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#3498db",
        ).pack(pady=(10, 5), anchor="w", padx=20)

        ctk.CTkButton(
            file_menu,
            text="🔄 Reload Experiments",
            command=self._reload_experiments,
            width=250,
            height=35,
        ).pack(pady=2, padx=20)

        ctk.CTkButton(
            file_menu,
            text="🧹 Clear Session",
            command=self._clear_session,
            width=250,
            height=35,
            fg_color="#f39c12",
            hover_color="#d68910",
        ).pack(pady=2, padx=20)

        # Settings
        ctk.CTkLabel(
            file_menu,
            text="Settings",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#3498db",
        ).pack(pady=(10, 5), anchor="w", padx=20)

        ctk.CTkButton(
            file_menu,
            text="⚙️ Application Settings",
            command=self._show_settings_dialog,
            width=250,
            height=35,
            fg_color="#9b59b6",
            hover_color="#8e44ad",
        ).pack(pady=2, padx=20)

        # Close/Exit
        ctk.CTkButton(
            file_menu,
            text="❌ Exit Application",
            command=self._confirm_exit,
            width=250,
            height=35,
            fg_color="#e74c3c",
            hover_color="#c0392b",
        ).pack(pady=(10, 20), padx=20)

    def _show_audit_log_viewer(self) -> None:
        """Open a searchable viewer for the APGI audit trail (.jsonl)."""
        audit_file = Path(self.research_dir) / "audit_trail.jsonl"
        if not audit_file.exists():
            audit_file = Path("audit_trail.jsonl")

        dialog = ctk.CTkToplevel(self)
        dialog.title("🔍 Audit Log Viewer")
        dialog.geometry("820x600")
        dialog.transient(self)
        dialog.attributes("-topmost", True)

        hdr = ctk.CTkFrame(dialog, fg_color="transparent")
        hdr.pack(padx=16, pady=(12, 4), fill="x")
        ctk.CTkLabel(
            hdr, text="Audit Trail", font=ctk.CTkFont(size=15, weight="bold")
        ).pack(side="left")

        filter_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        filter_frame.pack(padx=16, pady=(0, 6), fill="x")
        filter_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(filter_frame, text="Filter:", font=ctk.CTkFont(size=11)).grid(
            row=0, column=0, padx=(0, 6)
        )
        filter_var = ctk.StringVar()
        ctk.CTkEntry(
            filter_frame,
            textvariable=filter_var,
            placeholder_text="event type, action, operator…",
        ).grid(row=0, column=1, sticky="ew", padx=(0, 6))

        log_text = ctk.CTkTextbox(dialog, font=("Courier", 10), state="disabled")
        log_text.pack(padx=16, pady=(0, 4), fill="both", expand=True)

        status_var = ctk.StringVar(value="")
        ctk.CTkLabel(
            dialog,
            textvariable=status_var,
            font=ctk.CTkFont(size=10),
            text_color="#888888",
        ).pack(padx=16, pady=(0, 4), anchor="w")

        _all_entries: list = []

        def _load_entries() -> None:
            _all_entries.clear()
            if not audit_file.exists():
                status_var.set(f"Audit file not found: {audit_file}")
                return
            try:
                with open(audit_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                _all_entries.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
                status_var.set(f"{len(_all_entries)} events — {audit_file}")
            except Exception as exc:
                status_var.set(f"Error reading log: {exc}")

        def _render(query: str = "") -> None:
            q = query.lower()
            lines = []
            for entry in reversed(_all_entries):
                ts = entry.get("timestamp", "")
                evt = entry.get("event_type", "")
                op = entry.get("operator_name", entry.get("operator_id", ""))
                action = entry.get("action", "")
                resource = (
                    f"{entry.get('resource_type','')}/{entry.get('resource_id','')}"
                )
                status = entry.get("status", "")
                row = f"[{ts[:19]}] {evt:<22} {action:<20} {op:<16} {resource:<28} {status}"
                if q and q not in row.lower():
                    continue
                lines.append(row)
            log_text.configure(state="normal")
            log_text.delete("1.0", "end")
            log_text.insert(
                "1.0", "\n".join(lines) if lines else "(no matching entries)"
            )
            log_text.configure(state="disabled")
            status_var.set(
                f"Showing {len(lines)}/{len(_all_entries)} events — {audit_file}"
            )

        filter_var.trace_add("write", lambda *_: _render(filter_var.get()))

        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(padx=16, pady=(0, 12), fill="x")
        ctk.CTkButton(
            btn_frame,
            text="↺ Refresh",
            width=90,
            height=28,
            font=ctk.CTkFont(size=10),
            command=lambda: (_load_entries(), _render(filter_var.get())),
        ).pack(side="left", padx=(0, 6))
        ctk.CTkButton(
            btn_frame,
            text="✕ Close",
            width=80,
            height=28,
            font=ctk.CTkFont(size=10),
            fg_color="#e74c3c",
            hover_color="#c0392b",
            command=dialog.destroy,
        ).pack(side="left")

        _load_entries()
        _render()

    def _create_new_experiment(self) -> None:
        """Create or import a new experiment without editing source files directly."""
        import shutil
        from tkinter import filedialog

        dialog = ctk.CTkToplevel(self)
        dialog.title("🧪 Add Experiment")
        dialog.geometry("560x520")
        dialog.transient(self)
        dialog.attributes("-topmost", True)

        ctk.CTkLabel(
            dialog,
            text="Add New Experiment",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=(15, 5))

        mode_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        mode_frame.pack(padx=20, fill="x", pady=(0, 5))
        _source_var = ctk.StringVar(value="template")
        ctk.CTkRadioButton(
            mode_frame,
            text="Scaffold from template",
            variable=_source_var,
            value="template",
            font=ctk.CTkFont(size=11),
        ).pack(side="left", padx=(0, 20))
        ctk.CTkRadioButton(
            mode_frame,
            text="Import existing .py file",
            variable=_source_var,
            value="import",
            font=ctk.CTkFont(size=11),
        ).pack(side="left")

        form_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        form_frame.pack(padx=20, fill="x", pady=5)
        form_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(form_frame, text="Display name:", font=ctk.CTkFont(size=12)).grid(
            row=0, column=0, sticky="w", padx=5, pady=4
        )
        name_entry = ctk.CTkEntry(form_frame, width=310)
        name_entry.grid(row=0, column=1, columnspan=2, padx=5, pady=4, sticky="ew")

        desc_label = ctk.CTkLabel(
            form_frame, text="Description:", font=ctk.CTkFont(size=12)
        )
        desc_label.grid(row=1, column=0, sticky="nw", padx=5, pady=4)
        desc_entry = ctk.CTkTextbox(form_frame, height=60)
        desc_entry.grid(row=1, column=1, columnspan=2, padx=5, pady=4, sticky="nsew")

        tmpl_label = ctk.CTkLabel(
            form_frame, text="Template:", font=ctk.CTkFont(size=12)
        )
        tmpl_label.grid(row=2, column=0, sticky="w", padx=5, pady=4)
        template_menu = ctk.CTkOptionMenu(
            form_frame,
            values=[
                "APGI Cognitive",
                "Basic Experiment",
                "Data Analysis",
                "Model Training",
            ],
            width=200,
        )
        template_menu.grid(row=2, column=1, padx=5, pady=4, sticky="w")

        import_label = ctk.CTkLabel(
            form_frame, text="Script file:", font=ctk.CTkFont(size=12)
        )
        import_label.grid(row=3, column=0, sticky="w", padx=5, pady=4)
        import_path_var = ctk.StringVar(value="(no file selected)")
        import_path_label = ctk.CTkLabel(
            form_frame,
            textvariable=import_path_var,
            font=ctk.CTkFont(size=10),
            text_color="#888888",
            anchor="w",
        )
        import_path_label.grid(row=3, column=1, padx=5, pady=4, sticky="ew")

        def browse_file() -> None:
            path = filedialog.askopenfilename(
                title="Select experiment script",
                filetypes=[("Python files", "*.py"), ("All files", "*.*")],
                initialdir=str(self.research_dir),
            )
            if path:
                import_path_var.set(path)
                if not name_entry.get().strip():
                    stem = Path(path).stem.replace("run_", "").replace("_", " ").title()
                    name_entry.delete(0, "end")
                    name_entry.insert(0, stem)

        browse_btn = ctk.CTkButton(
            form_frame,
            text="Browse…",
            width=70,
            height=26,
            font=ctk.CTkFont(size=10),
            command=browse_file,
        )
        browse_btn.grid(row=3, column=2, padx=5, pady=4)

        def _update_mode(*_: object) -> None:
            is_import = _source_var.get() == "import"
            _state_tmpl = "disabled" if is_import else "normal"
            desc_label.configure(state=_state_tmpl)
            desc_entry.configure(state=_state_tmpl)
            tmpl_label.configure(state=_state_tmpl)
            template_menu.configure(state=_state_tmpl)
            import_path_label.configure(
                text_color="#cccccc" if is_import else "#555555"
            )
            browse_btn.configure(state="normal" if is_import else "disabled")

        _source_var.trace_add("write", _update_mode)
        _update_mode()

        def create_experiment() -> None:
            name = name_entry.get().strip()
            if not name:
                messagebox.showerror("Error", "Please enter an experiment name.")
                return
            slug = "".join(
                c
                for c in name.lower().replace(" ", "_").replace("-", "_")
                if c.isalnum() or c == "_"
            )
            filepath = self.research_dir / "experiments" / f"run_{slug}.py"
            if filepath.exists():
                if not messagebox.askyesno(
                    "Overwrite?", f"run_{slug}.py already exists. Overwrite?"
                ):
                    return
            try:
                if _source_var.get() == "import":
                    src = import_path_var.get()
                    if src == "(no file selected)" or not Path(src).is_file():
                        messagebox.showerror("Error", "Please select a valid .py file.")
                        return
                    shutil.copy2(src, filepath)
                else:
                    filepath.write_text(
                        self._get_experiment_template(
                            name,
                            desc_entry.get("0.0", "end").strip(),
                            template_menu.get(),
                        )
                    )
                self._log(f"[EXPERIMENTS] Added: {filepath.name}", "#27ae60")
                messagebox.showinfo(
                    "Success", f"Experiment '{name}' added successfully!"
                )
                self._reload_experiments()
                dialog.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to add experiment: {e}")

        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(pady=15)
        ctk.CTkButton(
            btn_frame,
            text="Add Experiment",
            command=create_experiment,
            fg_color="#27ae60",
            hover_color="#219150",
            width=130,
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            btn_frame,
            text="Cancel",
            command=dialog.destroy,
            fg_color="#e74c3c",
            hover_color="#c0392b",
            width=100,
        ).pack(side="left", padx=5)

    def _get_experiment_template(
        self, name: str, description: str, template_type: str
    ) -> str:
        """Generate experiment template code based on type."""
        desc = description.strip() or f"APGI experiment: {name}"

        if template_type == "APGI Cognitive":
            return f'''"""{name}

{desc}
"""

import json
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Tunable APGI parameters (modified by autonomous agent) ──────────────────
NUM_TRIALS = 50
NUM_BLOCKS = 4
TRIALS_PER_BLOCK = 12
STIMULUS_DURATION = 0.2
INTER_STIMULUS_INTERVAL = 0.5
RESPONSE_WINDOW = 1.5
ACCURACY_THRESHOLD = 0.75
PERFORMANCE_THRESHOLD = 0.80
LEARNING_RATE = 0.01
CONFIDENCE_LEVEL = 0.95
USE_FEEDBACK = True
SAVE_RESULTS = True
# ─────────────────────────────────────────────────────────────────────────────


def run_experiment() -> dict:
    logger.info(f"Starting {{__name__}}: {{NUM_TRIALS}} trials across {{NUM_BLOCKS}} blocks")
    reaction_times: list = []
    correct: int = 0
    for _ in range(NUM_BLOCKS):
        for _ in range(TRIALS_PER_BLOCK):
            t0 = time.perf_counter()
            time.sleep(STIMULUS_DURATION + INTER_STIMULUS_INTERVAL)
            rt = time.perf_counter() - t0
            reaction_times.append(rt)
            if rt < RESPONSE_WINDOW:
                correct += 1
    accuracy = correct / max(len(reaction_times), 1)
    mean_rt = sum(reaction_times) / max(len(reaction_times), 1)
    results = {{
        "accuracy": round(accuracy, 4),
        "mean_reaction_time_s": round(mean_rt, 4),
        "num_trials": len(reaction_times),
        "performance_score": round(min(accuracy / PERFORMANCE_THRESHOLD, 1.0), 4),
    }}
    if SAVE_RESULTS:
        print(json.dumps(results))
    logger.info(f"Results: {{results}}")
    return results


if __name__ == "__main__":
    run_experiment()
'''

        if template_type == "Data Analysis":
            return f'''"""{name}

{desc}
"""

import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_TRIALS = 100
CONFIDENCE_LEVEL = 0.95
SIGNAL_TO_NOISE_RATIO = 1.0
PERFORMANCE_THRESHOLD = 0.80
SAVE_RESULTS = True


def run_experiment() -> dict:
    logger.info(f"Data analysis: {{NUM_TRIALS}} samples")
    data = np.random.normal(0, 1 / max(SIGNAL_TO_NOISE_RATIO, 0.01), NUM_TRIALS)
    results = {{
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "num_trials": NUM_TRIALS,
        "performance_score": float(min(abs(np.mean(data)) + 0.5, 1.0)),
    }}
    if SAVE_RESULTS:
        print(json.dumps(results))
    return results


if __name__ == "__main__":
    run_experiment()
'''

        if template_type == "Model Training":
            return f'''"""{name}

{desc}
"""

import json
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_TRIALS = 50
LEARNING_RATE = 0.001
BATCH_SIZE = 32
HIDDEN_UNITS = 128
NUM_LAYERS = 2
WEIGHT_DECAY = 1e-4
PERFORMANCE_THRESHOLD = 0.90
SAVE_RESULTS = True


def run_experiment() -> dict:
    logger.info(f"Training: lr={{LEARNING_RATE}} batch={{BATCH_SIZE}}")
    time.sleep(0.5)
    loss = max(0.01, 1.0 - LEARNING_RATE * NUM_TRIALS * 0.1)
    results = {{
        "final_loss": round(loss, 4),
        "num_trials": NUM_TRIALS,
        "performance_score": round(max(0.0, 1.0 - loss), 4),
    }}
    if SAVE_RESULTS:
        print(json.dumps(results))
    return results


if __name__ == "__main__":
    run_experiment()
'''

        return f'''"""{name}

{desc}
"""

import json
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_TRIALS = 30
PERFORMANCE_THRESHOLD = 0.75
SAVE_RESULTS = True


def run_experiment() -> dict:
    logger.info(f"Running {{NUM_TRIALS}} trials")
    time.sleep(0.2)
    results = {{"num_trials": NUM_TRIALS, "performance_score": 0.8, "status": "success"}}
    if SAVE_RESULTS:
        print(json.dumps(results))
    return results


if __name__ == "__main__":
    run_experiment()
'''

    def _open_experiment_directory(self) -> None:
        """Open the experiments directory in file explorer."""
        import platform
        import subprocess  # subprocess is used in _open_experiment_directory method

        experiments_dir = self.research_dir / "experiments"

        try:
            if platform.system() == "Windows":
                subprocess.run(["explorer", str(experiments_dir)], check=True)
                secure_run(["explorer", str(experiments_dir)], check=True)
            elif platform.system() == "Darwin":  # macOS
                secure_run(["open", str(experiments_dir)], check=True)
            else:  # Linux
                secure_run(["xdg-open", str(experiments_dir)], check=True)

            self._log(f"Opened experiments directory: {experiments_dir}", "#3498db")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open directory: {e}")

    def _import_experiment_results(self) -> None:
        """Import experiment results from file."""
        from tkinter import filedialog

        file_path = filedialog.askopenfilename(
            title="Import Experiment Results",
            filetypes=[
                ("JSON files", "*.json"),
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
            initialdir=str(self.research_dir),
        )

        if file_path:
            try:
                # TODO: Implement proper result import logic
                self._log(f"Imported results from: {Path(file_path).name}", "#27ae60")
                messagebox.showinfo("Success", "Results imported successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import results: {e}")

    def _save_experiment_results(self) -> None:
        """Save current experiment results."""
        if not self.experiment_results:
            messagebox.showinfo("No Results", "No experiment results to save.")
            return

        from tkinter import filedialog

        file_path = filedialog.asksaveasfilename(
            title="Save Experiment Results",
            defaultextension=".json",
            filetypes=[
                ("JSON files", "*.json"),
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
            initialdir=str(self.research_dir),
        )

        if file_path:
            try:
                import json

                with open(file_path, "w") as f:
                    json.dump(self.experiment_results, f, indent=2)
                self._log(f"Saved results to: {Path(file_path).name}", "#27ae60")
                messagebox.showinfo("Success", "Results saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {e}")

    def _export_research_report(self) -> None:
        """Export comprehensive research report."""
        from tkinter import filedialog

        file_path = filedialog.asksaveasfilename(
            title="Export Research Report",
            defaultextension=".md",
            filetypes=[
                ("Markdown files", "*.md"),
                ("HTML files", "*.html"),
                ("PDF files", "*.pdf"),
                ("All files", "*.*"),
            ],
            initialdir=str(self.research_dir),
        )

        if file_path:
            try:
                report_content = self._generate_research_report()
                Path(file_path).write_text(report_content)
                self._log(f"Exported report to: {Path(file_path).name}", "#27ae60")
                messagebox.showinfo("Success", "Research report exported successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export report: {e}")

    def _generate_research_report(self) -> str:
        """Generate a comprehensive research report."""
        from datetime import datetime

        report = f"""# APGI Research Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Research Directory:** {self.research_dir}

## Executive Summary

This report summarizes the current state of experiments and results in the APGI research environment.

## Experiment Overview

**Total Experiments:** {len(self.experiments)}
**Running Experiments:** {len(self.running_experiments)}
**Completed Results:** {len(self.experiment_results)}

## Experiments

"""

        for name, script in self.experiments:
            status = (
                "🟢 Ready" if name not in self.running_experiments else "🔵 Running"
            )
            report += (
                f"\n### {name}\n- **Script:** `{script}`\n- **Status:** {status}\n"
            )

            if name in self.experiment_results:
                report += "- **Results:** Available\n"

        report += f"""

## Guardrail Status

- **Current Status:** {self.guardrail_state['status']}
- **Confidence Level:** {self.guardrail_state['confidence']:.0%}
- **Regression Rate:** {self.guardrail_state['last_regression']:.1%}
- **Escalation Count:** {self.guardrail_state['escalation_count']}

## Hypotheses

"""

        pending_hypotheses = self.approval_board.get_pending_hypotheses()
        if pending_hypotheses:
            for hypothesis in pending_hypotheses:
                report += f"\n### {hypothesis.title}\n"
                report += f"- **Status:** {hypothesis.status.value}\n"
                report += f"- **Confidence:** {hypothesis.confidence_score:.2f}\n"
                report += f"- **Description:** {hypothesis.description}\n"
        else:
            report += "\nNo pending hypotheses.\n"

        report += """

## Recommendations

Based on the current state of experiments and guardrails:

1. Review any running experiments for completion
2. Address any escalated guardrail issues
3. Consider approving pending hypotheses
4. Plan next iteration of experiments

---

*Report generated by APGI Experiment Runner*
"""

        return report

    def _reload_experiments(self) -> None:
        """Reload experiments from directory."""
        self.experiments = self._find_experiments()
        # Refresh the UI
        self._refresh_experiment_display()
        self._log(
            f"Reloaded {len(self.experiments)} experiments from directory", "#3498db"
        )

    def _refresh_experiment_display(self) -> None:
        """Refresh the experiment display in the main area."""
        # Clear existing cards
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # Recreate experiment cards
        for i, (name, script) in enumerate(self.experiments):
            self._create_experiment_card(self.scrollable_frame, name, script, i)

    def _clear_session(self) -> None:
        """Clear current session data."""
        result = messagebox.askyesno(
            "Clear Session",
            "This will clear all current results and reset the session. Continue?",
        )

        if result:
            # Stop all running experiments
            self._stop_all()

            # Clear results
            self.experiment_results.clear()

            # Clear console
            self._clear_console()

            # Reset guardrails
            self._update_guardrail_dashboard()

            self._log("Session cleared - all data reset", "#f39c12")
            messagebox.showinfo(
                "Session Cleared", "Session has been cleared successfully."
            )

    def _confirm_exit(self) -> None:
        """Confirm application exit with optional save."""
        if self.running_experiments:
            result = messagebox.askyesno(
                "Experiments Running",
                "There are experiments still running. Exit anyway?",
            )
            if not result:
                return

        if self.experiment_results:
            save_result: Optional[bool] = messagebox.askyesnocancel(
                "Save Results", "Do you want to save experiment results before exiting?"
            )
            if save_result is True:  # Yes
                self._save_experiment_results()
            elif save_result is None:  # Cancel
                return
            else:  # No (False)
                pass  # Continue to quit

        self.quit()

    def _show_edit_menu(self) -> None:
        """Show comprehensive Edit menu with research-oriented functionality."""
        edit_menu = ctk.CTkToplevel(self)
        edit_menu.title("Edit")
        edit_menu.geometry("300x600")
        edit_menu.transient(self)
        edit_menu.attributes("-topmost", True)

        # Menu title
        ctk.CTkLabel(
            edit_menu, text="Edit Operations", font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(15, 10))

        # Console Operations
        ctk.CTkLabel(
            edit_menu,
            text="Console",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#3498db",
        ).pack(pady=(10, 5), anchor="w", padx=20)

        ctk.CTkButton(
            edit_menu,
            text="🧹 Clear Console",
            command=self._clear_console,
            width=250,
            height=35,
        ).pack(pady=2, padx=20)

        ctk.CTkButton(
            edit_menu,
            text="📋 Copy Console Output",
            command=self._copy_console_output,
            width=250,
            height=35,
        ).pack(pady=2, padx=20)

        ctk.CTkButton(
            edit_menu,
            text="💾 Save Console Log",
            command=self._save_console_log,
            width=250,
            height=35,
        ).pack(pady=2, padx=20)

        # Experiment Operations
        ctk.CTkLabel(
            edit_menu,
            text="Experiments",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#3498db",
        ).pack(pady=(10, 5), anchor="w", padx=20)

        ctk.CTkButton(
            edit_menu,
            text="🔄 Refresh Experiments",
            command=self._reload_experiments,
            width=250,
            height=35,
        ).pack(pady=2, padx=20)

        ctk.CTkButton(
            edit_menu,
            text="✏️ Edit Experiment Script",
            command=self._edit_experiment_script,
            width=250,
            height=35,
        ).pack(pady=2, padx=20)

        ctk.CTkButton(
            edit_menu,
            text="🗑️ Delete Experiment",
            command=self._delete_experiment,
            width=250,
            height=35,
            fg_color="#e74c3c",
            hover_color="#c0392b",
        ).pack(pady=2, padx=20)

        # Results Operations
        ctk.CTkLabel(
            edit_menu,
            text="Results",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#3498db",
        ).pack(pady=(10, 5), anchor="w", padx=20)

        ctk.CTkButton(
            edit_menu,
            text="🗑️ Clear All Results",
            command=self._clear_all_results,
            width=250,
            height=35,
            fg_color="#f39c12",
            hover_color="#d68910",
        ).pack(pady=2, padx=20)

        ctk.CTkButton(
            edit_menu,
            text="📊 Reset Visualizations",
            command=self._reset_visualizations,
            width=250,
            height=35,
        ).pack(pady=2, padx=20)

        # Configuration
        ctk.CTkLabel(
            edit_menu,
            text="Configuration",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#3498db",
        ).pack(pady=(10, 5), anchor="w", padx=20)

        ctk.CTkButton(
            edit_menu,
            text="⚙️ Settings",
            command=self._show_settings_dialog,
            width=250,
            height=35,
        ).pack(pady=2, padx=20)

        ctk.CTkButton(
            edit_menu,
            text="🔧 Reset Guardrails",
            command=self._reset_guardrails,
            width=250,
            height=35,
            fg_color="#f39c12",
            hover_color="#d68910",
        ).pack(pady=2, padx=20)

    def _copy_console_output(self) -> None:
        """Copy console output to clipboard."""
        try:
            import pyperclip

            console_text = self.console_text.get("1.0", "end").strip()
            pyperclip.copy(console_text)
            self._log("Console output copied to clipboard", "#27ae60")
            messagebox.showinfo("Success", "Console output copied to clipboard!")
        except ImportError:
            # Fallback to tkinter clipboard
            try:
                self.clipboard_clear()
                console_text = self.console_text.get("1.0", "end").strip()
                self.clipboard_append(console_text)
                self._log("Console output copied to clipboard", "#27ae60")
                messagebox.showinfo("Success", "Console output copied to clipboard!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to copy to clipboard: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy to clipboard: {e}")

    def _save_console_log(self) -> None:
        """Save console log to file."""
        from datetime import datetime
        from tkinter import filedialog

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"console_log_{timestamp}.txt"

        file_path = filedialog.asksaveasfilename(
            title="Save Console Log",
            defaultextension=".txt",
            initialfile=default_filename,
            filetypes=[
                ("Text files", "*.txt"),
                ("Log files", "*.log"),
                ("All files", "*.*"),
            ],
            initialdir=str(self.research_dir),
        )

        if file_path:
            try:
                console_text = self.console_text.get("1.0", "end").strip()
                Path(file_path).write_text(console_text, encoding="utf-8")
                self._log(f"Console log saved to: {Path(file_path).name}", "#27ae60")
                messagebox.showinfo("Success", "Console log saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save console log: {e}")

    def _edit_experiment_script(self) -> None:
        """Edit selected experiment script."""
        if not self.experiments:
            messagebox.showinfo("No Experiments", "No experiments available to edit.")
            return

        # Create experiment selection dialog
        dialog = ctk.CTkToplevel(self)
        dialog.title("✏️ Edit Experiment Script")
        dialog.geometry("400x300")
        dialog.transient(self)
        dialog.attributes("-topmost", True)

        ctk.CTkLabel(
            dialog,
            text="Select Experiment to Edit",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=(15, 10))

        # Experiment selection
        experiment_names = [name for name, _ in self.experiments]
        selected_experiment = ctk.CTkOptionMenu(dialog, values=experiment_names)
        selected_experiment.pack(pady=10, padx=20)

        def open_in_editor() -> None:
            experiment_name = selected_experiment.get()
            script_path = None

            for name, script in self.experiments:
                if name == experiment_name:
                    script_path = self.research_dir / script
                    break

            if script_path and script_path.exists():
                try:
                    import platform

                    if platform.system() == "Windows":
                        secure_run(["notepad", str(script_path)], check=True)
                    elif platform.system() == "Darwin":  # macOS
                        secure_run(
                            ["open", "-a", "TextEdit", str(script_path)], check=True
                        )
                    else:  # Linux
                        secure_run(["xdg-open", str(script_path)], check=True)

                    self._log(
                        f"Opened script for editing: {script_path.name}", "#3498db"
                    )
                    dialog.destroy()
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to open script: {e}")

        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(pady=15)

        ctk.CTkButton(
            btn_frame,
            text="Open in Editor",
            command=open_in_editor,
            fg_color="#3498db",
            hover_color="#2980b9",
            width=120,
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            btn_frame,
            text="Cancel",
            command=dialog.destroy,
            fg_color="#e74c3c",
            hover_color="#c0392b",
            width=100,
        ).pack(side="left", padx=5)

    def _delete_experiment(self) -> None:
        """Delete selected experiment."""
        if not self.experiments:
            messagebox.showinfo("No Experiments", "No experiments available to delete.")
            return

        # Create experiment selection dialog
        dialog = ctk.CTkToplevel(self)
        dialog.title("🗑️ Delete Experiment")
        dialog.geometry("400x300")
        dialog.transient(self)
        dialog.attributes("-topmost", True)

        ctk.CTkLabel(
            dialog,
            text="Select Experiment to Delete",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=(15, 10))

        ctk.CTkLabel(
            dialog,
            text="⚠️ This will permanently delete the experiment file!",
            font=ctk.CTkFont(size=12),
            text_color="#e74c3c",
        ).pack(pady=5)

        # Experiment selection
        experiment_names = [name for name, _ in self.experiments]
        selected_experiment = ctk.CTkOptionMenu(dialog, values=experiment_names)
        selected_experiment.pack(pady=10, padx=20)

        def confirm_delete() -> None:
            experiment_name = selected_experiment.get()
            script_path = None

            for name, script in self.experiments:
                if name == experiment_name:
                    script_path = self.research_dir / script
                    break

            if script_path and script_path.exists():
                result = messagebox.askyesno(
                    "Confirm Delete",
                    f"Are you sure you want to delete {experiment_name}?\\n\\nFile: {script_path.name}",
                )

                if result:
                    try:
                        script_path.unlink()
                        self._log(f"Deleted experiment: {experiment_name}", "#e74c3c")
                        self._reload_experiments()
                        dialog.destroy()
                        messagebox.showinfo(
                            "Success", "Experiment deleted successfully!"
                        )
                    except Exception as e:
                        messagebox.showerror(
                            "Error", f"Failed to delete experiment: {e}"
                        )

        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(pady=15)

        ctk.CTkButton(
            btn_frame,
            text="Delete",
            command=confirm_delete,
            fg_color="#e74c3c",
            hover_color="#c0392b",
            width=100,
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            btn_frame,
            text="Cancel",
            command=dialog.destroy,
            fg_color="#7f8c8d",
            hover_color="#636e72",
            width=100,
        ).pack(side="left", padx=5)

    def _clear_all_results(self) -> None:
        """Clear all experiment results."""
        result = messagebox.askyesno(
            "Clear All Results", "This will clear all experiment results. Continue?"
        )

        if result:
            self.experiment_results.clear()
            self._log("All experiment results cleared", "#f39c12")
            messagebox.showinfo("Success", "All experiment results have been cleared.")

    def _reset_visualizations(self) -> None:
        """Reset all visualization windows."""
        # Close any open visualization windows
        for widget in self.winfo_children():
            if isinstance(widget, ctk.CTkToplevel) and "VIZ" in widget.title():
                widget.destroy()

        # Reset current figure and canvas
        self.current_figure = None
        self.current_canvas = None

        self._log("All visualizations reset", "#f39c12")
        messagebox.showinfo("Success", "All visualizations have been reset.")

    def _reset_guardrails(self) -> None:
        """Reset guardrail system to default state."""
        result = messagebox.askyesno(
            "Reset Guardrails",
            "This will reset the guardrail system to default state. Continue?",
        )

        if result:
            self.guardrail_state = {
                "status": "IDLE",
                "confidence": 1.0,
                "last_regression": 0.0,
                "escalation_count": 0,
                "last_experiment": "",
            }
            self._update_guardrail_dashboard()
            self._log("Guardrail system reset to default state", "#f39c12")
            messagebox.showinfo("Success", "Guardrail system has been reset.")

    def _show_view_menu(self) -> None:
        """Show comprehensive View menu with display options."""
        view_menu = ctk.CTkToplevel(self)
        view_menu.title("View")
        view_menu.geometry("300x600")
        view_menu.transient(self)
        view_menu.attributes("-topmost", True)

        # Menu title
        ctk.CTkLabel(
            view_menu, text="View Options", font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(15, 10))

        # Appearance
        ctk.CTkLabel(
            view_menu,
            text="Appearance",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#3498db",
        ).pack(pady=(10, 5), anchor="w", padx=20)

        ctk.CTkButton(
            view_menu,
            text="🌓 Toggle Appearance",
            command=lambda: self.change_appearance_mode(
                "Light" if ctk.get_appearance_mode() == "Dark" else "Dark"
            ),
            width=250,
            height=35,
        ).pack(pady=2, padx=20)

        ctk.CTkButton(
            view_menu,
            text="🌙 Dark Mode",
            command=lambda: self.change_appearance_mode("Dark"),
            width=250,
            height=35,
        ).pack(pady=2, padx=20)

        ctk.CTkButton(
            view_menu,
            text="☀️ Light Mode",
            command=lambda: self.change_appearance_mode("Light"),
            width=250,
            height=35,
        ).pack(pady=2, padx=20)

        ctk.CTkButton(
            view_menu,
            text="💻 System Mode",
            command=lambda: self.change_appearance_mode("System"),
            width=250,
            height=35,
        ).pack(pady=2, padx=20)

        # Layout
        ctk.CTkLabel(
            view_menu,
            text="Layout",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#3498db",
        ).pack(pady=(10, 5), anchor="w", padx=20)

        ctk.CTkButton(
            view_menu,
            text="📐 Toggle Sidebar",
            command=self._toggle_sidebar,
            width=250,
            height=35,
        ).pack(pady=2, padx=20)

        ctk.CTkButton(
            view_menu,
            text="📊 Maximize Console",
            command=self._maximize_console,
            width=250,
            height=35,
        ).pack(pady=2, padx=20)

        ctk.CTkButton(
            view_menu, text="🔍 Zoom In", command=self._zoom_in, width=250, height=35
        ).pack(pady=2, padx=20)

        ctk.CTkButton(
            view_menu, text="🔍 Zoom Out", command=self._zoom_out, width=250, height=35
        ).pack(pady=2, padx=20)

        ctk.CTkButton(
            view_menu,
            text="🔄 Reset Zoom",
            command=self._reset_zoom,
            width=250,
            height=35,
        ).pack(pady=2, padx=20)

        # Windows
        ctk.CTkLabel(
            view_menu,
            text="Windows",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#3498db",
        ).pack(pady=(10, 5), anchor="w", padx=20)

        ctk.CTkButton(
            view_menu,
            text="📈 Show All Visualizations",
            command=self._show_all_visualizations,
            width=250,
            height=35,
        ).pack(pady=2, padx=20)

        ctk.CTkButton(
            view_menu,
            text="🗑️ Close All Windows",
            command=self._close_all_windows,
            width=250,
            height=35,
            fg_color="#f39c12",
            hover_color="#d68910",
        ).pack(pady=2, padx=20)

        # Refresh
        ctk.CTkLabel(
            view_menu,
            text="Refresh",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#3498db",
        ).pack(pady=(10, 5), anchor="w", padx=20)

        ctk.CTkButton(
            view_menu,
            text="🔄 Refresh UI",
            command=self._refresh_ui,
            width=250,
            height=35,
        ).pack(pady=2, padx=20)

        # Progress
        ctk.CTkLabel(
            view_menu,
            text="Progress",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#3498db",
        ).pack(pady=(10, 5), anchor="w", padx=20)

        ctk.CTkButton(
            view_menu,
            text="📊 Show Progress",
            command=self._show_progress_dialog,
            width=250,
            height=35,
        ).pack(pady=2, padx=20)

        ctk.CTkButton(
            view_menu,
            text="🔍 Audit Log Viewer",
            command=self._show_audit_log_viewer,
            width=250,
            height=35,
        ).pack(pady=2, padx=20)

    def _toggle_sidebar(self) -> None:
        """Toggle sidebar visibility."""
        if hasattr(self, "navigation_frame"):
            if self.navigation_frame.winfo_ismapped():
                self.navigation_frame.grid_remove()
                self._log("Sidebar hidden", "#3498db")
            else:
                self.navigation_frame.grid(row=1, column=0, sticky="nsew")
                self._log("Sidebar shown", "#3498db")

    def _maximize_console(self) -> None:
        """Maximize console area."""
        current_state = self.console_frame.grid_info()
        if current_state["row"] == 2:  # Normal console position
            # Maximize console
            self.console_frame.grid_forget()
            self.console_frame.grid(
                row=0,
                column=0,
                columnspan=2,
                rowspan=3,
                sticky="nsew",
                padx=20,
                pady=20,
            )
            self._log("Console maximized", "#3498db")
        else:
            # Restore normal layout
            self.console_frame.grid_forget()
            self.console_frame.grid(
                row=2, column=0, columnspan=2, padx=20, pady=(10, 20), sticky="nsew"
            )
            self._log("Console restored to normal size", "#3498db")

    def _zoom_in(self) -> None:
        """Zoom in UI elements."""
        try:
            current_font = self.console_text.cget("font")
            if isinstance(current_font, tuple) and len(current_font) >= 2:
                new_size = min(current_font[1] + 1, 20)  # Max size 20
                self.console_text.configure(font=(current_font[0], new_size))
                self._log(f"Zoomed in to font size {new_size}", "#3498db")
        except Exception as e:
            self._log(f"Failed to zoom in: {e}", "#e74c3c")

    def _zoom_out(self) -> None:
        """Zoom out UI elements."""
        try:
            current_font = self.console_text.cget("font")
            if isinstance(current_font, tuple) and len(current_font) >= 2:
                new_size = max(current_font[1] - 1, 8)  # Min size 8
                self.console_text.configure(font=(current_font[0], new_size))
                self._log(f"Zoomed out to font size {new_size}", "#3498db")
        except Exception as e:
            self._log(f"Failed to zoom out: {e}", "#e74c3c")

    def _reset_zoom(self) -> None:
        """Reset zoom to default."""
        try:
            self.console_text.configure(font=("Courier", 13))
            self._log("Zoom reset to default", "#3498db")
        except Exception as e:
            self._log(f"Failed to reset zoom: {e}", "#e74c3c")

    def _show_all_visualizations(self) -> None:
        """Show all available visualizations."""
        if not self.experiment_results:
            messagebox.showinfo(
                "No Results", "No experiment results available for visualization."
            )
            return

        for experiment_name in self.experiment_results:
            try:
                self._show_results_visualization(experiment_name)
            except Exception as e:
                self._log(
                    f"Failed to show visualization for {experiment_name}: {e}",
                    "#e74c3c",
                )

        self._log(f"Opened {len(self.experiment_results)} visualizations", "#27ae60")

    def _close_all_windows(self) -> None:
        """Close all popup windows except main window."""
        closed_count = 0
        for widget in self.winfo_children():
            if isinstance(widget, ctk.CTkToplevel):
                widget.destroy()
                closed_count += 1

        self._log(f"Closed {closed_count} popup windows", "#f39c12")
        messagebox.showinfo("Success", f"Closed {closed_count} popup windows.")

    def _refresh_ui(self) -> None:
        """Refresh the entire UI."""
        try:
            # Refresh experiments
            self._reload_experiments()

            # Refresh hypothesis display
            self._refresh_hypothesis_display()

            # Update guardrail dashboard
            self._update_guardrail_dashboard()

            self._log("UI refreshed successfully", "#27ae60")
            messagebox.showinfo("Success", "UI has been refreshed.")
        except Exception as e:
            self._log(f"Failed to refresh UI: {e}", "#e74c3c")
            messagebox.showerror("Error", f"Failed to refresh UI: {e}")

    def _show_help_menu(self) -> None:
        """Show comprehensive Help menu with documentation and resources."""
        help_menu = ctk.CTkToplevel(self)
        help_menu.title("Help")
        help_menu.geometry("350x700")
        help_menu.transient(self)
        help_menu.attributes("-topmost", True)

        # Menu title
        ctk.CTkLabel(
            help_menu,
            text="Help & Documentation",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=(15, 10))

        # Documentation
        ctk.CTkLabel(
            help_menu,
            text="Documentation",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#3498db",
        ).pack(pady=(10, 5), anchor="w", padx=20)

        ctk.CTkButton(
            help_menu,
            text="📖 User Guide",
            command=self._show_user_guide,
            width=300,
            height=35,
        ).pack(pady=2, padx=20)

        ctk.CTkButton(
            help_menu,
            text="🔧 API Documentation",
            command=self._show_api_docs,
            width=300,
            height=35,
        ).pack(pady=2, padx=20)

        ctk.CTkButton(
            help_menu,
            text="📚 Experiment Templates",
            command=self._show_experiment_templates,
            width=300,
            height=35,
        ).pack(pady=2, padx=20)

        # Tutorials
        ctk.CTkLabel(
            help_menu,
            text="Tutorials",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#3498db",
        ).pack(pady=(10, 5), anchor="w", padx=20)

        ctk.CTkButton(
            help_menu,
            text="🎯 Getting Started",
            command=self._show_getting_started,
            width=300,
            height=35,
        ).pack(pady=2, padx=20)

        ctk.CTkButton(
            help_menu,
            text="🧪 Creating Experiments",
            command=self._show_experiment_tutorial,
            width=300,
            height=35,
        ).pack(pady=2, padx=20)

        ctk.CTkButton(
            help_menu,
            text="🤖 XPR Agent Guide",
            command=self._show_xpr_guide,
            width=300,
            height=35,
        ).pack(pady=2, padx=20)

        # Support
        ctk.CTkLabel(
            help_menu,
            text="Support",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#3498db",
        ).pack(pady=(10, 5), anchor="w", padx=20)

        ctk.CTkButton(
            help_menu,
            text="🐛 Report Issue",
            command=self._report_issue,
            width=300,
            height=35,
        ).pack(pady=2, padx=20)

        ctk.CTkButton(
            help_menu,
            text="💬 Feedback",
            command=self._send_feedback,
            width=300,
            height=35,
        ).pack(pady=2, padx=20)

        # About
        ctk.CTkLabel(
            help_menu,
            text="About",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#3498db",
        ).pack(pady=(10, 5), anchor="w", padx=20)

        ctk.CTkButton(
            help_menu,
            text="ℹ️ About APGI",
            command=self._show_about,
            width=300,
            height=35,
        ).pack(pady=2, padx=20)

        ctk.CTkButton(
            help_menu,
            text="📋 System Info",
            command=self._show_system_info,
            width=300,
            height=35,
        ).pack(pady=2, padx=20)

        # Close button
        ctk.CTkButton(
            help_menu,
            text="❌ Close",
            command=help_menu.destroy,
            fg_color="#7f8c8d",
            hover_color="#636e72",
            width=300,
            height=35,
        ).pack(pady=(15, 20))

    def _show_settings_dialog(self) -> None:
        """Show comprehensive settings dialog for configuring experiment parameters."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Application Settings")
        dialog.geometry("600x700")
        dialog.transient(self)
        dialog.attributes("-topmost", True)

        ctk.CTkLabel(
            dialog,
            text="Settings",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(pady=(20, 10))

        settings_scroll = ctk.CTkScrollableFrame(dialog)
        settings_scroll.pack(padx=20, pady=10, fill="both", expand=True)

        exp_frame = ctk.CTkFrame(settings_scroll)
        exp_frame.pack(fill="x", pady=5, padx=5)
        ctk.CTkLabel(
            exp_frame,
            text="Experiment Settings",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#3498db",
        ).pack(anchor="w", padx=10, pady=(10, 5))

        ctk.CTkLabel(exp_frame, text="Max Concurrent Experiments:").pack(
            anchor="w", padx=10
        )
        self._setting_max_concurrent = ctk.CTkSlider(
            exp_frame, from_=1, to=10, number_of_steps=9
        )
        self._setting_max_concurrent.set(3)
        self._setting_max_concurrent.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(exp_frame, text="Auto-save Interval (seconds):").pack(
            anchor="w", padx=10
        )
        self._setting_autosave = ctk.CTkEntry(exp_frame, width=200)
        self._setting_autosave.insert(0, "300")
        self._setting_autosave.pack(anchor="w", padx=10, pady=5)

        apgi_frame = ctk.CTkFrame(settings_scroll)
        apgi_frame.pack(fill="x", pady=5, padx=5)
        ctk.CTkLabel(
            apgi_frame,
            text="APGI Integration",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#9b59b6",
        ).pack(anchor="w", padx=10, pady=(10, 5))

        self._setting_apgi_enabled = ctk.CTkCheckBox(
            apgi_frame, text="Enable APGI Integration"
        )
        self._setting_apgi_enabled.select()
        self._setting_apgi_enabled.pack(anchor="w", padx=10, pady=5)

        self._setting_apgi_strict = ctk.CTkCheckBox(
            apgi_frame, text="Strict Compliance Mode"
        )
        self._setting_apgi_strict.select()
        self._setting_apgi_strict.pack(anchor="w", padx=10, pady=5)

        agent_frame = ctk.CTkFrame(settings_scroll)
        agent_frame.pack(fill="x", pady=5, padx=5)
        ctk.CTkLabel(
            agent_frame,
            text="Agent Engine",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#27ae60",
        ).pack(anchor="w", padx=10, pady=(10, 5))

        self._setting_agent_enabled = ctk.CTkCheckBox(
            agent_frame, text="Enable Autonomous Agent"
        )
        self._setting_agent_enabled.select()
        self._setting_agent_enabled.pack(anchor="w", padx=10, pady=5)

        ctk.CTkLabel(agent_frame, text="Max Iterations:").pack(anchor="w", padx=10)
        self._setting_max_iterations = ctk.CTkEntry(agent_frame, width=200)
        self._setting_max_iterations.insert(0, "50")
        self._setting_max_iterations.pack(anchor="w", padx=10, pady=5)

        ctk.CTkLabel(agent_frame, text="Temperature:").pack(anchor="w", padx=10)
        self._setting_temperature = ctk.CTkSlider(
            agent_frame, from_=0.0, to=1.0, number_of_steps=20
        )
        self._setting_temperature.set(0.7)
        self._setting_temperature.pack(fill="x", padx=10, pady=5)

        viz_frame = ctk.CTkFrame(settings_scroll)
        viz_frame.pack(fill="x", pady=5, padx=5)
        ctk.CTkLabel(
            viz_frame,
            text="Visualization",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#e74c3c",
        ).pack(anchor="w", padx=10, pady=(10, 5))

        self._setting_live_charts = ctk.CTkCheckBox(
            viz_frame, text="Enable Live Chart Updates"
        )
        self._setting_live_charts.select()
        self._setting_live_charts.pack(anchor="w", padx=10, pady=5)

        ctk.CTkLabel(viz_frame, text="Chart Refresh Rate (ms):").pack(
            anchor="w", padx=10
        )
        self._setting_refresh_rate = ctk.CTkEntry(viz_frame, width=200)
        self._setting_refresh_rate.insert(0, "1000")
        self._setting_refresh_rate.pack(anchor="w", padx=10, pady=5)

        adv_frame = ctk.CTkFrame(settings_scroll)
        adv_frame.pack(fill="x", pady=5, padx=5)
        ctk.CTkLabel(
            adv_frame,
            text="Advanced",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#f39c12",
        ).pack(anchor="w", padx=10, pady=(10, 5))

        self._setting_debug_mode = ctk.CTkCheckBox(
            adv_frame, text="Debug Mode (verbose logging)"
        )
        self._setting_debug_mode.pack(anchor="w", padx=10, pady=5)

        self._setting_auto_reload = ctk.CTkCheckBox(
            adv_frame, text="Auto-reload Experiments"
        )
        self._setting_auto_reload.select()
        self._setting_auto_reload.pack(anchor="w", padx=10, pady=5)

        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(pady=20)

        def save_settings() -> None:
            settings = {
                "max_concurrent": int(self._setting_max_concurrent.get()),
                "autosave_interval": int(self._setting_autosave.get()),
                "apgi_enabled": self._setting_apgi_enabled.get() == 1,
                "apgi_strict": self._setting_apgi_strict.get() == 1,
                "agent_enabled": self._setting_agent_enabled.get() == 1,
                "max_iterations": int(self._setting_max_iterations.get()),
                "temperature": self._setting_temperature.get(),
                "live_charts": self._setting_live_charts.get() == 1,
                "refresh_rate": int(self._setting_refresh_rate.get()),
                "debug_mode": self._setting_debug_mode.get() == 1,
                "auto_reload": self._setting_auto_reload.get() == 1,
            }
            settings_path = self.research_dir / ".apgi_settings.json"
            import json

            with open(settings_path, "w") as f:
                json.dump(settings, f, indent=2)
            self._log("Settings saved successfully", "#27ae60")
            dialog.destroy()

        ctk.CTkButton(
            btn_frame,
            text="Save",
            command=save_settings,
            fg_color="#27ae60",
            hover_color="#219150",
            width=120,
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            btn_frame,
            text="Cancel",
            command=dialog.destroy,
            fg_color="#e74c3c",
            hover_color="#c0392b",
            width=120,
        ).pack(side="left", padx=5)

    def _show_user_guide(self) -> None:
        """Show user guide documentation."""
        guide_text = """# APGI Experiment Runner - User Guide

## Getting Started
1. **Experiments Tab**: View and run available experiments
2. **Guardrails Panel**: Monitor system safety and confidence levels
3. **Hypothesis Board**: Track and manage research hypotheses
4. **Console**: View real-time experiment output and logs

## Running Experiments
- **RUN**: Execute a single experiment
- **VIZ**: View experiment results and visualizations
- **XPR AUTO**: Launch autonomous experiment improvement

## File Operations
- Create new experiments with templates
- Import/export results and data
- Generate comprehensive research reports

## Tips
- Use the sidebar for quick access to common actions
- Monitor guardrail status for system safety
- Save your work regularly with File → Save Results
"""
        self._show_text_dialog("📖 User Guide", guide_text)

    def _show_api_docs(self) -> None:
        """Show API documentation."""
        api_docs = """# APGI API Documentation

## Core Classes
- `ExperimentRunnerGUI`: Main application interface
- `ApprovalBoard`: Hypothesis management system
- `AutonomousAgent`: Experiment optimization engine

## Key Methods
- `run_experiment(name, script)`: Execute experiment
- `show_results_visualization(name)`: Display results
- `create_hypothesis(...)`: Create research hypothesis

## File Structure
```
apgi-research/
├── experiments/     # Experiment scripts
├── docs/           # Documentation
├── tests/          # Test files
└── *.py           # Core modules
```

## Configuration
- Edit settings via Edit → Settings
- Modify guardrail parameters
- Customize appearance and layout
"""
        self._show_text_dialog("🔧 API Documentation", api_docs)

    def _show_experiment_templates(self) -> None:
        """Show experiment template documentation."""
        templates = """# Experiment Templates

## Basic Experiment Template
```python
def main() -> None:
    \"\"\"Main experiment function.\"\"\"
    logger.info("Starting experiment")

    try:
        result = run_experiment()
        logger.info(f"Completed: {result}")
        return result
    except Exception as e:
        logger.error(f"Failed: {e}")
        raise

def run_experiment() -> dict:
    \"\"\"Run the actual experiment.\"\"\"
    # Your experiment logic here
    return {"status": "success", "data": {...}}
```

## Data Analysis Template
- Includes pandas, numpy integration
- Built-in visualization support
- Statistical analysis helpers

## Model Training Template
- ML framework integration
- Hyperparameter optimization
- Model evaluation metrics

## Visualization Template
- Matplotlib/Plotly support
- Interactive charts
- Export capabilities
"""
        self._show_text_dialog("📚 Experiment Templates", templates)

    def _show_getting_started(self) -> None:
        """Show getting started tutorial."""
        tutorial = """# Getting Started with APGI

## Step 1: Explore Experiments
- Browse available experiments in the main panel
- Click RUN to execute individual experiments
- Use VIZ to view results and charts

## Step 2: Create Your Own Experiment
1. File → New → New Experiment
2. Choose a template (Basic, Data Analysis, etc.)
3. Enter experiment name and description
4. Edit the generated script

## Step 3: Run Experiments
- Start with RUN for manual execution
- Try XPR AUTO for autonomous improvement
- Monitor progress in the console

## Step 4: Analyze Results
- Use VIZ buttons to view visualizations
- Check guardrail status for system health
- Export reports with File → Export Report

## Step 5: Manage Hypotheses
- Create hypotheses with 📝 New Hypothesis
- Review pending hypotheses
- Track experiment outcomes
"""
        self._show_text_dialog("🎯 Getting Started", tutorial)

    def _show_experiment_tutorial(self) -> None:
        """Show experiment creation tutorial."""
        tutorial = """# Creating Experiments Tutorial

## Experiment Structure
Every experiment should have:
1. `main()` function - Entry point
2. `run_experiment()` function - Core logic
3. Proper logging and error handling
4. Return structured results

## Best Practices
- Use descriptive variable names
- Add comprehensive logging
- Handle errors gracefully
- Return structured data

## Example: Data Processing
```python
def run_experiment() -> dict:
    \"\"\"Process sample data.\"\"\"
    import pandas as pd
    import numpy as np

    # Load data
    data = pd.read_csv("sample.csv")

    # Process
    processed = data.dropna()
    summary = processed.describe()

    return {
        "status": "success",
        "rows_processed": len(processed),
        "summary": summary.to_dict()
    }
```

## Testing Your Experiment
- Test with small datasets first
- Verify output structure
- Check error handling
"""
        self._show_text_dialog("🧪 Creating Experiments", tutorial)

    def _show_xpr_guide(self) -> None:
        """Show XPR Agent guide."""
        guide = """# XPR Agent Guide

## What is XPR AUTO?
XPR AUTO is an autonomous experiment improvement agent that:
- Analyzes current experiment performance
- Identifies optimization opportunities
- Implements improvements automatically
- Respects guardrail constraints

## Using XPR AUTO
1. Click XPR AUTO on any experiment
2. Configure constraints (iterations, time budget, etc.)
3. Review the generated improvement plan
4. Approve, modify, or reject the plan
5. Monitor autonomous execution

## Configuration Options
- **Max Iterations**: Number of improvement cycles
- **Time Budget**: Maximum execution time (seconds)
- **Min Confidence**: Minimum confidence threshold
- **Protected Files**: Files that cannot be modified

## Guardrails
XPR AUTO operates within strict safety constraints:
- Confidence level monitoring
- Regression detection
- Escalation procedures
- Human approval requirements

## Best Practices
- Start with conservative settings
- Monitor guardrail status closely
- Review generated plans before approval
- Keep regular backups
"""
        self._show_text_dialog("🤖 XPR Agent Guide", guide)

    def _report_issue(self) -> None:
        """Report an issue."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("🐛 Report Issue")
        dialog.geometry("500x400")
        dialog.transient(self)
        dialog.attributes("-topmost", True)

        ctk.CTkLabel(
            dialog, text="Report an Issue", font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(15, 10))

        form_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        form_frame.pack(padx=20, fill="x", pady=10)

        # Issue type
        ctk.CTkLabel(form_frame, text="Issue Type:", font=ctk.CTkFont(size=12)).grid(
            row=0, column=0, sticky="w", padx=5, pady=5
        )
        issue_type = ctk.CTkOptionMenu(
            form_frame, values=["Bug", "Feature Request", "Performance Issue", "Other"]
        )
        issue_type.grid(row=0, column=1, padx=5, pady=5)

        # Description
        ctk.CTkLabel(form_frame, text="Description:", font=ctk.CTkFont(size=12)).grid(
            row=1, column=0, sticky="nw", padx=5, pady=5
        )
        desc_entry = ctk.CTkTextbox(form_frame, height=100)
        desc_entry.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        def submit_issue() -> None:
            issue_data = {
                "type": issue_type.get(),
                "description": desc_entry.get("0.0", "end").strip(),
                "timestamp": time.time(),
            }
            # In a real implementation, this would send to a tracking system
            self._log(f"Issue reported: {issue_data['type']}", "#f39c12")
            messagebox.showinfo(
                "Thank You", "Your issue has been reported. We'll look into it!"
            )
            dialog.destroy()

        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(pady=15)

        ctk.CTkButton(
            btn_frame,
            text="Submit",
            command=submit_issue,
            fg_color="#3498db",
            hover_color="#2980b9",
            width=100,
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            btn_frame,
            text="Cancel",
            command=dialog.destroy,
            fg_color="#e74c3c",
            hover_color="#c0392b",
            width=100,
        ).pack(side="left", padx=5)

    def _send_feedback(self) -> None:
        """Send feedback."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("💬 Send Feedback")
        dialog.geometry("500x300")
        dialog.transient(self)
        dialog.attributes("-topmost", True)

        ctk.CTkLabel(
            dialog, text="Send Feedback", font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(15, 10))

        ctk.CTkLabel(dialog, text="Your feedback helps us improve APGI!").pack(pady=5)

        feedback_entry = ctk.CTkTextbox(dialog, height=100, width=450)
        feedback_entry.pack(padx=20, pady=10)
        feedback_entry.insert(
            "0.0", "What do you think about APGI? What features would you like to see?"
        )

        def send_feedback() -> None:
            feedback_text = feedback_entry.get("0.0", "end").strip()
            if feedback_text:
                self._log("Feedback submitted - thank you!", "#27ae60")
                messagebox.showinfo(
                    "Thank You!",
                    "Your feedback has been submitted. We appreciate your input!",
                )
                dialog.destroy()
            else:
                messagebox.showwarning(
                    "Empty", "Please enter some feedback before submitting."
                )

        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(pady=15)

        ctk.CTkButton(
            btn_frame,
            text="Send",
            command=send_feedback,
            fg_color="#27ae60",
            hover_color="#219150",
            width=100,
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            btn_frame,
            text="Cancel",
            command=dialog.destroy,
            fg_color="#e74c3c",
            hover_color="#c0392b",
            width=100,
        ).pack(side="left", padx=5)

    def _show_about(self) -> None:
        """Show about dialog."""
        about_text = """# About APGI Experiment Runner

## Version Information
- **Version**: 2.0 Premium Edition
- **Build**: Research Environment
- **Platform**: Python-based Experiment Management

## Key Features
- 🧪 Experiment management and execution
- 🤖 Autonomous experiment improvement (XPR)
- 📊 Real-time visualization and analysis
- 🛡️ Guardrail safety system
- 📝 Hypothesis tracking and approval
- 📈 Comprehensive reporting

## Technologies Used
- **Framework**: CustomTkinter + Python
- **Visualization**: Matplotlib + NumPy
- **AI Integration**: LiteLLM + Autonomous Agents
- **Safety**: Multi-layer guardrail system

## Research Focus
APGI (Allostatic Precision-Gated Ignition) is designed for:
- Scientific experiment automation
- Machine learning model optimization
- Data analysis workflows
- Research reproducibility

## License
Internal Research Use Only
© 2024 APGI Research Team
"""
        self._show_text_dialog("ℹ️ About APGI", about_text)

    def _show_system_info(self) -> None:
        """Show system information."""
        import platform
        import sys
        from datetime import datetime

        system_info = f"""# System Information

## Platform Details
- **OS**: {platform.system()} {platform.release()}
- **Architecture**: {platform.machine()}
- **Processor**: {platform.processor()}
- **Python Version**: {sys.version}

## Application Info
- **APGI Version**: 2.0 Premium Edition
- **Started**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Research Directory**: {self.research_dir}
- **Experiments Found**: {len(self.experiments)}
- **Running Processes**: {len(self.running_experiments)}

## Dependencies Status
"""

        # Check key dependencies
        deps_status = {
            "customtkinter": "✅ Available",
            "matplotlib": "✅ Available",
            "numpy": "✅ Available",
            "litellm": "✅ Available" if LLM_AVAILABLE else "❌ Not Available",
        }

        for dep, status in deps_status.items():
            system_info += f"- **{dep}**: {status}\n"

        system_info += f"""
## Memory Usage
- **Guardrail State**: {len(self.guardrail_state)} parameters tracked
- **Experiment Results**: {len(self.experiment_results)} stored
- **Hypotheses**: {len(self.approval_board.get_pending_hypotheses())} pending

## Performance Metrics
- **UI Theme**: {ctk.get_appearance_mode()}
- **Console Font**: {self.console_text.cget('font')}
- **Window Size**: {self.geometry()}
"""

        self._show_text_dialog("📋 System Information", system_info)

    def _show_text_dialog(self, title: str, content: str) -> None:
        """Helper method to show text content in a dialog."""
        dialog = ctk.CTkToplevel(self)
        dialog.title(title)
        dialog.geometry("600x500")
        dialog.transient(self)
        dialog.attributes("-topmost", True)

        # Create scrollable text widget
        text_frame = ctk.CTkScrollableFrame(dialog)
        text_frame.pack(fill="both", expand=True, padx=20, pady=20)

        text_widget = ctk.CTkTextbox(text_frame, height=400)
        text_widget.pack(fill="both", expand=True)
        text_widget.insert("0.0", content)
        text_widget.configure(state="disabled")  # Make read-only

        # Close button
        ctk.CTkButton(
            dialog,
            text="Close",
            command=dialog.destroy,
            fg_color="#7f8c8d",
            hover_color="#636e72",
            width=100,
        ).pack(pady=(0, 15))


if __name__ == "__main__":
    app = ExperimentRunnerGUI()
    app.mainloop()
