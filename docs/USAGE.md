# Auto-Improvement System for Iowa Gambling Task Experiments

## Overview

The auto-improvement system adapts the autonomous research framework from the original autoresearch project (designed for LLM training) to psychological experimentation. Instead of optimizing neural network training metrics, it optimizes decision-making task sensitivity.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for the configured time budget (5-10 minutes depending on experiment type), checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the Markdown files (e.g., `iowa.md`) that provide context to the AI agents and set up your autonomous research org. The default `iowa.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069).

## How it works

The repo is deliberately kept small and only really has these key files:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits for LLM experiments. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`run_igt.py`** — the file for Iowa Gambling Task experiments (agent-modified).
- **`iowa.md`** — baseline instructions for IGT experiments. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, LLM training runs for a **fixed 5-minute time budget**, while IGT experiments run for **10 minutes** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) for LLM training — lower is better, and vocab-size-independent so architectural changes are fairly compared.

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

## Quick start

**Requirements:** A single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```markdown
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```text
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## Platform support

This code currently requires that you have a single NVIDIA GPU. In principle it is quite possible to support CPU, MPS and other platforms but this would also bloat the code. I'm not 100% sure that I want to take this on personally right now. People can reference (or have their agents reference) the full/parent nanochat repository that has wider platform support and shows the various solutions (e.g. a Flash Attention 3 kernels fallback implementation, generic device support, autodetection, etc.), feel free to create forks or discussions for other platforms and I'm happy to link to them here in the README in some new notable forks section or etc.

Seeing as there seems to be a lot of interest in tinkering with autoresearch on much smaller compute platforms than an H100, a few extra words. If you're going to try running autoresearch on smaller computers (Macbooks etc.), I'd recommend one of the forks below. On top of this, here are some recommendations for how to tune the defaults for much smaller models for aspiring forks:

1. To get half-decent results I'd use a dataset with a lot less entropy, e.g. this [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean). These are GPT-4 generated short stories. Because the data is a lot narrower in scope, you will see reasonable results with a lot smaller models (if you try to sample from them after training).
2. You might experiment with decreasing `vocab_size`, e.g. from 8192 down to 4096, 2048, 1024, or even - simply byte-level tokenizer with 256 possibly bytes after utf-8 encoding.
3. In `prepare.py`, you'll want to lower `MAX_SEQ_LEN` a lot, depending on the computer even down to 256 etc. As you lower `MAX_SEQ_LEN`, you may want to experiment with increasing `DEVICE_BATCH_SIZE` in `train.py` slightly to compensate. The number of tokens per fwd/bwd pass is the product of these two.
4. Also in `prepare.py`, you'll want to decrease `EVAL_TOKENS` so that your validation loss is evaluated on a lot less data.
5. In `train.py`, the primary single knob that controls model complexity is the `DEPTH` (default 8, here). A lot of variables are just functions of this, so e.g. lower it down to e.g. 4.
6. You'll want to most likely use `WINDOW_PATTERN` of just "L", because "SSSL" uses alternating banded attention pattern that may be very inefficient for you. Try it.
7. You'll want to lower `TOTAL_BATCH_SIZE` a lot, but keep it powers of 2, e.g. down to `2**14` (~16K) or so even, hard to tell.

It would be the reasonable hyperparameters to play with. Ask your favorite coding agent for help and copy paste them this guide, as well as the full source code.

## Core Architecture Comparison

### Original Autoresearch System

- **`prepare.py`** - Fixed data prep and utilities (read-only)
- **`train.py`** - Model and training code (agent-modified)
- **`EXPERIMENT.md`** - Agent instructions (human-modified)
- **Goal**: Minimize validation bits per byte (val_bpb)
- **Time Budget**: 5 minutes per experiment

### IGT Adaptation

- **`prepare.py`** - Fixed IGT deck configurations and metrics (read-only)
- **`run_igt.py`** - Task parameters and trial logic (agent-modified)
- **`iowa.md`** - Experiment instructions (human-modified)
- **Goal**: Maximize net score (advantageous - disadvantageous choices)
- **Time Budget**: 10 minutes per experiment

## Step-by-Step Setup Instructions

### Phase 1: Initial Setup

#### Create Experiment Branch

```bash
# Generate tag based on current date
TAG=$(date +%b%d | tr '[:upper:]' '[:lower:]')  # e.g., mar20

# Check if branch exists
if git show-ref --verify --quiet refs/heads/igt/$TAG; then
    echo "Branch igt/$TAG already exists"
    TAG=$(date +%b%d-%H%M)  # Add time if needed
fi

# Create and checkout branch
git checkout -b igt/$TAG
```

#### Read Context Files

- Read `USAGE.md` for repository context
- Read `prepare.py` to understand fixed constants and deck configurations
- Read `run_igt.py` to understand modifiable parameters

#### Verify Data Setup

```bash
# Check if stimuli data exists
if [ ! -d "data/stimuli" ]; then
    echo "Running data preparation..."
    uv run prepare.py
fi
```

#### Initialize Results Tracking

```bash
# Create results file with header
echo -e "commit\tnet_score\ttime_min\tmemory_gb\tstatus\tdescription" > results.tsv
```

### Phase 2: Autonomous Experiment Loop

The core innovation is the infinite autonomous loop that continuously optimizes the task:

**LOOP FOREVER:**

#### Check Git State

```bash
# Ensure clean working directory
if [ -n "$(git status --porcelain)" ]; then
    echo "Working directory not clean - committing changes"
    git add .
    git commit -m "Experiment checkpoint"
fi
```

#### Generate Experiment Idea

- Review previous results in `results.tsv`
- Identify promising modification directions
- Examples: adjust reward magnitudes, modify feedback delays, change trial counts

#### Modify `run_igt.py`

- Edit task parameters within allowed scope
- Maintain core IGT logic integrity
- Log modification description

#### Commit Changes

```bash
git add run_igt.py
git commit -m "Experiment: [description]"
```

#### Execute Experiment

```bash
# Run with timeout and logging
timeout 15m uv run run_igt.py > run.log 2>&1
EXIT_CODE=$?
```

#### Extract Results

```bash
# Parse key metrics from log
net_score=$(grep "^net_score:" run.log | awk '{print $2}')
completion_time=$(grep "^completion_time_s:" run.log | awk '{print $2}')
peak_memory=$(grep "^peak_vram_mb:" run.log | awk '{print $2}')

# Convert to required units
time_min=$(echo "$completion_time / 60" | bc -l | xargs printf "%.1f")
memory_gb=$(echo "$peak_memory / 1024" | bc -l | xargs printf "%.1f")
```

#### Handle Crashes/Timeouts

```bash
if [ $EXIT_CODE -eq 124 ]; then
    # Timeout - mark as failure
    echo "Experiment timed out"
    status="crash"
elif [ $EXIT_CODE -ne 0 ]; then
    # Crash - check if trivial
    tail -n 50 run.log
    # If trivial error, fix and retry; otherwise mark as crash
    status="crash"
else
    status="keep"  # tentatively
fi
```

#### Record Results

```bash
commit_hash=$(git rev-parse --short HEAD)
echo -e "$commit_hash\t$net_score\t$time_min\t$memory_gb\t$status\t$description" >> results.tsv
```

#### Decision Logic

```bash
# Compare with previous best score
best_score=$(awk 'NR>1 && $5=="keep" {print $2}' results.tsv | sort -nr | head -1)

if (( $(echo "$net_score > $best_score" | bc -l) )); then
    echo "Improvement! Keeping commit"
    # Commit is already kept
else
    echo "No improvement. Resetting..."
    git reset --hard HEAD~1
fi
```

### Phase 3: Experiment Strategy

#### Allowed Modifications in `run_igt.py`

- **Task Parameters**: Number of trials, inter-trial intervals, deck selection probabilities
- **Reward Schedules**: Win/loss magnitudes, frequency distributions
- **Feedback Mechanisms**: Visual/auditory feedback timing, delay durations
- **Data Collection**: Additional metrics (reaction times, confidence ratings)
- **Analysis**: Learning curve calculations, statistical tests

#### Prohibited Modifications

- Changing core deck probabilities in `prepare.py`
- Adding external dependencies
- Modifying evaluation metric definition

#### Optimization Targets

- **Primary**: Net score (advantageous - disadvantageous deck selections)
- **Secondary**: Learning rate, completion time, memory efficiency

### Phase 4: Advanced Adaptations

#### For Psychological Research Integration

##### Stimulus Presentation Tools

```python
# Integrate with psychopy for precise timing
from psychopy import visual, core

# Replace simple feedback with psychopy stimuli
def show_feedback(win, outcome, deck_type):
    feedback_text = visual.TextStim(win, text=f"Outcome: {outcome}")
    feedback_text.draw()
    win.flip()
    core.wait(1.0)  # Controlled feedback duration
```

##### Advanced Data Analysis

```python
# Add statistical tests to run_igt.py
import scipy.stats as stats

def analyze_learning_curve(choices, outcomes):
    # Calculate learning rate across trial blocks
    blocks = np.array_split(choices, 5)  # 5 blocks
    learning_scores = [calculate_block_score(block) for block in blocks]
    
    # Test for significant learning trend
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        range(len(learning_scores)), learning_scores
    )
    return learning_rate, p_value
```

##### Reproducibility Enhancements

```python
# Log all random seeds and parameters
experiment_config = {
    'random_seed': np.random.randint(0, 10000),
    'deck_config': deck_configuration,
    'reward_schedule': reward_magnitudes,
    'feedback_delay': delay_duration,
    'timestamp': datetime.now().isoformat()
}
```

## Key Advantages

- **Autonomous Optimization**: Continuously improves task sensitivity without human intervention
- **Systematic Exploration**: Tests parameter combinations systematically
- **Reproducible Research**: Every change tracked with git and results logged
- **Time-Bounded Experiments**: Each run limited to 10 minutes for high-throughput testing
- **Adaptive to Platform**: Optimizes for specific hardware/time constraints

## Expected Workflow

- **Setup Phase** (30 minutes): Branch creation, data verification, baseline run
- **Autonomous Phase** (Overnight): ~60 experiments (6 hours × 6 experiments/hour)
- **Analysis Phase** (Morning): Review `results.tsv`, identify optimal configuration
- **Validation Phase**: Run final optimized configuration with multiple participants

This system transforms psychological task optimization from manual trial-and-error to autonomous, systematic exploration, potentially discovering novel parameter combinations that maximize task sensitivity for detecting decision-making differences.

## APGI Experiments - Auto-Improvement System

## System Overview

This directory contains experiment protocols for the APGI (Allostatic Precision-Gated Ignition) auto-improvement system. Each experiment follows a two-file structure:

- **`prepare_<experiment>.py`** - Fixed configurations and data preparation (READ-ONLY)
- **`run_<experiment>.py`** - Agent-editable task parameters and simulation logic

## Core Principles

- **Time Budget**: All experiments have a 10-minute (600 second) time budget
- **Goal**: Maximize the primary metric defined in each experiment
- **Git-based Learning**: The system is designed for an AI agent to autonomously modify `run_*.py` files, execute experiments, and track results via Git commits

## Available Experiments

### Primary IGT Experiments

#### **Iowa Gambling Task (IGT)**

- **Files**: `prepare_iowa_gambling_task.py`, `run_iowa_gambling_task.py`
- **Primary Metric**: `net_score` (advantageous - disadvantageous choices)
- **Description**: Decision-making under uncertainty with four decks, win/loss probabilities, and learning

#### **Attentional Blink**

- **Files**: `prepare_attentional_blink.py`, `run_attentional_blink.py`
- **Primary Metric**: `blink_magnitude` (attentional blink effect at optimal lags)
- **Description**: Rapid serial visual presentation with attentional bottlenecks

#### **Visual Search**

- **Files**: `prepare_visual_search.py`, `run_visual_search.py`
- **Primary Metric**: `conjunction_present_slope` (ms/item search efficiency)
- **Description**: Visual search for targets among distractors with varying display sizes

#### **Posner Cueing**

- **Files**: `prepare_posner_cueing.py`, `run_posner_cueing.py`
- **Primary Metric**: `validity_effect_ms` (cue validity effect on reaction times)
- **Description**: Spatial cueing with valid/invalid trial types and SOA manipulation

#### **Stroop Effect**

- **Files**: `prepare_stroop_effect.py`, `run_stroop_effect.py`
- **Primary Metric**: `interference_effect_ms` (congruent vs incongruent RT difference)
- **Description**: Cognitive interference with color-word compatibility tasks

#### **Go/No-Go**

- **Files**: `prepare_go_no_go.py`, `run_go_no_go.py`
- **Primary Metric**: `d_prime` (response inhibition)
- **Description**: Response inhibition with frequent go stimuli and rare no-go trials

#### **Dual N-Back**

- **Files**: `prepare_dual_n_back.py`, `run_dual_n_back.py`
- **Primary Metric**: `d_prime` (working memory performance)
- **Description**: Working memory task with n-back levels and target matching

#### **Stop Signal**

- **Files**: `prepare_stop_signal.py`, `run_stop_signal.py`
- **Primary Metric**: `ssrt_ms` (stop-signal reaction time)
- **Description**: Inhibitory control with varying stop-signal delays

#### **Change Blindness**

- **Files**: `prepare_change_blindness.py`, `run_change_blindness.py`
- **Primary Metric**: `detection_rate` (ability to detect changes)
- **Description**: Change detection with masked stimuli and varying mask durations

#### **Eriksen Flanker**

- **Files**: `prepare_eriksen_flanker.py`, `run_eriksen_flanker.py`
- **Primary Metric**: `flanker_effect_ms` (incongruent vs congruent RT difference)
- **Description**: Executive function with arrow stimulus interference

#### **Masking**

- **Files**: `prepare_masking.py`, `run_masking.py`
- **Primary Metric**: `masking_effect_ms` (backward masking interference)
- **Description**: Visual masking with various mask types and SOA conditions

#### **Binocular Rivalry**

- **Files**: `prepare_binocular_rivalry.py`, `run_binocular_rivalry.py`
- **Primary Metric**: `masking_effect_ms` (binocular rivalry alternation rate)
- **Description**: Binocular rivalry with dichoptic presentation and perceptual switching

#### **Inattentional Blindness**

- **Files**: `prepare_inattentional_blindness.py`, `run_inattentional_blindness.py`
- **Primary Metric**: `accuracy` (global vs local stimulus detection)
- **Description**: Sustained attention to global vs local stimulus features

#### **Sternberg Memory**

- **Files**: `prepare_sternberg_memory.py`, `run_sternberg_memory.py`
- **Primary Metric**: `search_slope_ms_per_item` (memory scanning efficiency)
- **Description**: Working memory with set sizes and memory scanning performance

#### **Working Memory Span**

- **Files**: `prepare_working_memory_span.py`, `run_working_memory_span.py`
- **Primary Metric**: `d_prime` (working memory capacity)
- **Description**: Complex working memory task with letter sequences and distractions

#### **DRM False Memory**

- **Files**: `prepare_drm_false_memory.py`, `run_drm_false_memory.py`
- **Primary Metric**: `accuracy` (recognition memory performance)
- **Description**: Recognition memory with lists, sentences, and recognition accuracy

#### **Navon Task (Global-Local)**

- **Files**: `prepare_navon_task.py`, `run_navon_task.py`
- **Primary Metric**: `global_advantage_ms` (global vs local processing advantage)
- **Description**: Attentional processing with global vs local stimulus features

#### **Multisensory Integration**

- **Files**: `prepare_multisensory_integration.py`, `run_multisensory_integration.py`
- **Primary Metric**: `multisensory_gain_ms` (cross-modal integration benefit)
- **Description**: Multisensory processing with visual, auditory, and bimodal stimuli

#### **Serial Reaction Time**

- **Files**: `prepare_serial_reaction_time.py`, `run_serial_reaction_time.py`
- **Primary Metric**: `learning_effect_ms` (procedural learning improvement)
- **Description**: Simple and choice reaction time with learning components

#### **Time Estimation**

- **Files**: `prepare_time_estimation.py`, `run_time_estimation.py`
- **Primary Metric**: `mean_error_percent` (temporal estimation accuracy)
- **Description**: Time perception and estimation with varying intervals

#### **Artificial Grammar Learning**

- **Files**: `prepare_artificial_grammar_learning.py`, `run_artificial_grammar_learning.py`
- **Primary Metric**: `grammar_accuracy` (rule learning performance)
- **Description**: Learning artificial grammar rules with feedback and complexity levels

#### **Virtual Navigation**

- **Files**: `prepare_virtual_navigation.py`, `run_virtual_navigation.py`
- **Primary Metric**: `path_efficiency` (navigation planning and execution efficiency)
- **Description**: Spatial navigation and planning in virtual environments

#### **Probabilistic Category Learning**

- **Files**: `prepare_probabilistic_category_learning.py`, `run_probabilistic_category_learning.py`
- **Primary Metric**: `learning_rate` (category learning and adaptation)
- **Description**: Probabilistic learning of categories with feedback and strategy switching

#### **Interoceptive Gating**

- **Files**: `prepare_interoceptive_gating.py`, `run_interoceptive_gating.py`
- **Primary Metric**: `gating_threshold` (interoceptive signal detection)
- **Description**: Interoceptive signal detection with varying stimulus intensities

#### **Somatic Marker Priming**

- **Files**: `prepare_somatic_marker_priming.py`, `run_somatic_marker_priming.py`
- **Primary Metric**: `priming_effect_ms` (somatic influence on perception)
- **Description**: Somatic marker priming with same/different markers and response effects

#### **Metabolic Cost**

- **Files**: `prepare_metabolic_cost.py`, `run_metabolic_cost.py`
- **Primary Metric**: `metabolic_cost_ratio` (energetic efficiency)
- **Description**: Resource allocation and decision-making with metabolic costs

## Usage Instructions

### For Each Experiment

#### Running Individual Experiments

```bash
# Run a specific experiment
uv run run_<experiment>.py

# Example: Run the Iowa Gambling Task
uv run run_iowa_gambling_task.py
```

#### Auto-Improvement Workflow

1. **Modify Parameters**: Edit `run_<experiment>.py` to optimize task parameters
2. **Execute Experiment**: Run the experiment to collect performance data
3. **Analyze Results**: Review primary metric and adjust parameters accordingly
4. **Commit Changes**: Use Git to track parameter modifications and performance improvements

### Key Metrics for Auto-Improvement

Each experiment outputs its primary metric as the final line, formatted for easy parsing by the auto-improvement system:

```text
primary_metric: <value>
completion_time_s: <seconds>
```

## File Structure

```text
auto-improvement/
├── prepare_<experiment>.py     # Fixed configurations (READ-ONLY)
├── run_<experiment>.py         # Modifiable parameters and simulation
└── USAGE.md                  # This documentation
```

## Development Guidelines

- **Time Budget**: All experiments respect the 600-second time limit
- **Reproducibility**: Fixed configurations ensure consistent experimental conditions
- **Modifiability**: Agent parameters allow for systematic optimization
- **Performance Tracking**: Primary metrics enable automated performance comparison
- **Git Integration**: Version control tracks the evolution of optimal strategies

## Implementation Notes

- All `prepare_*.py` files import from their corresponding modules
- All `run_*.py` files follow the IGT template pattern
- Simulated participants model human-like performance characteristics
- Each experiment includes comprehensive result calculation and display

This system enables systematic exploration and optimization of cognitive task performance through automated experimentation.
