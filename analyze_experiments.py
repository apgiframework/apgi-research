#!/usr/bin/env python3
"""
APGI Experiment Results Analyzer and Reporter
Analyzes metrics from all experiments, fixes validation issues,
creates visualizations, and generates comprehensive reports.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Type definition for experiment data
ExperimentData = Dict[str, Any]

# Parsed APGI metrics from the console output
EXPERIMENT_RESULTS: Dict[str, ExperimentData] = {
    "Artificial Grammar Learning": {
        "primary_metric": 0.475,
        "accuracy": 47.5,
        "ignition_rate": 2.50,
        "mean_surprise": 0.008,
        "metabolic_cost": 0.003,
        "mean_somatic_marker": 0.012,
        "mean_threshold": 0.498,
        "d_prime": -0.050,
        "completion_time_s": 0.00,
        "trials": 40,
        "status": "Success",
    },
    "Attentional Blink": {
        "primary_metric": 0.6458,
        "accuracy": 64.58,
        "ignition_rate": 11.00,
        "mean_surprise": 0.031,
        "metabolic_cost": 0.031,
        "mean_somatic_marker": -0.053,
        "mean_threshold": 0.503,
        "blink_magnitude": 0.500,
        "completion_time_s": 0.01,
        "trials": 100,
        "status": "Success",
    },
    "Ai Benchmarking": {
        "primary_metric": 0.540,
        "benchmark_accuracy": 0.540,
        "completion_time_s": 50.2,
        "trials": 50,
        "correct": 27,
        "incorrect": 23,
        "mean_response_time_ms": 2170.9,
        "total_tokens": 5700,
        "status": "Success",
    },
    "Binocular Rivalry": {
        "primary_metric": 0.227,
        "alternation_rate": 0.227,
        "completion_time_s": 120.3,
        "trials": 60,
        "total_alternations": 817,
        "mean_dominance_duration_s": 4.10,
        "dominance_duration_cv": 1.516,
        "status": "Success",
    },
    "Change Blindness": {
        "primary_metric": 0.176,
        "detection_rate": 0.176,
        "completion_time_s": 0.00,
        "trials": 60,
        "correct_rejection_rate": 80.8,
        "threshold_ms": 500,
        "mean_rt_ms": 489.3,
        "status": "Success",
    },
    "Change Blindness Full Apgi": {
        "primary_metric": 0.000,
        "detection_rate": 0.000,
        "ignition_rate": 8.33,
        "mean_surprise": 0.023,
        "mean_threshold": 0.602,
        "mean_somatic_marker": -0.026,
        "metabolic_cost": 0.014,
        "completion_time_s": 0.0,
        "status": "Success",
    },
    "Drm False Memory": {"status": "Pending"},
    "Dual N Back": {
        "primary_metric": 0.208,
        "d_prime": 0.208,
        "ignition_rate": 6.25,
        "mean_surprise": 0.029,
        "metabolic_cost": 0.023,
        "mean_somatic_marker": -0.009,
        "mean_threshold": 0.514,
        "completion_time_s": 0.01,
        "trials": 80,
        "n_back_level": 2,
        "hit_rate": 68.8,
        "correct_rejection": 52.1,
        "mean_rt_ms": 595.1,
        "status": "Success",
    },
    "Eriksen Flanker": {
        "primary_metric": 88.55,
        "accuracy": 97.0,
        "flanker_effect_ms": 88.55,
        "ignition_rate": 9.00,
        "mean_surprise": 0.014,
        "metabolic_cost": 0.014,
        "mean_somatic_marker": 0.054,
        "mean_threshold": 0.499,
        "completion_time_s": 0.02,
        "trials": 100,
        "congruent_rt_ms": 502.7,
        "incongruent_rt_ms": 591.2,
        "status": "Success",
    },
    "Go No Go": {
        "primary_metric": 0.933,
        "d_prime": 0.933,
        "accuracy": 100.0,
        "ignition_rate": 10.00,
        "mean_surprise": 0.013,
        "metabolic_cost": 0.013,
        "mean_somatic_marker": -0.017,
        "mean_threshold": 0.498,
        "completion_time_s": 0.01,
        "trials": 100,
        "go_accuracy": 100.0,
        "no_go_accuracy": 93.3,
        "mean_go_rt_ms": 448.0,
        "status": "Success",
    },
    "Igt": {"status": "Pending"},
    "Inattentional Blindness": {
        "primary_metric": 0.950,
        "accuracy": 95.0,
        "completion_time_s": 60.1,
        "trials": 40,
        "unexpected_trials": 6,
        "detected_count": 4,
        "detection_accuracy": 0.667,
        "false_alarm_rate": 0.0,
        "mean_response_time_ms": 1231.2,
        "status": "Success",
    },
    "Interoceptive Gating": {
        "primary_metric": 0.743,
        "gating_threshold": 0.743,
        "completion_time_s": 0.00,
        "trials": 100,
        "gating_effect": 0.34,
        "hit_rate": 0.0,
        "false_alarm_rate": 0.0,
        "mean_rt_ms": 0.0,
        "status": "Success",
    },
    "Iowa Gambling Task": {
        "primary_metric": 16.0,
        "net_score": 16.0,
        "ignition_rate": 15.00,
        "mean_surprise": 0.040,
        "metabolic_cost": 0.040,
        "mean_somatic_marker": 0.002,
        "mean_threshold": 0.398,
        "completion_time_s": 0.00,
        "trials": 100,
        "final_money": 2825,
        "advantageous_choices": 88,
        "disadvantageous_choices": 12,
        "learning_rate": 0.060,
        "status": "Success",
    },
    "Masking": {"status": "Pending"},
    "Metabolic Cost": {
        "primary_metric": 1.000,
        "metabolic_cost_ratio": 1.000,
        "completion_time_s": 0.00,
        "trials": 80,
        "mean_cost": 0.5,
        "performance_score": 0.500,
        "status": "Success",
    },
    "Multisensory Integration": {
        "primary_metric": 66.97,
        "accuracy": 95.0,
        "multisensory_gain_ms": 66.97,
        "ignition_rate": 11.00,
        "mean_surprise": 0.018,
        "metabolic_cost": 0.018,
        "mean_somatic_marker": 0.052,
        "mean_threshold": 0.509,
        "completion_time_s": 0.01,
        "trials": 100,
        "visual_rt_ms": 469.4,
        "auditory_rt_ms": 403.9,
        "bimodal_rt_ms": 369.7,
        "status": "Success",
    },
    "Navon Task": {
        "primary_metric": 74.74,
        "accuracy": 94.0,
        "global_advantage_ms": 74.74,
        "ignition_rate": 5.00,
        "mean_surprise": 0.009,
        "metabolic_cost": 0.009,
        "mean_somatic_marker": 0.010,
        "mean_threshold": 0.494,
        "completion_time_s": 0.00,
        "trials": 100,
        "global_rt_ms": 438.1,
        "local_rt_ms": 512.9,
        "interference_effect_ms": 28.5,
        "status": "Success",
    },
    "Posner Cueing": {
        "primary_metric": 62.19,
        "validity_effect_ms": 62.19,
        "ignition_rate": 7.00,
        "mean_surprise": 0.017,
        "metabolic_cost": 0.017,
        "mean_somatic_marker": 0.074,
        "mean_threshold": 0.483,
        "completion_time_s": 0.00,
        "trials": 100,
        "valid_rt_ms": 354.8,
        "invalid_rt_ms": 417.0,
        "benefit_ms": 27.1,
        "cost_ms": 35.0,
        "status": "Success",
    },
    "Probabilistic Category Learning": {
        "primary_metric": 0.75,
        "accuracy": 75.0,
        "learning_rate": 0.12,
        "completion_time_s": 45.2,
        "trials": 100,
        "status": "Success",
    },
    "Serial Reaction Time": {
        "primary_metric": 79.80,
        "accuracy": 91.7,
        "learning_effect_ms": 79.80,
        "ignition_rate": 8.50,
        "mean_surprise": 0.018,
        "metabolic_cost": 0.015,
        "mean_somatic_marker": 0.045,
        "mean_threshold": 0.498,
        "completion_time_s": 52.3,
        "trials": 120,
        "sequential_rt_ms": 441.0,
        "random_rt_ms": 520.8,
        "status": "Success",
    },
    "Simon Effect": {
        "primary_metric": 36.88,
        "accuracy": 93.8,
        "simon_effect_ms": 36.88,
        "completion_time_s": 0.01,
        "trials": 80,
        "congruent_rt_ms": 445.1,
        "incongruent_rt_ms": 482.0,
        "status": "Success",
    },
    "Somatic Marker Priming": {
        "primary_metric": 28.5,
        "accuracy": 85.0,
        "priming_effect_ms": 28.5,
        "ignition_rate": 5.00,
        "mean_surprise": 0.033,
        "metabolic_cost": 0.033,
        "mean_somatic_marker": 0.102,
        "mean_threshold": 0.507,
        "completion_time_s": 48.7,
        "trials": 100,
        "same_marker_rt_ms": 398.0,
        "different_marker_rt_ms": 426.5,
        "status": "Success",
    },
    "Sternberg Memory": {
        "primary_metric": 39.87,
        "accuracy": 94.0,
        "search_slope_ms_per_item": 39.87,
        "completion_time_s": 0.01,
        "trials": 100,
        "positive_rt_ms": 585.3,
        "negative_rt_ms": 596.1,
        "status": "Success",
    },
    "Stop Signal": {
        "primary_metric": 198.20,
        "accuracy": 81.8,
        "ssrt_ms": 198.20,
        "completion_time_s": 0.01,
        "trials": 100,
        "go_accuracy": 100.0,
        "stop_success_rate": 81.8,
        "status": "Success",
    },
    "Stroop Effect": {
        "primary_metric": 108.09,
        "accuracy": 88.8,
        "interference_effect_ms": 108.09,
        "ignition_rate": 4.38,
        "mean_surprise": 0.009,
        "metabolic_cost": 0.014,
        "mean_somatic_marker": -0.011,
        "mean_threshold": 0.525,
        "completion_time_s": 0.00,
        "trials": 80,
        "congruent_rt_ms": 584.4,
        "incongruent_rt_ms": 692.5,
        "status": "Success",
    },
    "Time Estimation": {
        "primary_metric": -13.87,
        "mean_error_percent": -13.87,
        "ignition_rate": 8.00,
        "mean_surprise": 0.024,
        "metabolic_cost": 0.012,
        "mean_somatic_marker": -0.003,
        "mean_threshold": 0.499,
        "completion_time_s": 0.00,
        "trials": 50,
        "mean_error_ms": -321.9,
        "variability_cv": 1.106,
        "status": "Success",
    },
    "Virtual Navigation": {
        "primary_metric": 0.9383,
        "path_efficiency": 0.9383,
        "completion_time_s": 0.00,
        "trials": 20,
        "mean_excess_length_steps": 1.0,
        "status": "Success",
    },
    "Visual Search": {
        "primary_metric": 25.66,
        "accuracy": 95.0,
        "conjunction_present_slope": 25.66,
        "ignition_rate": 5.62,
        "mean_surprise": 0.009,
        "metabolic_cost": 0.014,
        "mean_somatic_marker": 0.015,
        "mean_threshold": 0.519,
        "completion_time_s": 0.94,
        "trials": 80,
        "feature_search_slope_ms_per_item": 4.16,
        "conjunction_search_slope_ms_per_item": 25.66,
        "slope_ratio": 6.16,
        "status": "Success",
    },
    "Working Memory Span": {
        "primary_metric": 1.383,
        "overall_accuracy": 0.211,
        "d_prime": 1.383,
        "completion_time_s": 60.2,
        "trials": 60,
        "working_span": 3,
        "mean_response_time_ms": 3275.1,
        "status": "Success",
    },
}


def get_apgi_experiments() -> Dict[str, Dict[str, Any]]:
    """Get all experiments with APGI metrics."""
    apgi_keys = [
        "ignition_rate",
        "mean_surprise",
        "metabolic_cost",
        "mean_somatic_marker",
        "mean_threshold",
    ]
    return {
        name: data
        for name, data in EXPERIMENT_RESULTS.items()
        if any(k in data for k in apgi_keys)
    }


def analyze_apgi_metrics() -> Dict[str, Any]:
    """Analyze APGI metrics across all experiments."""
    apgi_exp: Dict[str, ExperimentData] = get_apgi_experiments()

    analysis: Dict[str, Any] = {
        "total_experiments": len(apgi_exp),
        "metrics_summary": {},
        "correlations": {},
        "outliers": [],
    }

    # Collect metrics
    metrics_data: Dict[str, List[Tuple[str, Any]]] = {
        "ignition_rate": [],
        "mean_surprise": [],
        "metabolic_cost": [],
        "mean_somatic_marker": [],
        "mean_threshold": [],
    }

    for name, data in apgi_exp.items():
        for key in metrics_data:
            if key in data:
                metrics_data[key].append((name, data[key]))

    # Calculate statistics
    for metric, values in metrics_data.items():
        if values:
            nums = [v[1] for v in values]
            analysis["metrics_summary"][metric] = {
                "count": len(nums),
                "mean": sum(nums) / len(nums),
                "min": min(nums),
                "max": max(nums),
                "range": max(nums) - min(nums),
                "experiments": [v[0] for v in values],
            }
    # Create summary section for tests
    ignition_rates = [v[1] for v in metrics_data.get("ignition_rate", [])]
    metabolic_costs = [v[1] for v in metrics_data.get("metabolic_cost", [])]
    surprises = [v[1] for v in metrics_data.get("mean_surprise", [])]

    analysis["summary"] = {
        "total_experiments": len(apgi_exp),
        "avg_ignition_rate": (
            sum(ignition_rates) / len(ignition_rates) if ignition_rates else 0
        ),
        "avg_metabolic_cost": (
            sum(metabolic_costs) / len(metabolic_costs) if metabolic_costs else 0
        ),
        "avg_surprise": sum(surprises) / len(surprises) if surprises else 0,
    }

    # Create detailed analysis section
    analysis["detailed_analysis"] = {}
    for name, data in apgi_exp.items():
        analysis["detailed_analysis"][name] = data

    return analysis


def identify_issues() -> List[Dict[str, Any]]:
    """Identify experiments with issues."""
    issues = []

    for name, data in EXPERIMENT_RESULTS.items():
        exp_issues: List[str] = []

        # Check for zero or very low threshold
        if "mean_threshold" in data:
            if data["mean_threshold"] <= 0:
                exp_issues.append(
                    f"Mean Threshold is {data['mean_threshold']} (must be > 0)"
                )
            elif data["mean_threshold"] < 0.1:
                exp_issues.append(
                    f"Mean Threshold is very low ({data['mean_threshold']})"
                )

        # Check for zero accuracy
        if "accuracy" in data:
            if data["accuracy"] == 0:
                exp_issues.append(
                    "Accuracy is 0% (experiment may not be running correctly)"
                )
            elif data["accuracy"] < 10:
                exp_issues.append("Accuracy is very low ({}%)".format(data["accuracy"]))

        # Check for zero ignition rate
        if "ignition_rate" in data and data["ignition_rate"] == 0:
            exp_issues.append("Ignition Rate is 0% (APGI dynamics not activated)")

        # Check for zero metabolic cost
        if "metabolic_cost" in data and data["metabolic_cost"] == 0:
            exp_issues.append("Metabolic Cost is 0 (check APGI integration)")

        # Check for missing primary metric
        if "primary_metric" not in data and data.get("status") == "Success":
            exp_issues.append("Missing primary metric for successful experiment")

        # Check for abnormal completion times
        if "completion_time_s" in data:
            if data["completion_time_s"] > 300:  # 5 minutes
                exp_issues.append(
                    "Very long completion time ({}s)".format(data["completion_time_s"])
                )
            elif data["completion_time_s"] < 0.001:  # Less than 1ms
                exp_issues.append(
                    "Suspiciously short completion time ({}s)".format(
                        data["completion_time_s"]
                    )
                )

        # Check for missing status
        if "status" not in data:
            exp_issues.append("Missing experiment status")

        if exp_issues:
            # Categorize issues by type
            if any("threshold" in i.lower() for i in exp_issues):
                issue_type = "parameter_error"
                severity = "medium"
            elif any("accuracy" in i.lower() for i in exp_issues):
                issue_type = "performance_issue"
                severity = "high"
            elif any("completion time" in i.lower() for i in exp_issues):
                issue_type = "performance_issue"
                severity = "medium"
            elif any("missing" in i.lower() for i in exp_issues):
                issue_type = "data_integrity_error"
                severity = "high"
            else:
                issue_type = "validation_error"
                severity = "medium"

            issues.append(
                {
                    "experiment": name,
                    "issues": exp_issues,
                    "severity": severity,
                    "issue_type": issue_type,
                    "description": "; ".join(exp_issues),
                    "suggested_fix": "Review and fix the identified issues",
                }
            )

    return issues


def generate_fixes() -> Dict[str, List[str]]:
    """Generate fix recommendations for each experiment with issues."""
    fixes = {}

    issues = identify_issues()
    for item in issues:
        exp_name = item["experiment"]
        exp_fixes = []

        for issue in item["issues"]:
            if "Mean Threshold" in issue and "0" in issue:
                exp_fixes.append(
                    f"In {exp_name}: Set default threshold > 0 in APGI initialization. "
                    f"Add check: if threshold <= 0: threshold = 0.5"
                )
            elif "Accuracy is 0" in issue:
                exp_fixes.append(
                    f"In {exp_name}: Check if trials are completing. "
                    f"Add trial completion validation and debug output."
                )
            elif "Accuracy is very low" in issue:
                exp_fixes.append(
                    f"In {exp_name}: Review experiment parameters. "
                    f"Check data quality and model convergence."
                )
            elif "Ignition Rate is 0" in issue:
                exp_fixes.append(
                    f"In {exp_name}: Verify APGI integration is enabled. "
                    f"Check if ignition probability is being calculated."
                )
            elif "Metabolic Cost is 0" in issue:
                exp_fixes.append(
                    f"In {exp_name}: Check APGI integration. "
                    f"Verify metabolic cost calculation is active."
                )
            elif "Missing primary metric" in issue:
                exp_fixes.append(
                    f"In {exp_name}: Ensure primary metric is logged. "
                    f"Add metric calculation and output."
                )
            elif "Very long completion time" in issue:
                exp_fixes.append(
                    f"In {exp_name}: Optimize experiment efficiency. "
                    f"Check for infinite loops or blocking operations."
                )
            elif "Suspiciously short completion time" in issue:
                exp_fixes.append(
                    f"In {exp_name}: Verify timing accuracy. "
                    f"Check for premature termination."
                )
            elif "Missing experiment status" in issue:
                exp_fixes.append(
                    f"In {exp_name}: Add status tracking. "
                    f"Implement experiment lifecycle management."
                )

        if exp_fixes:
            fixes[exp_name] = exp_fixes

    return fixes


def generate_html_report(analysis: Dict, issues: List[Dict], fixes: Dict) -> str:
    """Generate comprehensive HTML report."""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>APGI Experiment Analysis Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-dark: #1a1a2e;
            --bg-card: #16213e;
            --accent: #0f3460;
            --highlight: #e94560;
            --text: #eaeaea;
            --success: #2ecc71;
            --warning: #f39c12;
            --error: #e74c3c;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-dark);
            color: var(--text);
            line-height: 1.6;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        header {{
            text-align: center;
            padding: 40px 20px;
            background: linear-gradient(135deg, var(--accent), var(--highlight));
            border-radius: 15px;
            margin-bottom: 30px;
        }}
        header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        header p {{ opacity: 0.9; font-size: 1.1em; }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: var(--bg-card);
            padding: 25px;
            border-radius: 12px;
            border-left: 4px solid var(--highlight);
        }}
        .summary-card h3 {{
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
            opacity: 0.8;
            margin-bottom: 10px;
        }}
        .summary-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            color: var(--highlight);
        }}
        .section {{
            background: var(--bg-card);
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
        }}
        .section h2 {{
            font-size: 1.5em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--accent);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--accent);
        }}
        th {{
            background: var(--accent);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }}
        tr:hover {{ background: rgba(255,255,255,0.05); }}
        .badge {{
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
        }}
        .badge-success {{ background: var(--success); color: #000; }}
        .badge-warning {{ background: var(--warning); color: #000; }}
        .badge-error {{ background: var(--error); }}
        .chart-container {{
            position: relative;
            height: 400px;
            margin: 30px 0;
        }}
        .issue-box {{
            background: rgba(231, 76, 60, 0.1);
            border: 1px solid var(--error);
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }}
        .issue-box h4 {{ color: var(--error); margin-bottom: 8px; }}
        .fix-box {{
            background: rgba(46, 204, 113, 0.1);
            border: 1px solid var(--success);
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }}
        .fix-box h4 {{ color: var(--success); margin-bottom: 8px; }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .metric-card {{
            background: var(--accent);
            padding: 20px;
            border-radius: 10px;
        }}
        .metric-card h4 {{ margin-bottom: 10px; color: var(--highlight); }}
        .metric-value {{
            font-size: 1.8em;
            font-weight: bold;
        }}
        .metric-range {{
            font-size: 0.9em;
            opacity: 0.7;
            margin-top: 5px;
        }}
        footer {{
            text-align: center;
            padding: 30px;
            opacity: 0.6;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔬 APGI Experiment Analysis Report</h1>
            <p>Comprehensive analysis of 27 cognitive experiments with APGI integration</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>

        <div class="summary-grid">
            <div class="summary-card">
                <h3>Total Experiments</h3>
                <div class="value">27</div>
            </div>
            <div class="summary-card">
                <h3>Completed Successfully</h3>
                <div class="value">25</div>
            </div>
            <div class="summary-card">
                <h3>With APGI Metrics</h3>
                <div class="value">{analysis['total_experiments']}</div>
            </div>
            <div class="summary-card">
                <h3>Issues Found</h3>
                <div class="value" style="color: {'var(--success)' if not issues else 'var(--error)'};">{len(issues)}</div>
            </div>
        </div>
"""

    # APGI Metrics Summary Section
    html += """
        <div class="section">
            <h2>📊 APGI Metrics Summary</h2>
            <div class="metrics-grid">
"""

    for metric, stats in analysis["metrics_summary"].items():
        metric_name = metric.replace("_", " ").title()
        html += f"""
                <div class="metric-card">
                    <h4>{metric_name}</h4>
                    <div class="metric-value">{stats['mean']:.3f}</div>
                    <div class="metric-range">
                        Range: {stats['min']:.3f} - {stats['max']:.3f} |
                        Experiments: {stats['count']}
                    </div>
                </div>
"""

    html += """
            </div>
            <div class="chart-container">
                <canvas id="apgiMetricsChart"></canvas>
            </div>
        </div>
"""

    # Issues Section
    if issues:
        html += """
        <div class="section">
            <h2>⚠️ Issues Identified</h2>
"""
        for item in issues:
            severity_class = (
                "badge-error" if item["severity"] == "high" else "badge-warning"
            )
            html += f"""
            <div class="issue-box">
                <h4>{item['experiment']} <span class="badge {severity_class}">{item['severity'].upper()}</span></h4>
                <ul>
"""
            for issue in item["issues"]:
                html += f"""                    <li>{issue}</li>
"""
            html += """                </ul>
            </div>
"""
        html += """        </div>
"""

    # Fixes Section
    if fixes:
        html += """
        <div class="section">
            <h2>🔧 Recommended Fixes</h2>
"""
        for exp_name, exp_fixes in fixes.items():
            html += f"""
            <div class="fix-box">
                <h4>{exp_name}</h4>
                <ul>
"""
            for fix in exp_fixes:
                html += f"""                    <li>{fix}</li>
"""
            html += """                </ul>
            </div>
"""
        html += """        </div>
"""

    # Complete Experiment Results Table
    html += """
        <div class="section">
            <h2>📋 Complete Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Experiment</th>
                        <th>Primary Metric</th>
                        <th>Accuracy</th>
                        <th>Ignition Rate</th>
                        <th>Mean Surprise</th>
                        <th>Metabolic Cost</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
"""

    for name, data in EXPERIMENT_RESULTS.items():
        status = data.get("status", "Unknown")
        status_class = (
            "badge-success"
            if status == "Success" and not data.get("issues")
            else "badge-warning" if data.get("issues") else "badge-success"
        )

        html += f"""                    <tr>
                        <td><strong>{name}</strong></td>
                        <td>{data.get('primary_metric', 'N/A')}</td>
                        <td>{data.get('accuracy', 'N/A')}</td>
                        <td>{data.get('ignition_rate', 'N/A')}</td>
                        <td>{data.get('mean_surprise', 'N/A')}</td>
                        <td>{data.get('metabolic_cost', 'N/A')}</td>
                        <td><span class="badge {status_class}">{status}</span></td>
                    </tr>
"""

    html += """                </tbody>
            </table>
        </div>
"""

    # JavaScript for Charts
    html += """
        <script>
"""

    # APGI Metrics Chart
    apgi_exp = get_apgi_experiments()
    exp_names = list(apgi_exp.keys())
    ignition_rates = [apgi_exp[e].get("ignition_rate", 0) for e in exp_names]
    mean_surprises = [apgi_exp[e].get("mean_surprise", 0) for e in exp_names]
    metabolic_costs = [apgi_exp[e].get("metabolic_cost", 0) for e in exp_names]

    html += f"""
            const ctx = document.getElementById('apgiMetricsChart').getContext('2d');
            new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(exp_names)},
                    datasets: [
                        {{
                            label: 'Ignition Rate (%)',
                            data: {json.dumps(ignition_rates)},
                            backgroundColor: 'rgba(233, 69, 96, 0.7)',
                            borderColor: '#e94560',
                            borderWidth: 1
                        }},
                        {{
                            label: 'Mean Surprise',
                            data: {json.dumps(mean_surprises)},
                            backgroundColor: 'rgba(46, 204, 113, 0.7)',
                            borderColor: '#2ecc71',
                            borderWidth: 1
                        }},
                        {{
                            label: 'Metabolic Cost',
                            data: {json.dumps(metabolic_costs)},
                            backgroundColor: 'rgba(52, 152, 219, 0.7)',
                            borderColor: '#3498db',
                            borderWidth: 1
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ position: 'top' }},
                        title: {{ display: true, text: 'APGI Metrics Across Experiments' }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            grid: {{ color: 'rgba(255,255,255,0.1)' }},
                            ticks: {{ color: '#eaeaea' }}
                        }},
                        x: {{
                            grid: {{ display: false }},
                            ticks: {{
                                color: '#eaeaea',
                                maxRotation: 45,
                                minRotation: 45
                            }}
                        }}
                    }}
                }}
            }});
        </script>
"""

    html += """
        <footer>
            <p>APGI Research Hub - Automated Analysis Report</p>
        </footer>
    </div>
</body>
</html>
"""

    return html


def main() -> None:
    """Main function to run analysis and generate report."""
    print("=" * 60)
    print("APGI EXPERIMENT RESULTS ANALYZER")
    print("=" * 60)

    # Run analysis
    print("\n[1/4] Analyzing APGI metrics...")
    analysis = analyze_apgi_metrics()
    print(f"      Found {analysis['total_experiments']} experiments with APGI metrics")

    print("\n[2/4] Identifying issues...")
    issues = identify_issues()
    print(f"      Found {len(issues)} experiments with issues")

    print("\n[3/4] Generating fixes...")
    fixes = generate_fixes()
    print(f"      Generated {len(fixes)} fix recommendations")

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    print("\n📊 APGI Metrics Summary:")
    for metric, stats in analysis["metrics_summary"].items():
        print(f"  • {metric.replace('_', ' ').title()}:")
        print(
            f"    Mean: {stats['mean']:.3f}, Range: {stats['min']:.3f}-{stats['max']:.3f}"
        )

    if issues:
        print("\n⚠️ Issues Found:")
        for item in issues:
            print(f"\n  {item['experiment']} ({item['severity'].upper()}):")
            for issue in item["issues"]:
                print(f"    - {issue}")

    if fixes:
        print("\n🔧 Recommended Fixes:")
        for exp_name, exp_fixes in fixes.items():
            print(f"\n  {exp_name}:")
            for fix in exp_fixes:
                print(f"    → {fix[:100]}...")

    # Generate report
    print("\n[4/4] Generating HTML report...")
    report_html = generate_html_report(analysis, issues, fixes)

    # Save report
    report_path = Path("apgi_analysis_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_html)

    print(f"      Report saved to: {report_path.absolute()}")
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nOpen the report in your browser:")
    print("  file://{}".format(report_path.absolute()))


if __name__ == "__main__":
    main()
