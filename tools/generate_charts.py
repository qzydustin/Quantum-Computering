"""
Chart generation utilities.

Currently provides:
- plot_comprehensive_analysis_from_json: generate PDF/CDF/Errors and metrics

Usage:
    from tools.generate_charts import plot_comprehensive_analysis_from_json
    png_path, metrics = plot_comprehensive_analysis_from_json("result.json")
"""

from __future__ import annotations

import os
import json
import numpy as np
import matplotlib.pyplot as plt


def plot_comprehensive_analysis_from_json(json_path: str):
    """Generate comprehensive analysis charts (PDF, CDF, error bars) and metrics.

    Args:
        json_path: Path to a result JSON that contains 'theoretical_result' and 'real_device_result'.

    Returns:
        (output_png_path, metrics_dict)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1) Theoretical probability distribution
    theo = data.get('theoretical_result', {})
    theo_probs = theo.get('probabilities')
    if theo_probs is None:
        theo_counts = theo.get('counts', {})
        total = sum(theo_counts.values())
        theo_probs = {k: v / total for k, v in theo_counts.items()} if total > 0 else {}

    # 2) Real device distribution
    real = data.get('real_device_result', {})
    real_counts = real.get('counts', {})
    total_real = sum(real_counts.values())
    real_probs = {k: v / total_real for k, v in real_counts.items()} if total_real > 0 else {}

    # 3) Unified state set and natural binary order
    all_states = sorted(set(theo_probs.keys()) | set(real_probs.keys()), key=lambda x: int(x, 2))
    state_indices = [int(state, 2) for state in all_states]
    theo_prob_list = np.array([theo_probs.get(s, 0.0) for s in all_states])
    real_prob_list = np.array([real_probs.get(s, 0.0) for s in all_states])

    # 4) Metrics
    theo_cdf = np.cumsum(theo_prob_list)
    real_cdf = np.cumsum(real_prob_list)
    with np.errstate(divide='ignore', invalid='ignore'):
        kl_div = np.sum(real_prob_list * np.log((real_prob_list + 1e-10) / (theo_prob_list + 1e-10)))
    tv_distance = 0.5 * np.sum(np.abs(theo_prob_list - real_prob_list))
    fidelity = np.sum(np.sqrt(theo_prob_list * real_prob_list))

    # 5) Figures
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # PDF comparison
    ax1 = axes[0, 0]
    x = np.arange(len(all_states))
    width = 0.35
    ax1.bar(x - width/2, theo_prob_list, width, label='Theoretical', alpha=0.7, color='skyblue')
    ax1.bar(x + width/2, real_prob_list, width, label='Real Device', alpha=0.7, color='firebrick')
    ax1.set_xlabel('Quantum States (Binary Order)')
    ax1.set_ylabel('Probability')
    ax1.set_title('Probability Distribution (PDF)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{s}\n({int(s,2)})' for s in all_states], rotation=45, ha='right', fontsize=8)

    # CDF comparison
    ax2 = axes[0, 1]
    ax2.plot(state_indices, theo_cdf, label='Theoretical CDF', color='skyblue', linewidth=2)
    ax2.plot(state_indices, real_cdf, label='Real Device CDF', color='firebrick', linewidth=2)
    ax2.set_xlabel('State Value (Decimal)')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Cumulative Distribution (CDF)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Error bars (Real - Theoretical)
    ax3 = axes[1, 0]
    error = real_prob_list - theo_prob_list
    ax3.bar(x, error, alpha=0.7, color=['red' if e < 0 else 'green' for e in error])
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_xlabel('Quantum States (Binary Order)')
    ax3.set_ylabel('Probability Error')
    ax3.set_title('Probability Error (Real - Theoretical)')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{s}\n({int(s,2)})' for s in all_states], rotation=45, ha='right', fontsize=8)

    # Metrics panel
    ax4 = axes[1, 1]
    ax4.axis('off')
    meta = data.get('circuit_info', {})
    stats_text = (
        "Statistical Metrics:\n\n"
        f"Algorithm: {meta.get('algorithm', 'Unknown').upper()}\n"
        f"Qubits: {meta.get('parameters', {}).get('n_qubits', 'N/A')}\n"
        f"Circuit Depth: {meta.get('depth', 'N/A')}\n"
        f"Backend: {data.get('backend', 'Unknown')}\n\n"
        f"• Fidelity: {fidelity:.4f}\n"
        f"• Total Variation: {tv_distance:.4f}\n"
        f"• KL Divergence: {kl_div:.4f}\n\n"
        f"• Max Theoretical Prob: {float(np.max(theo_prob_list)) if theo_prob_list.size else 0.0:.4f}\n"
        f"• Max Real Prob: {float(np.max(real_prob_list)) if real_prob_list.size else 0.0:.4f}\n"
        f"• Mean Absolute Error: {float(np.mean(np.abs(error))) if error.size else 0.0:.4f}\n"
        f"• Standard Deviation Error: {float(np.std(error)) if error.size else 0.0:.4f}"
    )
    ax4.text(
        0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10, verticalalignment='top',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
    )

    fig.suptitle("Comprehensive Quantum Circuit Analysis", fontsize=14)
    plt.tight_layout()

    output_path = os.path.splitext(json_path)[0] + "_comprehensive.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path, {
        'fidelity': float(fidelity),
        'tv_distance': float(tv_distance),
        'kl_divergence': float(kl_div),
        'mean_abs_error': float(np.mean(np.abs(error))) if error.size else 0.0,
    }


