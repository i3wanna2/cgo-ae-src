#!/usr/bin/env python3
"""
Plot tuning performance over time with phase separation.
Visualizes how performance improves during graph tuning and parameter tuning phases.
"""

import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict


# Hardware display names and colors
HW_CONFIG = {
    'NVIDIA_H800_80GB_HBM3': {'name': 'H800', 'color': '#1f77b4', 'linestyle': '--'},
    'NVIDIA_H100_80GB_HBM3': {'name': 'H100', 'color': '#1f77b4', 'linestyle': '-'},
    'NVIDIA_A100_80GB_PCIe': {'name': 'A100', 'color': '#1f77b4', 'linestyle': ':'},
    'NVIDIA_A100_SXM4_80GB': {'name': 'A100', 'color': '#1f77b4', 'linestyle': ':'},
}


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON data from file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def group_by_hardware(data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group data by hardware type."""
    grouped = defaultdict(list)
    for entry in data:
        hw = entry.get('hw', 'Unknown')
        grouped[hw].append(entry)
    return grouped


def extract_time_performance(data: List[Dict[str, Any]]) -> Tuple[List[float], List[float], List[str], float]:
    """
    Extract time and performance data, and find the phase boundary.
    Returns: (times, performances, methods, phase_boundary_time)
    """
    # Sort by time
    sorted_data = sorted(data, key=lambda x: x['real_time'])
    
    times = [d['real_time'] for d in sorted_data]
    perfs = [d['avg'] for d in sorted_data]
    methods = [d['method'] for d in sorted_data]
    
    # Find phase boundary (last "graph" entry time)
    phase_boundary = 0
    for d in sorted_data:
        if d['method'] == 'graph':
            phase_boundary = d['real_time']
    
    return times, perfs, methods, phase_boundary


def convert_latency_to_throughput(perfs: List[float], baseline: float = None) -> List[float]:
    """
    Convert latency (ms) to relative throughput/performance.
    Higher is better for the plot.
    """
    if baseline is None:
        baseline = max(perfs)  # Use worst (highest latency) as baseline
    
    # Throughput is inversely proportional to latency
    # Normalize so baseline = 1.0
    return [baseline / p for p in perfs]


def plot_tuning_curves(
    json_files: List[str],
    output_file: str = 'tuning_curve.png',
    title: str = 'Tuning Performance Over Time',
    show_throughput: bool = True,
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Plot tuning curves for multiple hardware configurations.
    """
    plt.figure(figsize=figsize)
    plt.rcParams['font.size'] = 12
    
    all_data = []
    hw_data = {}
    
    # Load all data
    for json_file in json_files:
        data = load_json_data(json_file)
        all_data.extend(data)
    
    # Group by hardware
    hw_groups = group_by_hardware(all_data)
    
    if not hw_groups:
        print("No data found!")
        return
    
    # Find global phase boundary and normalize time
    global_phase_boundary = 0
    max_time = 0
    
    for hw, data in hw_groups.items():
        _, _, _, boundary = extract_time_performance(data)
        global_phase_boundary = max(global_phase_boundary, boundary)
        for d in data:
            max_time = max(max_time, d['real_time'])
    
    # Plot each hardware
    legend_handles = []
    
    for hw, data in hw_groups.items():
        times, perfs, methods, phase_boundary = extract_time_performance(data)
        
        if not times:
            continue
        
        # Get hardware config
        hw_config = HW_CONFIG.get(hw, {'name': hw.split('_')[1] if '_' in hw else hw, 
                                        'color': '#1f77b4', 'linestyle': '-'})
        
        if show_throughput:
            # Convert to throughput (higher is better)
            y_values = convert_latency_to_throughput(perfs)
            ylabel = 'Performance (Relative Throughput)'
        else:
            # Use latency directly (lower is better)
            y_values = perfs
            ylabel = 'Latency (ms)'
        
        # Plot the curve
        line, = plt.plot(times, y_values, 
                        linestyle=hw_config['linestyle'],
                        color=hw_config['color'],
                        linewidth=2,
                        label=hw_config['name'])
        legend_handles.append(line)
        
        # Add hardware label at the end of the line
        plt.annotate(hw_config['name'], 
                    xy=(times[-1], y_values[-1]),
                    xytext=(5, 0),
                    textcoords='offset points',
                    fontsize=11,
                    color=hw_config['color'],
                    va='center')
    
    # Draw phase boundary line
    if global_phase_boundary > 0:
        plt.axvline(x=global_phase_boundary, color='#1f77b4', linestyle='-', linewidth=1.5)
    
    # Add phase labels
    ax = plt.gca()
    ylim = ax.get_ylim()
    y_label_pos = ylim[0] - (ylim[1] - ylim[0]) * 0.08
    
    # Phase labels at bottom
    if global_phase_boundary > 0:
        # "graph" label
        graph_center = global_phase_boundary / 2
        plt.text(graph_center, y_label_pos, 'graph', 
                ha='center', va='top', fontsize=12, color='#1f77b4')
        
        # "parameter tuning" label
        param_center = (global_phase_boundary + max_time) / 2
        plt.text(param_center, y_label_pos, 'parameter tuning',
                ha='center', va='top', fontsize=12, color='#1f77b4')
    
    # Style the plot
    plt.xlabel('time', fontsize=14, color='#1f77b4')
    plt.ylabel('latency (ms)', fontsize=14, color='#1f77b4')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#1f77b4')
    ax.spines['bottom'].set_color('#1f77b4')
    
    # Set tick colors
    ax.tick_params(axis='x', colors='#1f77b4')
    ax.tick_params(axis='y', colors='#1f77b4')
    
    # Add arrows to axes
    ax.annotate('', xy=(max_time * 1.05, ylim[0]), xytext=(0, ylim[0]),
                arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.5))
    ax.annotate('', xy=(0, ylim[1] * 1.05), xytext=(0, ylim[0]),
                arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.5))
    
    # Hide default axis ticks for cleaner look
    plt.xticks([])
    plt.yticks([])
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Saved plot to: {output_file}")
    
    # Also show the plot
    plt.show()


def plot_simple_style(
    json_files: List[str],
    output_file: str = 'tuning_curve.png',
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Plot in a simple style matching the reference image.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set background to white
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    all_data = []
    
    # Load all data
    for json_file in json_files:
        data = load_json_data(json_file)
        all_data.extend(data)
    
    # Group by hardware
    hw_groups = group_by_hardware(all_data)
    
    if not hw_groups:
        print("No data found!")
        return
    
    # Find global stats
    global_phase_boundary = 0
    max_time = 0
    max_latency = 0
    
    for hw, data in hw_groups.items():
        _, perfs, _, boundary = extract_time_performance(data)
        global_phase_boundary = max(global_phase_boundary, boundary)
        for d in data:
            max_time = max(max_time, d['real_time'])
            max_latency = max(max_latency, d['avg'])
    
    # Color scheme
    line_color = '#4A90D9'  # Blue color similar to the image
    
    # Line styles for different hardware
    linestyles = {
        'H800': '--',
        'H100': '-',
        'A100': ':',
    }
    
    # Plot each hardware
    for hw, data in hw_groups.items():
        times, perfs, methods, phase_boundary = extract_time_performance(data)
        
        if not times:
            continue
        
        # Get hardware name
        hw_config = HW_CONFIG.get(hw, {'name': hw.split('_')[1] if '_' in hw else hw})
        hw_name = hw_config['name']
        
        # Get linestyle
        ls = linestyles.get(hw_name, '-')
        
        # Plot
        ax.plot(times, perfs, linestyle=ls, color=line_color, 
                linewidth=2, label=hw_name)
        
        # Add label at end
        ax.annotate(hw_name, 
                   xy=(times[-1], perfs[-1]),
                   xytext=(8, 0),
                   textcoords='offset points',
                   fontsize=12,
                   color=line_color,
                   va='center',
                   fontweight='normal')
    
    # Draw phase boundary
    if global_phase_boundary > 0:
        ylim = ax.get_ylim()
        ax.axvline(x=global_phase_boundary, color=line_color, linestyle='-', linewidth=1.5)
    
    # Configure axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(line_color)
    ax.spines['bottom'].set_color(line_color)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Add arrows
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Hide ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Axis labels
    ax.set_ylabel('latency (ms)', fontsize=14, color=line_color, labelpad=10)
    ax.set_xlabel('time', fontsize=14, color=line_color, labelpad=10)
    
    # Phase labels
    if global_phase_boundary > 0:
        # Extend xlim slightly for labels
        ax.set_xlim(xlim[0], max_time * 1.1)
        
        # Calculate label positions
        graph_center = global_phase_boundary / 2
        param_center = (global_phase_boundary + max_time) / 2
        
        # Add phase labels below x-axis
        y_label = ylim[0] - (ylim[1] - ylim[0]) * 0.05
        ax.text(graph_center, y_label, 'graph', 
               ha='center', va='top', fontsize=12, color=line_color)
        ax.text(param_center, y_label, 'parameter tuning',
               ha='center', va='top', fontsize=12, color=line_color)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(output_file, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved plot to: {output_file}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot tuning performance over time')
    parser.add_argument('--input', '-i', type=str, nargs='+', required=True,
                       help='Input JSON file(s)')
    parser.add_argument('--output', '-o', type=str, default='tuning_curve.png',
                       help='Output image file')
    parser.add_argument('--style', '-s', type=str, default='simple',
                       choices=['simple', 'detailed'],
                       help='Plot style')
    parser.add_argument('--figsize', type=int, nargs=2, default=[10, 6],
                       help='Figure size (width height)')
    
    args = parser.parse_args()
    
    if args.style == 'simple':
        plot_simple_style(
            json_files=args.input,
            output_file=args.output,
            figsize=tuple(args.figsize),
        )
    else:
        plot_tuning_curves(
            json_files=args.input,
            output_file=args.output,
            figsize=tuple(args.figsize),
        )


if __name__ == '__main__':
    main()
