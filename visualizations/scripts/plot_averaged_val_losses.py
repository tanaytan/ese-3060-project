#!/usr/bin/env python3
"""
Script to average validation losses from multiple log files and plot them.
Averages baseline_1, baseline_2, baseline_3 into baseline_val_avg
Averages gqa_1, gqa_2, gqa_3 into gqa_val_avg
Plots these along with baseline_mdha_gqa_1

Usage:
    python plot_averaged_val_losses.py
"""

import re
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import numpy as np


def parse_val_losses(log_file_path):
    """
    Parse a log file and extract step numbers, cumulative train_time (ms), and val_loss values.
    Returns only validation entries from the last 1000 training steps.
    
    Returns:
        val_data: list of tuples (step, time, loss) sorted by step
    """
    all_val_entries = []
    max_step = 0
    
    # Patterns to match log lines - capture step, loss, and train_time
    train_pattern = re.compile(r'step:(\d+)/\d+\s+train_loss:([\d.]+)\s+train_time:([\d.]+)ms')
    val_pattern = re.compile(r'step:(\d+)/\d+\s+val_loss:([\d.]+)\s+train_time:([\d.]+)ms')
    
    # train_time resets at step 10, so we need to track the offset
    time_offset = 0.0
    last_time_before_reset = 0.0
    
    with open(log_file_path, 'r') as f:
        for line in f:
            # Check for train_loss to track time offset and max step
            train_match = train_pattern.search(line)
            if train_match:
                step = int(train_match.group(1))
                train_time_ms = float(train_match.group(3))
                max_step = max(max_step, step)
                
                # Handle the reset at step 10
                if step == 10:
                    last_time_before_reset = train_time_ms
                elif step == 11:
                    # After reset, calculate offset needed to make times continuous
                    time_offset = last_time_before_reset
            
            # Check for val_loss
            val_match = val_pattern.search(line)
            if val_match:
                step = int(val_match.group(1))
                loss = float(val_match.group(2))
                train_time_ms = float(val_match.group(3))
                max_step = max(max_step, step)
                
                # Calculate cumulative time
                if step >= 11:
                    cumulative_time = time_offset + train_time_ms
                else:
                    cumulative_time = train_time_ms
                
                all_val_entries.append((step, cumulative_time, loss))
    
    # Filter to only include entries from the last 1000 steps
    min_step = max(0, max_step - 1000)
    filtered_entries = [(step, time, loss) for step, time, loss in all_val_entries if step >= min_step]
    
    # Sort by step
    filtered_entries.sort(key=lambda x: x[0])
    
    return filtered_entries


def average_val_losses(log_files, label):
    """
    Average validation losses from multiple log files.
    
    Args:
        log_files: list of log file paths
        label: label for the averaged series
    
    Returns:
        averaged_data: list of tuples (step, avg_time, avg_loss)
    """
    all_data = []
    
    # Parse all files
    for log_file in log_files:
        if not Path(log_file).exists():
            print(f"Warning: File '{log_file}' not found, skipping...")
            continue
        print(f"Parsing {log_file}...")
        data = parse_val_losses(log_file)
        all_data.append(data)
    
    if not all_data:
        return []
    
    # Group by step number
    step_data = defaultdict(list)  # step -> list of (time, loss) tuples
    
    for data in all_data:
        for step, time, loss in data:
            step_data[step].append((time, loss))
    
    # Average at each step
    averaged_data = []
    for step in sorted(step_data.keys()):
        times = [t for t, _ in step_data[step]]
        losses = [l for _, l in step_data[step]]
        avg_time = np.mean(times)
        avg_loss = np.mean(losses)
        averaged_data.append((step, avg_time, avg_loss))
    
    print(f"  Averaged {len(averaged_data)} validation entries for {label}")
    return averaged_data


def plot_averaged_val_losses(baseline_data, gqa_data, mdha_data, output_file):
    """
    Plot averaged validation losses.
    
    Args:
        baseline_data: list of (step, time, loss) tuples for baseline average
        gqa_data: list of (step, time, loss) tuples for gqa average
        mdha_data: list of (step, time, loss) tuples for baseline_mdha_gqa_1
        output_file: path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Extract times and losses
    if baseline_data:
        baseline_times = [time for _, time, _ in baseline_data]
        baseline_losses = [loss for _, _, loss in baseline_data]
        plt.plot(baseline_times, baseline_losses, label='baseline_val_avg', 
                alpha=0.7, linewidth=2, color='#1f77b4', marker='o', markersize=4)
    
    if gqa_data:
        gqa_times = [time for _, time, _ in gqa_data]
        gqa_losses = [loss for _, _, loss in gqa_data]
        plt.plot(gqa_times, gqa_losses, label='gqa_val_avg', 
                alpha=0.7, linewidth=2, color='#ff7f0e', marker='s', markersize=4)
    
    if mdha_data:
        mdha_times = [time for _, time, _ in mdha_data]
        mdha_losses = [loss for _, _, loss in mdha_data]
        plt.plot(mdha_times, mdha_losses, label='baseline_mdha_gqa_1', 
                alpha=0.7, linewidth=2, color='#2ca02c', marker='^', markersize=4)
    
    plt.xlabel('Train Time (ms)', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('Averaged Validation Loss Over Time (Last 1000 Steps)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Also show the plot
    plt.show()


def main():
    logs_dir = Path('logs')
    
    # Define file groups
    baseline_files = [
        logs_dir / 'baseline_1.txt',
        logs_dir / 'baseline_2.txt',
        logs_dir / 'baseline_3.txt'
    ]
    
    gqa_files = [
        logs_dir / 'gqa_1.txt',
        logs_dir / 'gqa_2.txt',
        logs_dir / 'gqa_3.txt'
    ]
    
    mdha_file = logs_dir / 'baseline_mdha_gqa_1.txt'
    
    print("Averaging validation losses from the last 1000 steps...")
    print("=" * 60)
    
    # Average baseline files
    baseline_data = average_val_losses(baseline_files, 'baseline_val_avg')
    
    # Average gqa files
    gqa_data = average_val_losses(gqa_files, 'gqa_val_avg')
    
    # Parse mdha file
    mdha_data = []
    if mdha_file.exists():
        print(f"Parsing {mdha_file}...")
        mdha_data = parse_val_losses(mdha_file)
        print(f"  Found {len(mdha_data)} validation entries")
    else:
        print(f"Warning: File '{mdha_file}' not found, skipping...")
    
    if not baseline_data and not gqa_data and not mdha_data:
        print("Error: No validation loss data found.")
        sys.exit(1)
    
    # Generate output filename
    output_file = 'averaged_val_losses.png'
    
    plot_averaged_val_losses(baseline_data, gqa_data, mdha_data, output_file)


if __name__ == '__main__':
    main()
