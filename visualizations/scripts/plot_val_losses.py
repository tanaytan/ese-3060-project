#!/usr/bin/env python3
"""
Script to plot validation loss over time from multiple log files (up to 3).
Shows only validation losses from the last 1000 training steps (epochs).

Usage:
    python plot_val_losses.py <log_file1> [log_file2] [log_file3]
    python plot_val_losses.py logs/baseline_1.txt logs/gqa_3.txt
"""

import re
import sys
import matplotlib.pyplot as plt
from pathlib import Path


def parse_val_losses(log_file_path):
    """
    Parse a log file and extract cumulative train_time (ms) and val_loss values.
    Returns only validation entries from the last 1000 training steps.
    
    Returns:
        val_times: list of cumulative train_time values (ms) where val_loss was recorded
        val_losses: list of val_loss values
    """
    all_val_entries = []  # Store (step, time, loss) tuples
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
    
    # Extract times and losses
    val_times = [time for _, time, _ in filtered_entries]
    val_losses = [loss for _, _, loss in filtered_entries]
    
    return val_times, val_losses


def plot_val_losses(log_files_data, output_file):
    """
    Plot validation loss over time for multiple log files on one graph.
    
    Args:
        log_files_data: list of tuples (label, val_times, val_losses)
        output_file: path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    markers = ['o', 's', '^']
    
    for i, (label, val_times, val_losses) in enumerate(log_files_data):
        if val_times and val_losses:
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            plt.plot(val_times, val_losses, label=label, alpha=0.7, 
                    linewidth=1.5, marker=marker, markersize=4, color=color)
    
    plt.xlabel('Train Time (ms)', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('Validation Loss Over Time (Last 1000 Steps)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Also show the plot
    plt.show()


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage: python plot_val_losses.py <log_file1> [log_file2] [log_file3]")
        print("       Up to 3 log files can be plotted together.")
        sys.exit(1)
    
    log_files = sys.argv[1:]
    
    # Validate files exist
    for log_file in log_files:
        if not Path(log_file).exists():
            print(f"Error: File '{log_file}' not found.")
            sys.exit(1)
    
    # Parse all log files
    log_files_data = []
    for log_file in log_files:
        print(f"Parsing log file: {log_file}")
        val_times, val_losses = parse_val_losses(log_file)
        label = Path(log_file).stem
        log_files_data.append((label, val_times, val_losses))
        print(f"  Found {len(val_times)} validation entries from the last 1000 steps")
    
    if not any(val_times for _, val_times, _ in log_files_data):
        print("Error: No validation loss entries found in any log file.")
        sys.exit(1)
    
    # Generate output filename
    if len(log_files) == 1:
        output_file = Path(log_files[0]).stem + '_val_losses.png'
    else:
        output_file = 'comparison_val_losses.png'
    
    plot_val_losses(log_files_data, output_file)


if __name__ == '__main__':
    main()
