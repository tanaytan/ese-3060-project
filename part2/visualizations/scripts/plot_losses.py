#!/usr/bin/env python3
"""
Script to plot train loss and test (val) loss over time from a log file.

Usage:
    python plot_losses.py <log_file>
    python plot_losses.py logs/baseline_1.txt
"""

import re
import sys
import matplotlib.pyplot as plt
from pathlib import Path


def parse_log_file(log_file_path):
    """
    Parse a log file and extract train_time (ms), train_loss, and val_loss values.
    
    Returns:
        train_times: list of train_time values (ms) where train_loss was recorded
        train_losses: list of train_loss values
        val_times: list of train_time values (ms) where val_loss was recorded
        val_losses: list of val_loss values
    """
    train_times = []
    train_losses = []
    val_times = []
    val_losses = []
    
    # Patterns to match log lines - capture step, loss, and train_time
    train_pattern = re.compile(r'step:(\d+)/\d+\s+train_loss:([\d.]+)\s+train_time:([\d.]+)ms')
    val_pattern = re.compile(r'step:(\d+)/\d+\s+val_loss:([\d.]+)\s+train_time:([\d.]+)ms')
    
    # train_time resets at step 10, so we need to track the offset
    # For steps 0-10, times are weird (include initialization). 
    # For steps 11+, train_time is cumulative since step 10 reset.
    time_offset = 0.0
    last_time_before_reset = 0.0
    
    with open(log_file_path, 'r') as f:
        for line in f:
            # Check for train_loss
            train_match = train_pattern.search(line)
            if train_match:
                step = int(train_match.group(1))
                loss = float(train_match.group(2))
                train_time_ms = float(train_match.group(3))
                
                # Handle the reset at step 10
                if step == 10:
                    last_time_before_reset = train_time_ms
                elif step == 11:
                    # After reset, calculate offset needed to make times continuous
                    time_offset = last_time_before_reset
                
                # For steps 11+, train_time is cumulative since step 10
                # Add the offset to make it cumulative from the start
                if step >= 11:
                    cumulative_time = time_offset + train_time_ms
                else:
                    # For steps 0-10, use time as-is (though these may be inaccurate)
                    cumulative_time = train_time_ms
                
                train_times.append(cumulative_time)
                train_losses.append(loss)
            
            # Check for val_loss
            val_match = val_pattern.search(line)
            if val_match:
                step = int(val_match.group(1))
                loss = float(val_match.group(2))
                train_time_ms = float(val_match.group(3))
                
                # Same logic as train_loss
                if step >= 11:
                    cumulative_time = time_offset + train_time_ms
                else:
                    cumulative_time = train_time_ms
                
                val_times.append(cumulative_time)
                val_losses.append(loss)
    
    return train_times, train_losses, val_times, val_losses


def plot_losses(train_times, train_losses, val_times, val_losses, log_file_path):
    """
    Plot train loss and test (val) loss over train time in one graph.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot train loss
    if train_times and train_losses:
        plt.plot(train_times, train_losses, label='Train Loss', alpha=0.7, linewidth=1.5)
    
    # Plot val loss
    if val_times and val_losses:
        plt.plot(val_times, val_losses, label='Test Loss (Val)', alpha=0.7, linewidth=1.5, marker='o', markersize=3)
    
    plt.xlabel('Train Time (ms)', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training and Test Loss Over Time\n{Path(log_file_path).name}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    output_file = Path(log_file_path).stem + '_losses.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Also show the plot
    plt.show()


def main():
    if len(sys.argv) != 2:
        print("Usage: python plot_losses.py <log_file>")
        sys.exit(1)
    
    log_file_path = sys.argv[1]
    
    if not Path(log_file_path).exists():
        print(f"Error: File '{log_file_path}' not found.")
        sys.exit(1)
    
    print(f"Parsing log file: {log_file_path}")
    train_times, train_losses, val_times, val_losses = parse_log_file(log_file_path)
    
    print(f"Found {len(train_times)} train loss entries")
    print(f"Found {len(val_times)} test (val) loss entries")
    
    if not train_times and not val_times:
        print("Error: No loss entries found in the log file.")
        sys.exit(1)
    
    plot_losses(train_times, train_losses, val_times, val_losses, log_file_path)


if __name__ == '__main__':
    main()
