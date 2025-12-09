#!/usr/bin/env python3
"""
Script to extract and tabulate training statistics from log files.

Extracts:
- Overall train time (ms)
- Step average (ms)
- Final validation loss

Usage:
    python extract_training_stats.py
"""

import re
import sys
from pathlib import Path


def extract_stats(log_file_path):
    """
    Extract training statistics from a log file.
    
    Returns:
        tuple: (train_time_ms, step_avg_ms, final_val_loss) or None if not found
    """
    if not Path(log_file_path).exists():
        return None
    
    # Patterns to match log lines - capture step number too
    train_pattern = re.compile(r'step:(\d+)/\d+\s+train_loss:[\d.]+\s+train_time:([\d.]+)ms\s+step_avg:([\d.]+)ms')
    val_pattern = re.compile(r'step:(\d+)/\d+\s+val_loss:([\d.]+)\s+train_time:([\d.]+)ms\s+step_avg:([\d.]+)ms')
    
    # Track time offset for cumulative time calculation
    time_offset = 0.0
    last_time_before_reset = 0.0
    
    last_train_time = None
    last_step_avg = None
    final_val_loss = None
    final_train_time = None
    final_step_avg = None
    final_step = None
    
    with open(log_file_path, 'r') as f:
        for line in f:
            # Check for train_loss to track time offset and last values
            train_match = train_pattern.search(line)
            if train_match:
                step = int(train_match.group(1))
                train_time_ms = float(train_match.group(2))
                step_avg = float(train_match.group(3))
                
                # Handle the reset at step 10
                if step == 10:
                    last_time_before_reset = train_time_ms
                elif step == 11:
                    time_offset = last_time_before_reset
                
                last_train_time = train_time_ms
                last_step_avg = step_avg
            
            # Check for val_loss - we want the last one
            val_match = val_pattern.search(line)
            if val_match:
                step = int(val_match.group(1))
                loss = float(val_match.group(2))
                train_time_ms = float(val_match.group(3))
                step_avg = float(val_match.group(4))
                
                # Calculate cumulative time if needed
                if step >= 11:
                    cumulative_time = time_offset + train_time_ms
                else:
                    cumulative_time = train_time_ms
                
                final_val_loss = loss
                final_train_time = cumulative_time
                final_step_avg = step_avg
                final_step = step
    
    # Use final validation values if available, otherwise use last train values
    if final_val_loss is not None:
        return (final_train_time, final_step_avg, final_val_loss)
    elif last_train_time is not None:
        # For train-only entries, also calculate cumulative time
        cumulative_train_time = time_offset + last_train_time if time_offset > 0 else last_train_time
        return (cumulative_train_time, last_step_avg, None)
    else:
        return None


def format_time(ms):
    """Format milliseconds to a readable string."""
    if ms is None:
        return "N/A"
    if ms < 1000:
        return f"{ms:.0f}ms"
    elif ms < 60000:
        return f"{ms/1000:.2f}s"
    else:
        minutes = int(ms // 60000)
        seconds = (ms % 60000) / 1000
        return f"{minutes}m {seconds:.2f}s"


def main():
    logs_dir = Path('logs')
    
    # Define files to process
    files = [
        ('baseline_1', logs_dir / 'baseline_1.txt'),
        ('baseline_2', logs_dir / 'baseline_2.txt'),
        ('baseline_3', logs_dir / 'baseline_3.txt'),
        ('gqa_1', logs_dir / 'gqa_1.txt'),
        ('gqa_2', logs_dir / 'gqa_2.txt'),
        ('gqa_3', logs_dir / 'gqa_3.txt'),
        ('baseline_mdha_gqa_1', logs_dir / 'baseline_mdha_gqa_1.txt'),
    ]
    
    # Extract stats for each file
    table_data = []
    for name, log_file in files:
        stats = extract_stats(log_file)
        if stats is None:
            print(f"Warning: Could not extract stats from {log_file}", file=sys.stderr)
            table_data.append([name, "N/A", "N/A", "N/A"])
        else:
            train_time_ms, step_avg_ms, final_val_loss = stats
            table_data.append([
                name,
                format_time(train_time_ms),
                f"{step_avg_ms:.2f}ms",
                f"{final_val_loss:.4f}" if final_val_loss is not None else "N/A"
            ])
    
    # Create table
    headers = ["Log File", "Overall Train Time", "Step Avg", "Final Val Loss"]
    
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in table_data:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Create separator line
    separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    
    # Print table
    print(separator)
    # Print header
    header_row = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    print(header_row)
    print(separator)
    # Print data rows
    for row in table_data:
        data_row = "| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + " |"
        print(data_row)
    print(separator)
    
    # Also save to file
    output_file = 'training_stats.txt'
    with open(output_file, 'w') as f:
        f.write(separator + '\n')
        f.write(header_row + '\n')
        f.write(separator + '\n')
        for row in table_data:
            data_row = "| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + " |"
            f.write(data_row + '\n')
        f.write(separator + '\n')
    print(f"\nTable saved to: {output_file}")


if __name__ == '__main__':
    main()
