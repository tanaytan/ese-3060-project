#!/usr/bin/env python3
"""
Script to plot validation loss over time for GQA 250 iteration log files,
and then generate a statistics table.

Usage:
    python plot_gqa_250_val_losses.py
"""

import re
import sys
import matplotlib.pyplot as plt
from pathlib import Path


def parse_val_losses(log_file_path):
    """
    Parse a log file and extract step numbers, cumulative train_time (ms), and val_loss values.
    
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
    
    # Sort by step
    all_val_entries.sort(key=lambda x: x[0])
    
    return all_val_entries


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


def plot_val_losses(log_files_data, output_file):
    """
    Plot validation loss over time for multiple log files on one graph.
    
    Args:
        log_files_data: list of tuples (label, val_times, val_losses)
        output_file: path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'v', 'D', 'p']
    
    for i, (label, val_times, val_losses) in enumerate(log_files_data):
        if val_times and val_losses:
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            plt.plot(val_times, val_losses, label=label, alpha=0.7, 
                    linewidth=1.5, marker=marker, markersize=5, color=color)
    
    plt.xlabel('Train Time (ms)', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('Validation Loss Over Time (GQA 250 Iterations)', fontsize=14)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Also show the plot
    plt.show()


def create_table(log_files):
    """Create and print a statistics table for the log files."""
    table_data = []
    
    for name, log_file in log_files:
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
    print("\n" + "=" * 60)
    print("Training Statistics Table")
    print("=" * 60)
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
    output_file = 'gqa_250_training_stats.txt'
    with open(output_file, 'w') as f:
        f.write("Training Statistics Table\n")
        f.write("=" * 60 + "\n")
        f.write(separator + '\n')
        f.write(header_row + '\n')
        f.write(separator + '\n')
        for row in table_data:
            data_row = "| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + " |"
            f.write(data_row + '\n')
        f.write(separator + '\n')
    print(f"\nTable saved to: {output_file}")


def main():
    logs_dir = Path('logs')
    
    # Define files - note: one has double .txt extension
    file_names = [
        'gqa_plain_250_iterations.txt',
        'gqa_most_sharing_moderate_layers_250_iterations.txt',
        'gqa_most_sharing_250_iterations.txt',
        'gqa_most_layers_250_iterations.txt',
        'gqa_moderate_sharing_250_iterations.txt.txt',  # Note: double .txt
        'gqa_moderate_layers_250_iterations.txt'
    ]
    
    log_files = [(Path(f).stem.replace('.txt', ''), logs_dir / f) for f in file_names]
    
    # Check which files exist
    existing_files = []
    for name, log_file in log_files:
        if log_file.exists():
            existing_files.append((name, log_file))
        else:
            # Try without double .txt
            alt_file = logs_dir / name.replace('_250_iterations', '_250_iterations.txt')
            if alt_file.exists():
                existing_files.append((name, alt_file))
            else:
                print(f"Warning: File '{log_file}' not found, skipping...")
    
    if not existing_files:
        print("Error: No log files found.")
        sys.exit(1)
    
    print("Plotting validation losses...")
    print("=" * 60)
    
    # Parse all log files for plotting
    log_files_data = []
    for name, log_file in existing_files:
        print(f"Parsing {log_file}...")
        val_data = parse_val_losses(log_file)
        val_times = [time for _, time, _ in val_data]
        val_losses = [loss for _, _, loss in val_data]
        log_files_data.append((name, val_times, val_losses))
        print(f"  Found {len(val_data)} validation entries")
    
    # Plot
    output_file = 'gqa_250_val_losses.png'
    plot_val_losses(log_files_data, output_file)
    
    # Create table
    create_table(existing_files)


if __name__ == '__main__':
    main()
