import numpy as np
import seaborn as sns
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------
# 1. Config: map experiment labels to their console log files
# ---------------------------------------------------------------------
LOG_FILES = {
    "baseline_50_runs_ep9.9": "baseline_50_runs_ep9.9_console.txt",
    "BR_grid_ep9.9":          "BR_grid_ep9.9_console.txt",
    "no_tta_50_runs_ep9.9":   "no_tta_50_runs_ep9.9_console.txt",
}

# If logs are in a subdirectory, set this:
LOG_DIR = Path("cifar10/logs")  # adjust if needed


# ---------------------------------------------------------------------
# 2. Parser for a single console log file
# ---------------------------------------------------------------------
ROW_REGEX = re.compile(
    r"^\|\s*(?P<run>\d+)\s*\|\s*(?P<epoch>\d+|eval)\s*\|\s*"
    r"(?P<train_loss>[0-9.]+)\s*\|\s*(?P<train_acc>[0-9.]+)\s*\|\s*"
    r"(?P<val_acc>[0-9.]+)\s*\|\s*(?P<tta_val_acc>[0-9.]*)\s*\|\s*"
    r"(?P<total_time_seconds>[0-9.]+)\s*\|"
)


def parse_eval_durations(log_path: Path, config_name: str):
    """
    Parse a console log and return a list of dicts:
    {
      'config': config_name,
      'run': int,
      'eval_duration': float,
      'final_total_time': float
    }
    """
    results = []

    with log_path.open("r") as f:
        last_times = {}  # (run -> last_train_epoch_total_time)

        for line in f:
            m = ROW_REGEX.match(line)
            if not m:
                continue

            run = int(m.group("run"))
            epoch = m.group("epoch")
            total_time = float(m.group("total_time_seconds"))

            if epoch == "eval":
                # We assume we saw at least one train epoch for this run
                if run not in last_times:
                    # If somehow eval comes first (shouldn't), skip
                    continue
                eval_duration = total_time - last_times[run]

                results.append({
                    "config": config_name,
                    "run": run,
                    "eval_duration": eval_duration,
                    "final_total_time": total_time,
                })
            else:
                # Numeric epoch -> update last train epoch time
                last_times[run] = total_time

    return results


# ---------------------------------------------------------------------
# 3. Aggregate across all configs into one DataFrame
# ---------------------------------------------------------------------
all_rows = []
for config_name, fname in LOG_FILES.items():
    path = LOG_DIR / fname
    rows = parse_eval_durations(path, config_name)
    all_rows.extend(rows)

df = pd.DataFrame(all_rows)
print(df.head())
print(df.groupby("config").agg(
    mean_eval=("eval_duration", "mean"),
    std_eval=("eval_duration", "std"),
    mean_total=("final_total_time", "mean"),
    std_total=("final_total_time", "std"),
))


# plt.figure(figsize=(10, 6))
# # sns.violinplot(
# #     data=df, x="config", y="eval_duration",
# #     inner=None, cut=0, scale="width"
# # )
# sns.stripplot(
#     data=df, x="config", y="eval_duration", alpha=0.5, jitter=0.2
# )
# plt.ylabel("Eval Duration (seconds)")
# plt.title("Distribution of Eval-Step Runtime Across Configurations")
# plt.tight_layout()
# plt.show()

# ---------------------------------------------------------------------
# 4. Clean up labels and produce a nicer strip plot
# ---------------------------------------------------------------------

# Map raw config names → human-friendly labels
label_map = {
    "baseline_50_runs_ep9.9": "Baseline (TTA=2)",
    "BR_grid_ep9.9": "Bottom-right Only (1 crop)",
    "no_tta_50_runs_ep9.9": "No TTA",
}

df["config_pretty"] = df["config"].map(label_map)

# Define consistent color palette
palette = {
    "Baseline (TTA=2)": sns.color_palette("Set2")[0],
    "Bottom-right Only (1 crop)": sns.color_palette("Set2")[0],
    "No TTA": sns.color_palette("Set2")[0],
}

# Order configs as baseline → BR → no-TTA
order = ["Baseline (TTA=2)", "Bottom-right Only (1 crop)", "No TTA"]

plt.figure(figsize=(10, 6))

sns.stripplot(
    data=df,
    x="config_pretty",
    y="eval_duration",
    order=order,
    palette=palette,
    size=6,        # larger markers
    alpha=0.6,
    jitter=0.25    # increase jitter to separate overlapping points
)

plt.ylabel("Eval Duration (seconds)", fontsize=12)
plt.xlabel("")  # cleaner look
plt.title("Eval-Step Runtime Across Configurations (Per Run)", fontsize=14)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------
# 4. Plot: boxplot of eval durations by config
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 5))
df.boxplot(column="eval_duration", by="config")
plt.ylabel("Eval duration (seconds)")
plt.title("Distribution of eval step time by config")
plt.suptitle("")  # remove automatic suptitle
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# 5. Plot: scatter of eval_duration vs final_total_time
# ---------------------------------------------------------------------
plt.figure(figsize=(7, 5))
for cfg, group in df.groupby("config"):
    plt.scatter(
        group["final_total_time"],
        group["eval_duration"],
        alpha=0.6,
        label=cfg,
    )

plt.xlabel("Final total_time_seconds (per run)")
plt.ylabel("Eval duration (seconds)")
plt.title("Eval time vs total run time by config")
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
for cfg, group in df.groupby("config"):
    sns.ecdfplot(group["eval_duration"], label=cfg)

plt.xlabel("Eval Duration (seconds)")
plt.ylabel("ECDF")
plt.title("ECDF of Eval-Step Runtime")
plt.legend()
plt.tight_layout()
plt.show()


stats = df.groupby("config")["eval_duration"].agg(["mean", "std", "count"])
stats["sem"] = stats["std"] / np.sqrt(stats["count"])
stats["ci95"] = 1.96 * stats["sem"]

plt.figure(figsize=(8, 6))
plt.bar(stats.index, stats["mean"], yerr=stats["ci95"], capsize=5)
plt.ylabel("Mean Eval Duration (seconds)")
plt.title("Mean Eval Time with 95% Confidence Intervals")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="eval_duration",
    y="eval_fraction",
    hue="config",
    alpha=0.6
)
plt.xlabel("Eval Duration (seconds)")
plt.ylabel("Eval Fraction of Total Time")
plt.title("Eval Duration vs Eval Fraction")
plt.tight_layout()
plt.show()
