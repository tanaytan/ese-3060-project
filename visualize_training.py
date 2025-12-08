import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Helper: parse one console log into a DataFrame
# -------------------------------------------------------------------


def parse_console_log(path: str) -> pd.DataFrame:
    """
    Parse an airbench94-style console log into a DataFrame with columns:
    run, epoch, train_loss, train_acc, val_acc, tta_val_acc, total_time_seconds.
    """
    rows = []

    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")

            # Data rows start with '|' but are not pure separator lines
            if not line.startswith("|"):
                continue
            if set(line.replace("|", "").strip()) <= {"-"}:
                continue

            parts = [p.strip() for p in line.split("|")[1:-1]]

            # Skip header row
            if not parts or parts[0] == "run":
                continue

            # Expect 7 columns:
            # run, epoch, train_loss, train_acc, val_acc, tta_val_acc, total_time_seconds
            if len(parts) != 7:
                continue

            run_str, epoch_str, train_loss_str, train_acc_str, val_acc_str, tta_val_acc_str, time_str = parts

            # Skip the eval row for training curves (we only want numeric epochs)
            if epoch_str == "eval":
                continue

            try:
                row = {
                    "run": int(run_str),
                    "epoch": int(epoch_str),
                    "train_loss": float(train_loss_str),
                    "train_acc": float(train_acc_str),
                    "val_acc": float(val_acc_str),
                    "tta_val_acc": float(tta_val_acc_str) if tta_val_acc_str else None,
                    "total_time_seconds": float(time_str),
                }
                rows.append(row)
            except ValueError:
                # Skip any malformed line
                continue

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# File paths for the three configurations
# -------------------------------------------------------------------
log_files = {
    "Baseline (TTA 2, 3 crops)": "cifar10/logs/baseline_50_runs_ep9.9_console.txt",
    "Bottom-right only TTA":     "cifar10/logs/BR_grid_ep9.9_console.txt",
    "No TTA":                    "cifar10/logs/no_tta_50_runs_ep9.9_console.txt",
}

# -------------------------------------------------------------------
# Build mean validation-accuracy curve for each config
# -------------------------------------------------------------------
plt.figure(figsize=(7, 5))

for label, path in log_files.items():
    df = parse_console_log(path)
    run0 = df[df["run"] == 0].sort_values(by="epoch")
    plt.plot(run0["epoch"], run0["train_loss"],
             marker="o", label=label + " (run 0)")

# for label, path in log_files.items():
#     df = parse_console_log(path)

#     if df.empty:
#         print(f"Warning: no data parsed from {path}")
#         continue

#     # Average val_acc over runs for each epoch
#     curve = (
#         df.groupby("epoch", as_index=False)["train_loss"]
#         .mean()
#         .sort_values(by="epoch")
#     )

#     plt.plot(
#         curve["epoch"],
#         curve["train_loss"],
#         marker="o",
#         label=label,
#     )

plt.xlabel("Epoch")
plt.ylabel("Validation accuracy")
plt.title("CIFAR-10 validation accuracy vs epoch (mean over runs)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()

plt.savefig("training_curves_val_acc_all_configs.png", dpi=300)
plt.show()
