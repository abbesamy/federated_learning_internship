from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "experiments_output"
CSV = OUTPUT_DIR / "experiments_results.csv"

if not CSV.exists():
    raise SystemExit(f"CSV not found: {CSV}")

df = pd.read_csv(CSV)

for (aggregator, attack), group in df.groupby(["aggregator", "attack"]):
    plt.figure()
    for nb_byz, sub in group.groupby("nb_byz"):
        plt.plot(sub["round"], sub["accuracy"], marker="o", label=f"{nb_byz} byz")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title(f"{aggregator} - {attack}")
    plt.legend()
    out_file = OUTPUT_DIR / f"plot_{aggregator}_{attack}.png"
    plt.savefig(out_file)
    plt.close()

print(f"Plots generated in {OUTPUT_DIR}")
