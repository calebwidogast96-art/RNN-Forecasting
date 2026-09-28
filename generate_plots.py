import json
import re
import matplotlib.pyplot as plt
import numpy as np

def generate_plots(filename):
    dataset = filename.split("_")[0]

    with open(filename) as f:
        data = json.load(f)

    models = []

    for r in data:
        if r["lookback"] != 30:
            continue

        match = re.match(r"(.+?)_(\d+)$", r["model"])
        if not match:
            continue

        rnn_type, units = match.groups()

        models.append({
            "name": r["model"],
            "mae": r["test_mae"]
        })

    x = np.arange(len(models))

    plt.figure(figsize=(8, 6))

    plt.bar(
        x,
        [m["mae"] for m in models]
    )

    plt.xlabel("Model")
    plt.ylabel("MAE")
    plt.title(f"{dataset}: MAE for Different RNN Models")

    plt.xticks(
        x,
        [m["name"] for m in models],
        rotation=45,
        ha="right"
    )

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{dataset}_plot.png", dpi=300)
    # plt.show()
# generate_plots("covid_single.json")
generate_plots("electricity_single.json")
# generate_plots("ETTh1_single.json")
# generate_plots("traffic_single.json")
# generate_plots("weather_single.json")


# with open("covid_single.json", "r") as f:
#     results = json.load(f)
# print(f"Covid: {min(results, key=lambda d: d["test_mae"])}")

# with open("electricity_single.json", "r") as f:
#     results = json.load(f)
# print(f"Electricity: {min(results, key=lambda d: d["test_mae"])}")

# with open("ETTh1_single.json", "r") as f:
#     results = json.load(f)
# print(f"ETT: {min(results, key=lambda d: d["test_mae"])}")

# with open("traffic_single.json", "r") as f:
#     results = json.load(f)
# print(f"Traffic: {min(results, key=lambda d: d["test_mae"])}")

# with open("weather_single.json", "r") as f:
#     results = json.load(f)
# print(f"Weather: {min(results, key=lambda d: d["test_mae"])}")