from single_layer import make_rnn_layers
from stacked_layers import make_gru_stacked, make_lstm_stacked, make_simple_rnn_stacked
from run_rnn import run_pipeline
import json
import sys
from pathlib import Path

stacked_rnns = {"gru": make_gru_stacked,
                "lstm": make_lstm_stacked,
                "simple": make_simple_rnn_stacked}

def run_experiment(filename, target, features, lb_min, lb_max, lb_step)->dict: 
    input_size = len(features)
    results = {}
    for lb in range(lb_min, lb_max, lb_step):
        layers = make_rnn_layers(lb, input_size)

        for model, layers in layers.items():
            test_loss, test_mae = run_pipeline(f"data/{filename}", target, features, lb, layers)
            results.append({"model":model, "lookback":lb, "test_loss":test_loss, "test_mae":test_mae})

    print(f"Results:{results}")

    with open({filename.replace(".csv", "_single.json")}, "w") as f:
        json.dump(results, f, indent=4)

    return min(results, key=lambda d: d["test_mae"])

def run_stacked_experiment(filename, target, features, lb, type, units)->dict: 
    input_size = len(features)
    results = {}

    layers = stacked_rnns[type](lb, input_size, units)

    for model, layers in layers.items():
        test_loss, test_mae = run_pipeline(f"data/{filename}", target, features, lb, layers)
        results.append({"model":model, "lookback":lb, "units":units, "test_loss":test_loss, "test_mae":test_mae})

    print(f"Stacked results:{results}")

    with open({filename.replace(".csv", "_stacked.json")}, "w") as f:
        json.dump(results, f, indent=4)

    return min(results, key=lambda d: d["test_mae"])

def main():
    if len(sys.argv) != 2:
        print("Usage: python program.py <arg>")
        sys.exit(1)

    filename = sys.argv[1]

    if filename == "covid.csv":
        pass
    elif filename == "electricity.csv":
        pass
    elif filename == "ETTm1.csv":
        pass
    elif filename == "traffic.csv":
        target = "OT"
        features = [str(i) for i in range(431)]
        lb_min = 12
        lb_step = 12
        lb_max = 168 + lb_step
    elif filename == "weather.csv":
        pass
    else: 
        print("Unsuported file")
        return

    if not Path(filename).exists():
        print("File doesn't exist")
        return

    best_single = run_experiment(filename, target, features,
                          lb_min, lb_max, lb_step)

    info = best_single["model"].split("_")
    best_type = info[0]

    if best_type in ["jordan", "multi"]:
        print(f"Best RNN: {best_type}")
        return
    else:
        print(f"Best single RNN: {best_type}, investigating stacked...")

    best_lb = best_single["lookback"]
    best_units = int(info[-1])

    best_stacked = run_stacked_experiment(filename, target, features,
                                          best_lb, best_type, best_units)

    print(f"Best stacked RNN: {best_stacked}")

if "__main__" == __name__:
    main()