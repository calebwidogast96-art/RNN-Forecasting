from single_layer import make_rnn_layers
from run_rnn import run_pipeline
import json
import sys
from pathlib import Path

stacked_rnns = {"gru": make_gru_stacked,
                "lstm": make_lstm_stacked,
                "simple": make_simple_rnn_stacked}

def run_experiment(filename, target, features, lb_min, lb_max, lb_step)->dict: 
    input_size = len(features)
    results = []

    json_file = filename.replace(".csv", "_single.json")
    if Path(json_file).exists():
        with open(json_file, "r") as f:
            results = json.load(f)

    count = len(results)
    for lb in range(lb_min, lb_max, lb_step):
        layers = make_rnn_layers(lb, input_size)

        for model, layers in layers.items():
            count -= 1
            if count >= 0: continue

            print(f"\n\nRunning {model} with {lb} lookback...")
            test_loss, test_mae = run_pipeline(f"data/{filename}", target, features, lb, layers)
            
            results.append({"model":model, "lookback":lb, "test_loss":test_loss, "test_mae":test_mae})

            print(f"\nResults: model:{model}, lookback:{lb}, test_loss:{test_loss}, test_mae:{test_mae}")

            with open(json_file, "w") as f:
                json.dump(results, f, indent=4)

    return min(results, key=lambda d: d["test_mae"])

def main():
    if len(sys.argv) != 2:
        print("Usage: python program.py <arg>")
        sys.exit(1)

    filename = str(sys.argv[1])

    if filename == "covid.csv":
        target = "55"
        features = [str(i) for i in range(1, 55)]
        lb_min = 1
        lb_step = 1
        lb_max = 21 + lb_step
    elif filename == "electricity.csv":
        target = "OT"
        features = [str(i) for i in range(320)]
        lb_min = 6
        lb_step = 6
        lb_max = 48 + lb_step
    elif filename == "ETTh1.csv":
        target = "OT"
        features = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
        lb_min = 6
        lb_step = 6
        lb_max = 60 + lb_step
    elif filename == "traffic.csv":
        target = "OT"
        features = [str(i) for i in range(430)]
        lb_min = 12
        lb_step = 12
        lb_max = 48 + lb_step
    elif filename == "weather.csv":
        target = "OT"
        features = ["p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)",
            "rh (%)", "VPmax (mbar)", "VPact (mbar)", "VPdef (mbar)",
            "sh (g/kg)", "H2OC (mmol/mol)", "rho (g/m**3)","wv (m/s)",
            "max. wv (m/s)", "wd (deg)", "rain (mm)", "raining (s)",
            "SWDR (W/m�)", "PAR (�mol/m�/s)", "max. PAR (�mol/m�/s)",
            "Tlog (degC)"]
       
        lb_min = 12 # 2 hour
        lb_step = 12
        lb_max = 36 + lb_step
    else: 
        print("Unsuported file")
        return

    if not Path(f"data/{filename}").exists():
        print("File doesn't exist")
        return

    best_single = run_experiment(filename, target, features,
                          lb_min, lb_max, lb_step)

    print(best_single)

if "__main__" == __name__:
    main()