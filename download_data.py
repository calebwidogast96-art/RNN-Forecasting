import kagglehub

# Download latest version
path = kagglehub.dataset_download("giochelavaipiatti/time-series-forecasts-popular-benchmark-datasets")

print("Path to dataset files:", path)