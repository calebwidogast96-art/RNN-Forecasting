import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")

# Suppress TensorFlow's native stderr during import
stderr = os.dup(2)
devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, 2)

import tensorflow as tf
from run_rnn import run_pipeline

model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(24, 400)
    ),

    tf.keras.layers.SimpleRNN(128),

    tf.keras.layers.Dense(32, activation="relu"),

    tf.keras.layers.Dense(1)
])


os.dup2(stderr, 2)
os.close(devnull)
os.close(stderr)