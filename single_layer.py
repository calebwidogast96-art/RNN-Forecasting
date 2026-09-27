import tensorflow as tf
from custom_layers import JordanRNN, MultiRecurrentRNN

def make_rnn_layers(lookback, input_size)->dict:
    rnn_layers = {

    # =========================
    # Simple / Elman RNN
    # =========================

    "simple_rnn_8": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        tf.keras.layers.SimpleRNN(8),
        tf.keras.layers.Dense(1)
    ],

    "simple_rnn_16": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        tf.keras.layers.SimpleRNN(16),
        tf.keras.layers.Dense(1)
    ],

    "simple_rnn_32": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        tf.keras.layers.SimpleRNN(32),
        tf.keras.layers.Dense(1)
    ],

    "simple_rnn_64": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        tf.keras.layers.SimpleRNN(64),
        tf.keras.layers.Dense(1)
    ],

    "simple_rnn_128": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        tf.keras.layers.SimpleRNN(128),
        tf.keras.layers.Dense(1)
    ],

    "simple_rnn_256": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        tf.keras.layers.SimpleRNN(256),
        tf.keras.layers.Dense(1)
    ],

    "simple_rnn_512": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        tf.keras.layers.SimpleRNN(512),
        tf.keras.layers.Dense(1)
    ],


    # =========================
    # Jordan RNN
    # =========================

    "jordan_rnn_8": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        JordanRNN(8, output_units=1),
        tf.keras.layers.Dense(1)
    ],

    "jordan_rnn_16": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        JordanRNN(16, output_units=1),
        tf.keras.layers.Dense(1)
    ],

    "jordan_rnn_32": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        JordanRNN(32, output_units=1),
        tf.keras.layers.Dense(1)
    ],

    "jordan_rnn_64": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        JordanRNN(64, output_units=1),
        tf.keras.layers.Dense(1)
    ],

    "jordan_rnn_128": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        JordanRNN(128, output_units=1),
        tf.keras.layers.Dense(1)
    ],

    "jordan_rnn_256": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        JordanRNN(256, output_units=1),
        tf.keras.layers.Dense(1)
    ],


    # =========================
    # Multi-Recurrent RNN
    # =========================

    "multi_rnn_delay_1": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        MultiRecurrentRNN(64, delays=(1,)),
        tf.keras.layers.Dense(1)
    ],

    "multi_rnn_delay_1_2": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        MultiRecurrentRNN(64, delays=(1, 2)),
        tf.keras.layers.Dense(1)
    ],

    "multi_rnn_delay_1_2_3": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        MultiRecurrentRNN(64, delays=(1, 2, 3)),
        tf.keras.layers.Dense(1)
    ],

    "multi_rnn_delay_1_2_3_4": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        MultiRecurrentRNN(64, delays=(1, 2, 3, 4)),
        tf.keras.layers.Dense(1)
    ],

    "multi_rnn_delay_1_6_12": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        MultiRecurrentRNN(64, delays=(1, 6, 12)),
        tf.keras.layers.Dense(1)
    ],

    "multi_rnn_delay_1_3_6_12": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        MultiRecurrentRNN(64, delays=(1, 3, 6, 12)),
        tf.keras.layers.Dense(1)
    ],

    "multi_rnn_delay_1_6_12_24": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        MultiRecurrentRNN(64, delays=(1, 6, 12, 24)),
        tf.keras.layers.Dense(1)
    ],


    # =========================
    # GRU
    # =========================

    "gru_16": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        tf.keras.layers.GRU(16),
        tf.keras.layers.Dense(1)
    ],

    "gru_32": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        tf.keras.layers.GRU(32),
        tf.keras.layers.Dense(1)
    ],

    "gru_64": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        tf.keras.layers.GRU(64),
        tf.keras.layers.Dense(1)
    ],

    "gru_128": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        tf.keras.layers.GRU(128),
        tf.keras.layers.Dense(1)
    ],

    "gru_256": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        tf.keras.layers.GRU(256),
        tf.keras.layers.Dense(1)
    ],


    # =========================
    # LSTM
    # =========================

    "lstm_16": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dense(1)
    ],

    "lstm_32": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ],

    "lstm_64": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(1)
    ],

    "lstm_128": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(1)
    ],

    "lstm_256": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        tf.keras.layers.LSTM(256),
        tf.keras.layers.Dense(1)
    ],
    }

    return rnn_layers