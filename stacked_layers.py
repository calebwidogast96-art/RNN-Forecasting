import tensorflow as tf

def make_simple_rnn_stacked(lookback, input_size, units)->dict:
    rnn_layers = {

    "simple_rnn_1_layers": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        tf.keras.layers.SimpleRNN(units),
        tf.keras.layers.Dense(1)
    ],

    "simple_rnn_2_layers": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        tf.keras.layers.SimpleRNN(units),
        tf.keras.layers.SimpleRNN(units),
        tf.keras.layers.Dense(1)
    ],

    "simple_rnn_3_layers": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        tf.keras.layers.SimpleRNN(units),
        tf.keras.layers.SimpleRNN(units),
        tf.keras.layers.SimpleRNN(units),
        tf.keras.layers.Dense(1)
    ],

    "simple_rnn_4_layers": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        tf.keras.layers.SimpleRNN(units),
        tf.keras.layers.SimpleRNN(units),
        tf.keras.layers.SimpleRNN(units),
        tf.keras.layers.SimpleRNN(units),
        tf.keras.layers.Dense(1)
    ],
    }

    return rnn_layers

def make_gru_stacked(lookback, input_size, units)->dict:
    rnn_layers = {

    "gru_1_layers": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        tf.keras.layers.GRU(units),
        tf.keras.layers.Dense(1)
    ],
    
    "gru_2_layers": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        tf.keras.layers.GRU(units),
        tf.keras.layers.GRU(units),
        tf.keras.layers.Dense(1)
    ],

    "gru_3_layers": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        tf.keras.layers.GRU(units),
        tf.keras.layers.GRU(units),
        tf.keras.layers.GRU(units),
        tf.keras.layers.Dense(1)
    ],

    "gru_4_layers": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        tf.keras.layers.GRU(units),
        tf.keras.layers.GRU(units),
        tf.keras.layers.GRU(units),
        tf.keras.layers.GRU(units),
        tf.keras.layers.Dense(1)
    ],
    }

    return rnn_layers


def make_lstm_stacked(lookback, input_size, units)->dict:
    rnn_layers = {

    "lstm_1_layers": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        tf.keras.layers.LSTM(units),
        tf.keras.layers.Dense(1)
    ],

    "lstm_2_layers": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        tf.keras.layers.LSTM(units),
        tf.keras.layers.LSTM(units),
        tf.keras.layers.Dense(1)
    ],

    "lstm_3_layers": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        tf.keras.layers.LSTM(units),
        tf.keras.layers.LSTM(units),
        tf.keras.layers.LSTM(units),
        tf.keras.layers.Dense(1)
    ],

    "lstm_4_layers": [
        tf.keras.layers.Input(shape=(lookback, input_size)),
        tf.keras.layers.LSTM(units),
        tf.keras.layers.LSTM(units),
        tf.keras.layers.LSTM(units),
        tf.keras.layers.Dense(1)
    ]}

    return rnn_layers