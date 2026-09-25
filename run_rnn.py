import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

layer_types = {
    "rnn": tf.keras.layers.SimpleRNN,
    "dense": tf.keras.layers.Dense,
}

def process_data(filename, target, features, lookback):
	df = pd.read_csv(filename)

	X = df[features].values
	y = df[target].values.reshape(-1, 1)

	n = len(df)

	train_end = int(n * 0.70)
	val_end = int(n * 0.85)

	X_train = X[:train_end]
	X_val   = X[train_end:val_end]
	X_test  = X[val_end:]

	y_train = y[:train_end]
	y_val   = y[train_end:val_end]
	y_test  = y[val_end:]

	X_scaler = MinMaxScaler()
	y_scaler = MinMaxScaler()

	X_train = X_scaler.fit_transform(X_train)
	X_val   = X_scaler.transform(X_val)
	X_test  = X_scaler.transform(X_test)

	y_train = y_scaler.fit_transform(y_train)
	y_val   = y_scaler.transform(y_val)
	y_test  = y_scaler.transform(y_test)

	X_train, y_train = create_sequences(X_train, y_train, lookback)
	X_val, y_val = create_sequences(X_val, y_val, lookback)
	X_test, y_test = create_sequences(X_test, y_test, lookback)

	print("X_train shape:", X_train.shape)
	print("y_train shape:", y_train.shape)

	return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(X_tr, y_tr, X_val, y_val, layers):
	model = tf.keras.Sequential(layers)
	model.compile(optimizer="adam", loss="mse", metrics=["mae"])

	history = model.fit(
		X_tr, y_tr,

		validation_data=(X_val, y_val),

		epochs=50, batch_size=32,

		callbacks=[tf.keras.callbacks.EarlyStopping(
					monitor="val_loss", patience=10,
					restore_best_weights=True)]
		)

	return model

def evaulate_model(X_test, y_test, model):
	return model.evaluate(X_test, y_test) # test_loss, test_mae



def create_sequences(X, y, lookback):
    X_seq = []
    y_seq = []

    for i in range(lookback, len(X)):
        X_seq.append(X[i-lookback:i])
        y_seq.append(y[i])

    return np.array(X_seq), np.array(y_seq)