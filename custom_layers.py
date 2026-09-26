import tensorflow as tf

class JordanRNN(tf.keras.layers.Layer):
    def __init__(self, units, output_units, activation="tanh"):
        super().__init__()

        self.units = units
        self.output_units = output_units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.Wx = self.add_weight(
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            name="Wx"
        )

        self.Wy = self.add_weight(
            shape=(self.output_units, self.units),
            initializer="glorot_uniform",
            name="Wy"
        )

        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            name="b"
        )

        self.Wout = self.add_weight(
            shape=(self.units, self.output_units),
            initializer="glorot_uniform",
            name="Wout"
        )

        self.bout = self.add_weight(
            shape=(self.output_units,),
            initializer="zeros",
            name="bout"
        )

    def call(self, inputs):
        # inputs: (batch, time, features)

        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]

        # Previous output y_{t-1}
        y_prev = tf.zeros(
            (batch_size, self.output_units),
            dtype=inputs.dtype
        )

        outputs = tf.TensorArray(
            dtype=inputs.dtype,
            size=time_steps
        )

        for t in tf.range(time_steps):
            x_t = inputs[:, t, :]

            h = self.activation(
                tf.matmul(x_t, self.Wx)
                + tf.matmul(y_prev, self.Wy)
                + self.b
            )

            y = tf.matmul(h, self.Wout) + self.bout

            outputs = outputs.write(t, y)

            y_prev = y

        # (time, batch, output) -> (batch, time, output)
        return tf.transpose(outputs.stack(), [1, 0, 2])

class MultiRecurrentRNN(tf.keras.layers.Layer):
    def __init__(self, units, delays=(1,), activation="tanh"):
        super().__init__()

        self.units = units
        self.delays = delays
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.Wx = self.add_weight(
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            name="Wx"
        )

        self.Ws = []

        for delay in self.delays:
            self.Ws.append(
                self.add_weight(
                    shape=(self.units, self.units),
                    initializer="orthogonal",
                    name=f"W_delay_{delay}"
                )
            )

        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            name="b"
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]

        max_delay = max(self.delays)

        # Store previous hidden states
        states = tf.TensorArray(
            dtype=inputs.dtype,
            size=time_steps + max_delay
        )

        # Initialize history with zeros
        for i in tf.range(max_delay):
            states = states.write(
                i,
                tf.zeros(
                    (batch_size, self.units),
                    dtype=inputs.dtype
                )
            )

        outputs = tf.TensorArray(
            dtype=inputs.dtype,
            size=time_steps
        )

        for t in tf.range(time_steps):
            x_t = inputs[:, t, :]

            h = tf.matmul(x_t, self.Wx) + self.b

            for W, delay in zip(self.Ws, self.delays):
                previous_h = states.read(
                    max_delay + t - delay
                )

                h += tf.matmul(previous_h, W)

            h = self.activation(h)

            states = states.write(max_delay + t, h)
            outputs = outputs.write(t, h)

        return tf.transpose(outputs.stack(), [1, 0, 2])