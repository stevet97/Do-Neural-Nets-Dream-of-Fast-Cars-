class MDNRNN:
    def __init__(self, hps):
        self.hps = hps
        self.build_rnn_model()

    def build_rnn_model(self):
        # Define LSTM model using keras.layers
        self.inputs = tf.keras.Input(shape=(self.hps.max_seq_len, self.hps.input_seq_width))
        lstm_layer = LSTM(self.hps.rnn_size, return_sequences=True, name='RNN_Layer')(self.inputs)
        self.outputs = Dense(self.hps.output_seq_width * self.hps.num_mixture * 3, name='output')(lstm_layer)
        self.rnn = tf.keras.Model(self.inputs, self.outputs)

    def compile_model(self):
        # Compile the model using the mdn_loss method within the class
        self.rnn.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.hps.learning_rate), 
            loss=self.mdn_loss
        )

    def mdn_loss(self, y_true, y_pred):
        """MDN Loss Function encapsulated within the class."""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Split y_pred into its components (logmix, mean, logstd)
        logmix, mean, logstd = tf.split(y_pred, num_or_size_splits=3, axis=-1)

        # Compute the loss components
        num_mixtures = self.hps.num_mixture
        mean = tf.reshape(mean, (-1, tf.shape(mean)[1], tf.shape(mean)[2] // num_mixtures, num_mixtures))
        logstd = tf.reshape(logstd, (-1, tf.shape(logstd)[1], tf.shape(logstd)[2] // num_mixtures, num_mixtures))

        # Expand y_true to match the number of mixtures
        y_true = tf.expand_dims(y_true, axis=-1)

        logmix = tf.reshape(logmix, (-1, tf.shape(logmix)[1], tf.shape(mean)[2], num_mixtures))
        logmix = logmix - tf.reduce_logsumexp(logmix, axis=-1, keepdims=True)

        logsqrtTwoPI = tf.cast(np.log(np.sqrt(2.0 * np.pi)), tf.float32)
        log_prob = -0.5 * tf.square((y_true - mean) / tf.exp(logstd)) - logstd - logsqrtTwoPI
        v = logmix + log_prob

        # Return negative log-likelihood
        return -tf.reduce_mean(tf.reduce_logsumexp(v, axis=-1))

    def manual_train(self, x_train, y_train, epochs, batch_size):
        """Manual training without tf.data.Dataset."""
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.hps.learning_rate)
        dataset_size = x_train.shape[0]
        steps_per_epoch = dataset_size // batch_size

        for epoch in range(epochs):
            tf.print(f"Epoch {epoch + 1}/{epochs}")

            # Shuffle the data
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            x_train = x_train[indices]
            y_train = y_train[indices]

            for step in range(steps_per_epoch):
                start_idx = step * batch_size
                end_idx = start_idx + batch_size
                x_batch = x_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]

                with tf.GradientTape() as tape:
                    predictions = self.rnn(x_batch, training=True)
                    loss = self.mdn_loss(y_batch, predictions)

                gradients = tape.gradient(loss, self.rnn.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.rnn.trainable_variables))

                if step % 10 == 0:
                    tf.print(f"Step {step}, Loss: ", loss)

