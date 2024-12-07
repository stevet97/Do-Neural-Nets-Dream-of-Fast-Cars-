# VAE model definition
class ConvVAE(object):
    def __init__(self, z_size=32, batch_size=1, learning_rate=0.0001, kl_tolerance=0.5):
        self.z_size = z_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.kl_tolerance = kl_tolerance
        self.build_vae_model()

    def build_vae_model(self):
        # Encoder
        self.inputs = tf.keras.Input(shape=(64, 64, 3))
        h = Conv2D(32, 4, strides=2, activation='relu')(self.inputs)
        h = Conv2D(64, 4, strides=2, activation='relu')(h)
        h = Conv2D(128, 4, strides=2, activation='relu')(h)
        h = Conv2D(256, 4, strides=2, activation='relu')(h)
        h = Flatten()(h)
        self.mu = Dense(self.z_size)(h)
        self.logvar = Dense(self.z_size)(h)
        self.z = self.mu + tf.keras.layers.Lambda(lambda x: tf.random.normal(tf.shape(x)) * tf.exp(self.logvar / 2))(self.logvar)

        # Decoder
        h = Dense(1024)(self.z)
        h = Reshape((1, 1, 1024))(h)
        h = Conv2DTranspose(128, 5, strides=2, activation='relu')(h)
        h = Conv2DTranspose(64, 5, strides=2, activation='relu')(h)
        h = Conv2DTranspose(32, 6, strides=2, activation='relu')(h)
        self.outputs = Conv2DTranspose(3, 6, strides=2, activation='sigmoid')(h)

        # Model creation
        self.vae = tf.keras.Model(self.inputs, self.outputs)
        self.vae.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate), loss='mse')



