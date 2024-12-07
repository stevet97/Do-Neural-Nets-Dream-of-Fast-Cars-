import gymnasium as gym
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape
import numpy as np
import gradio as gr

# Check library versions
print("Gymnasium version:", gym.__version__)
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("All libraries imported successfully!")

# Define KLRegularizer class
class KLRegularizer(tf.keras.layers.Layer):
    def __init__(self, kl_tolerance, z_size):
        super(KLRegularizer, self).__init__()
        self.kl_tolerance = kl_tolerance
        self.z_size = z_size

    def call(self, inputs):
        mu, logvar = inputs
        kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar))
        return tf.maximum(kl_loss, self.kl_tolerance)

# Define ConvVAE class
class ConvVAE:
    def __init__(self, z_size=32, learning_rate=0.0001, kl_tolerance=0.5):
        self.z_size = z_size
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

        # Latent variables
        self.mu = Dense(self.z_size)(h)
        self.logvar = Dense(self.z_size)(h)
        self.sigma = tf.keras.layers.Lambda(lambda x: tf.exp(x / 2.0))(self.logvar)
        self.epsilon = tf.keras.layers.Lambda(lambda x: tf.random.normal(tf.shape(x)))(self.mu)
        self.z = tf.keras.layers.Add()([self.mu, tf.keras.layers.Multiply()([self.sigma, self.epsilon])])

        # Decoder
        h = Dense(1024)(self.z)
        h = Reshape((1, 1, 1024))(h)
        h = Conv2DTranspose(128, 5, strides=2, activation='relu')(h)
        h = Conv2DTranspose(64, 5, strides=2, activation='relu')(h)
        h = Conv2DTranspose(32, 6, strides=2, activation='relu')(h)
        self.outputs = Conv2DTranspose(3, 6, strides=2, activation='sigmoid')(h)

        # KL Regularization
        KLRegularizer(self.kl_tolerance, self.z_size)([self.mu, self.logvar])

        # VAE model
        self.vae = tf.keras.Model(self.inputs, self.outputs)

    def compile(self):
        self.vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')

# Define MDN loss function
def mdn_loss(y_true, y_pred):
    logmix, mean, logstd = tf.split(y_pred, num_or_size_splits=3, axis=-1)
    y_true = tf.expand_dims(y_true, axis=-1)
    logmix = logmix - tf.reduce_logsumexp(logmix, axis=-1, keepdims=True)
    log_prob = -0.5 * tf.square((y_true - mean) / tf.exp(logstd)) - logstd - np.log(np.sqrt(2.0 * np.pi))
    return -tf.reduce_mean(tf.reduce_logsumexp(logmix + log_prob, axis=-1))

# Define MDNRNN class
class MDNRNN:
    def __init__(self, hps):
        self.hps = hps
        self.build_rnn_model()

    def build_rnn_model(self):
        self.inputs = tf.keras.Input(shape=(self.hps.max_seq_len, self.hps.input_seq_width))
        lstm_layer = LSTM(self.hps.rnn_size, return_sequences=True)(self.inputs)
        self.outputs = Dense(self.hps.output_seq_width * self.hps.num_mixture * 3)(lstm_layer)
        self.rnn = tf.keras.Model(self.inputs, self.outputs)

    def compile(self):
        self.rnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.hps.learning_rate), loss=mdn_loss)

# Define hyperparameters for MDNRNN
class HParams:
    def __init__(self):
        self.max_seq_len = 100
        self.input_seq_width = 160
        self.output_seq_width = 160
        self.rnn_size = 64
        self.num_mixture = 5
        self.learning_rate = 0.001
        self.batch_size = 16

# Instantiate VAE and MDNRNN
vae_model = ConvVAE()
vae_model.compile()

hps = HParams()
rnn_model = MDNRNN(hps)
rnn_model.compile()

# Helper function to derive actions from latent vectors
def derive_action_from_latent(z):
    steering = np.tanh(z[0])  # Steering between -1 and 1
    gas = np.clip(z[1], 0, 1)  # Gas between 0 and 1
    brake = np.clip(z[2], 0, 1)  # Brake between 0 and 1
    return [steering, gas, brake]

# Gradio interface for AI gameplay
def ai_play(latent_vector):
    try:
        env = gym.make('CarRacing-v2', render_mode='rgb_array')
        obs, _ = env.reset()

        # Decode latent vector to derive action
        action = derive_action_from_latent(latent_vector)
        obs, reward, terminated, truncated, _ = env.step(action)
        frame = env.render()

        # Clean up
        env.close()
        return frame, f"Reward: {reward}"
    except Exception as e:
        return None, f"Error: {str(e)}"

# Gradio interface
interface = gr.Interface(
    fn=ai_play,
    inputs=gr.Textbox(label="Enter Latent Vector (comma-separated values)"),
    outputs=[
        gr.Image(shape=(96, 96), label="Environment Frame"),
        gr.Text(label="Reward Output")
    ],
    title="AI Car Racing Simulator with VAE and MDNRNN",
    description="Input latent vectors to control the car. View the AI's actions and reward."
)

# Launch Gradio interface
interface.launch()




