# Main Script: Setting up and running the environment

```python
# Importing the necessary libraries
import gymnasium as gym
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import cv2
import matplotlib.pyplot as plt
import streamlit as st

print("All libraries imported successfully!")

print("Gymnasium version:", gym.__version__)
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("OpenCV version:", cv2.__version__)

# Test environment
try:
    env = gym.make('CarRacing-v3', render_mode='rgb_array')
    obs, info = env.reset()
    print("Environment initialized successfully with observation shape:", obs.shape)
    env.close()
except Exception as e:
    print(f"Error initializing the environment: {e}")

# Example of deriving an action and stepping through the environment
action = np.array([0.0, 1.0, 0.0])  # Example action: no steering, full gas, no brake
obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated

# Render the environment visually
frame = env.render()
plt.imshow(frame)
plt.axis('off')
from IPython.display import display
display(plt.gcf())

env.close()

