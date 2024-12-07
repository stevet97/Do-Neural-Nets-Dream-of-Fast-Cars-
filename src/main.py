#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install gymnasium')
get_ipython().system('pip install tensorflow')
get_ipython().system('pip install numpy')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install gradio')
get_ipython().system('pip install keyboard')


# In[2]:


# Importing the necessary libraries
import gymnasium as gym
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
print("All libraries imported successfully!")

print("Gymnasium version:", gym.__version__)
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)


# In[3]:


try:
    env = gym.make('CarRacing-v3', render_mode='rgb_array')
    obs, info = env.reset()
    print("Environment initialized successfully with observation shape:", obs.shape)
    env.close()
except Exception as e:
    print(f"Error initializing the environment: {e}")





