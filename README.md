# Do-Neural-Nets-Dream-of-Fast-Cars-
Simulating AI interaction with the CarRacing-v3 environment using VAE-MDN-RNN

# Do Neural Nets Dream of Fast Cars: AI-Powered Car Racing Simulation

Welcome to **Do Neural Nets Dream of Fast Cars**, an AI project that demonstrates the application of Variational Autoencoders (VAEs) and Mixture Density Network Recurrent Neural Networks (MDN-RNNs) to simulate an AI interacting with the classic CarRacing-v3 environment from Gymnasium. This project is a showcase of AI's ability to model and "dream" about complex environments, pushing the boundaries of reinforcement learning and generative modeling.

---

## **Project Overview**

This project builds on the classic concept of VAEs and MDN-RNNs to create a "world model" capable of:

1. **Learning latent space representations** of visual observations from the environment.
2. **Predicting future states** based on these latent representations.
3. **Interacting with the environment** to optimize performance.

The name **"Do Neural Nets Dream of Fast Cars"** reflects the idea that the AI dreams up future states of its world while navigating through a simulated racing game (Props if you saw the Bladerunner reference)

---

## **Features**

- **VAE:** Encodes the environment's visual input into a compact latent representation and decodes latent vectors back into visual frames.
- **MDN-RNN:** Predicts sequences in the latent space, enabling the model to dream and plan its actions.
- **CarRacing-v3 Simulation:** Uses the upgraded version of the classic CarRacing environment for compatibility and better performance.
- **Streamlit Integration:** Provides an interactive interface for users to explore the AI's decision-making and compare its performance against human players.

---

## **Getting Started**

### **Installation**
To replicate this project, you will need the following dependencies. Run these commands in your Python environment:

```bash
python -m pip install --upgrade pip
pip install gymnasium  
pip install tensorflow  
pip install numpy       
pip install opencv-python  
pip install matplotlib
pip install streamlit
```

### **Environment Setup**
To ensure compatibility:
1. Use `gymnasium` instead of the older `gym` library.
2. The `CarRacing-v3` environment is used instead of deprecated versions.
3. Ensure `pip` is up-to-date to avoid dependency conflicts.

### **Testing the Environment**
Before running the full project, verify the CarRacing-v3 environment:

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Test environment setup
try:
    env = gym.make('CarRacing-v3', render_mode='rgb_array')
    obs, info = env.reset()
    print("Environment initialized successfully with observation shape:", obs.shape)
    env.close()
except Exception as e:
    print(f"Error initializing the environment: {e}")
```

---

## **How It Works**

### **Variational Autoencoder (VAE)**
- **Purpose:** Compresses high-dimensional visual inputs into a compact latent space.
- **Architecture:**
  - Encoder: Extracts latent variables (`mu` and `logvar`).
  - Decoder: Reconstructs frames from the latent space.
- **Loss Function:** Mean Squared Error (MSE) combined with KL Divergence regularization.

### **Mixture Density Network RNN (MDN-RNN)**
- **Purpose:** Predicts sequences in the latent space, modeling transitions between states.
- **Architecture:**
  - LSTM layers predict latent space trajectories.
  - MDN layers handle probabilistic output.
- **Loss Function:** Mixture Density Loss to model probabilistic distributions.

### **AI Interaction**
The AI interacts with the environment using actions predicted from the latent space representation. These actions optimize for cumulative rewards over time.

---

## **Key Challenges and Solutions**

### 1. **Dependency Issues**
   - **Challenge:** Older projects using `gym` had compatibility issues with current dependencies.
   - **Solution:** Transitioned to `gymnasium` and `CarRacing-v3`. This ensured compatibility and reduced deprecation warnings.

### 2. **Environment Rendering**
   - **Challenge:** Rendering issues with `CarRacing` environments.
   - **Solution:** Used `rgb_array` mode for rendering frames in high quality.

### 3. **Streamlit Integration**
   - **Challenge:** Making the project interactive for users.
   - **Solution:** Implemented an interface for comparing AI and human performance, with live video rendering and downloadable gameplay videos.

---

## **Using the Streamlit App**

Run the Streamlit app to visualize the AI's interaction with the environment:

```bash
streamlit run app.py
```

Features:
- **AI vs Human Gameplay:** Compare the AI's decisions and reward outcomes with human inputs.
- **Video Output:** Download AI gameplay videos directly.
- **Interactive Controls:** Observe how the latent space influences actions in real time.

---

## **Future Directions**

- **Real-World Applications:** Expand this model to other environments and games.
- **Enhanced World Models:** Integrate attention mechanisms to improve planning.
- **Multi-Agent Scenarios:** Simulate collaborative or competitive tasks between agents.

---

## **Contributing**

Feel free to fork this repository, raise issues, and contribute to its development. Whether you’re a beginner looking to learn or an expert seeking to enhance this project, all contributions are welcome.

---

## **Contact**

For any queries or suggestions, reach out via [stephenthomas382@gmail.com].

---

### **Acknowledgments**

Inspired by David Ha and Jürgen Schmidhuber's work on "World Models" and the classic "Do Androids Dream of Electric Sheep?" by Philip K. Dick.

---
