# Utilities Script

```python
import numpy as np

# Action Derivation Helper
def derive_action_from_latent(z):
    """
    Converts latent representation into environment-specific action.
    
    Args:
        z (np.ndarray): Latent representation vector.

    Returns:
        list: Action values for steering, gas, and brake.
    """
    z = z.flatten()
    steering = np.tanh(z[0])  # Steering is between -1 and 1
    gas = np.clip(z[1], 0, 1)  # Gas is between 0 and 1
    brake = np.clip(z[2], 0, 1)  # Brake is between 0 and 1
    return [steering, gas, brake]


# Video Writer for Gameplay Recording
def setup_video_writer(env, filename='output.mp4'):
    """
    Sets up a video writer for recording environment gameplay.

    Args:
        env: The Gym environment.
        filename (str): Output video file name.

    Returns:
        cv2.VideoWriter: Initialized video writer object.
    """
    height, width, _ = env.render(mode='rgb_array').shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(filename, fourcc, 20.0, (width, height))

