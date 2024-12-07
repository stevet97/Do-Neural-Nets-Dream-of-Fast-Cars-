# Do Neural Nets Dream of Fast Cars?

This project demonstrates an AI agent playing the CarRacing-v3 game and allows users to compete against the AI in a simulated environment. The project was developed using **Gradio**, **Gymnasium**, and **Python**, with the AI leveraging simple predefined actions.

---

## Features
- **AI Gameplay:** Watch the AI play the CarRacing-v3 environment live and achieve a score.
- **Human Gameplay (WIP):** Compete against the AI using manual controls.
- **Interactive Interface:** Built using Gradio for an easy-to-use web interface.

---

## Screenshots
### AI Playing
![AI Gameplay Screenshot](assets/ai_play.png)

### Human Playing
![Human Gameplay Screenshot](assets/human_play.png)

---

## Challenges Faced
- Implementing real-time WASD controls for human gameplay in a browser-based interface remains a work-in-progress.
- Rendering live frames during gameplay required creative solutions and debugging.
- Gradio's current limitations for handling live keypress interactions posed technical challenges.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/car-racing-ai.git
   cd car-racing-ai
