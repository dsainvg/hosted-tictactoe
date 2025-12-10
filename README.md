# Tic-Tac-Toe with AI

A modern, interactive Tic-Tac-Toe game built with Streamlit and powered by a PyTorch neural network. Challenge the AI in a clean, responsive web interface.

## Features

- **AI Opponent:** Play against a trained neural network that learns from game states.
- **Interactive UI:** Built with Streamlit for a smooth, browser-based experience.
- **Score Tracking:** Keeps track of your wins, the AI's wins, and draws.
- **Responsive Design:** Optimized for both desktop and mobile views.
- **Game Controls:** Easily start a new game or reset all scores.

## Project Structure

- `main.py`: The main entry point for the Streamlit application. Handles the UI and game loop.
- `utils.py`: Contains the core game logic (`Board` class) and the neural network architecture (`NeuralNet` class).
- `model.pth`: The pre-trained PyTorch model weights for the AI.
- `requirements.txt`: List of Python dependencies required to run the project.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To start the game, run the following command in your terminal:

```bash
streamlit run main.py
```

This will launch the application in your default web browser (usually at `http://localhost:8501`).

## How to Play

1. The game starts with you (Player X) vs. the AI (Player O).
2. Click on any empty cell to make your move.
3. The AI will automatically calculate and make its move.
4. The game ends when a player wins or the board is full (draw).
5. Use the **New Game** button to clear the board and play again.
6. Use the **Reset All** button to clear the board and reset the score counters.

## AI Model

The AI uses a feedforward neural network defined in `utils.py`. It takes the current board state as input and outputs the probability distribution for the next best move. The model is loaded from `model.pth`.

- **Input:** 9 board positions (0 for empty, 1 for player, -1 for AI)
- **Architecture:** 4 hidden layers with ReLU activation.
- **Output:** Probabilities for each of the 9 positions.
