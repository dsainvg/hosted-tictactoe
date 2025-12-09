import streamlit as st
import torch
import warnings
from utils import Board, load_model

# Suppress torch warnings
warnings.filterwarnings('ignore')

# Initialize session state
if 'board' not in st.session_state:
    st.session_state.board = Board()
    st.session_state.model = load_model()
    st.session_state.game_over = False
    st.session_state.result_message = ""
    st.session_state.ai_turn = False

# Page configuration
st.set_page_config(page_title="Tic-Tac-Toe AI", page_icon="🎮", layout="centered")
st.title("🎮 Tic-Tac-Toe vs AI")
st.markdown("**You are X (1)** | **AI is O (-1)**")

# Function to get AI move
def get_ai_move():
    board_state = st.session_state.board.board_state.unsqueeze(0)
    with torch.no_grad():
        probabilities = st.session_state.model(board_state)
    move = torch.argmax(probabilities).item()
    return move

# Function to handle player move
def player_move(position):
    if not st.session_state.game_over and not st.session_state.ai_turn:
        game_over, result = st.session_state.board.play(1, position)
        
        if result == "invalid":
            st.warning("Invalid move! Cell already occupied.")
            return
        
        if game_over:
            st.session_state.game_over = True
            if result == "win":
                st.session_state.result_message = "🎉 You Win!"
            elif result == "draw":
                st.session_state.result_message = "🤝 It's a Draw!"
        else:
            # AI's turn
            st.session_state.ai_turn = True
            st.rerun()

# Function to handle AI move
def ai_move():
    if not st.session_state.game_over and st.session_state.ai_turn:
        move = get_ai_move()
        game_over, result = st.session_state.board.play(-1, move)
        
        if game_over:
            st.session_state.game_over = True
            if result == "win":
                st.session_state.result_message = "🤖 AI Wins!"
            elif result == "draw":
                st.session_state.result_message = "🤝 It's a Draw!"
        
        st.session_state.ai_turn = False

# Execute AI move if it's AI's turn
if st.session_state.ai_turn:
    ai_move()
    st.rerun()

# Display game board
board_state = st.session_state.board.board_state

# Create 3x3 grid
cols = st.columns(3)
for i in range(3):
    for j in range(3):
        position = i * 3 + j
        cell_value = board_state[position].item()
        
        # Determine button label
        if cell_value == 1:
            label = "❌"
            disabled = True
        elif cell_value == -1:
            label = "⭕"
            disabled = True
        else:
            label = "⬜"
            disabled = st.session_state.game_over or st.session_state.ai_turn
        
        # Create button
        with cols[j]:
            if st.button(label, key=f"cell_{position}", disabled=disabled, use_container_width=True):
                player_move(position)
                st.rerun()

# Display result message
if st.session_state.result_message:
    st.markdown(f"### {st.session_state.result_message}")

# Reset button
if st.button("🔄 New Game", use_container_width=True):
    st.session_state.board.clear()
    st.session_state.game_over = False
    st.session_state.result_message = ""
    st.session_state.ai_turn = False
    st.rerun()

# Display game status
if not st.session_state.game_over:
    if st.session_state.ai_turn:
        st.info("🤖 AI is thinking...")
    else:
        st.info("🎯 Your turn! Click a cell to place X")
