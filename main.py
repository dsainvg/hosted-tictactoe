import streamlit as st
import torch
import warnings
from utils import Board, load_model

warnings.filterwarnings('ignore')

# Initialize session state
if 'board' not in st.session_state:
    st.session_state.board = Board()
    st.session_state.model = load_model()
    st.session_state.game_over = False
    st.session_state.result_message = ""
    st.session_state.ai_turn = False
    st.session_state.player_wins = 0
    st.session_state.ai_wins = 0
    st.session_state.draws = 0

st.set_page_config(page_title="Tic-Tac-Toe", page_icon="🎮", layout="centered")

# Simple CSS - no scroll
st.markdown("""
<style>
    html, body { overflow: hidden !important; height: 100vh; }
    [data-testid="stAppViewContainer"] { 
        background: #1a1a2e;
        height: 100vh;
        overflow: hidden !important;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    [data-testid="stHeader"], footer { display: none; }
    [data-testid="stVerticalBlock"] { height: auto; overflow: visible !important; }
    .main .block-container {
        max-width: 340px;
        padding: 0.4rem !important;
        margin: 0 auto !important;
        height: auto;
    }
    .score { text-align: center; color: #888; font-size: 0.75rem; margin: 0.2rem 0; }
    .score span { color: #00d4ff; font-weight: bold; }
    .status { text-align: center; color: #00d4ff; font-size: 0.9rem; padding: 0.3rem; margin: 0.3rem 0; }
    div.stButton > button {
        width: 100%; height: 60px; font-size: 1.8rem; font-weight: bold;
        background: #252545; border: 2px solid #00d4ff; color: #fff; border-radius: 6px;
        padding: 0 !important; margin: 1px !important;
    }
    div.stButton > button:hover { background: #303060; }
    div.stButton > button:disabled { background: #252545; color: #fff; opacity: 1; }
    [data-testid="column"] { padding: 1px !important; }
    .btn-row button { height: 32px !important; font-size: 0.75rem !important; }
</style>
""", unsafe_allow_html=True)

st.markdown(f'<div class="score">❌ You: <span>{st.session_state.player_wins}</span> · 🤝 <span>{st.session_state.draws}</span> · ⭕ AI: <span>{st.session_state.ai_wins}</span></div>', unsafe_allow_html=True)

# Functions
def get_ai_move():
    board_state = st.session_state.board.board_state.unsqueeze(0)
    with torch.no_grad():
        probs = st.session_state.model(board_state)
    return torch.argmax(probs).item()

def player_move(pos):
    if st.session_state.game_over or st.session_state.ai_turn:
        return
    game_over, result = st.session_state.board.play(1, pos)
    if result == "invalid":
        return
    if game_over:
        st.session_state.game_over = True
        if result == "win":
            st.session_state.result_message = "You Win!"
            st.session_state.player_wins += 1
        else:
            st.session_state.result_message = "Draw!"
            st.session_state.draws += 1
    else:
        st.session_state.ai_turn = True

def ai_move():
    if st.session_state.game_over or not st.session_state.ai_turn:
        return
    move = get_ai_move()
    game_over, result = st.session_state.board.play(-1, move)
    if game_over:
        st.session_state.game_over = True
        if result == "win":
            st.session_state.result_message = "AI Wins!"
            st.session_state.ai_wins += 1
        else:
            st.session_state.result_message = "Draw!"
            st.session_state.draws += 1
    st.session_state.ai_turn = False

# Execute AI move
if st.session_state.ai_turn:
    ai_move()
    st.rerun()

# Status - centered
if st.session_state.game_over:
    status = st.session_state.result_message
else:
    status = "AI thinking..." if st.session_state.ai_turn else "Your turn"
st.markdown(f'<div class="status">{status}</div>', unsafe_allow_html=True)

# Game board
board_state = st.session_state.board.board_state

for row in range(3):
    cols = st.columns(3)
    for col in range(3):
        pos = row * 3 + col
        val = board_state[pos].item()
        
        if val == 1:
            label = "❌"
        elif val == -1:
            label = "⭕"
        else:
            label = " "
        
        with cols[col]:
            disabled = val != 0 or st.session_state.game_over or st.session_state.ai_turn
            if st.button(label, key=f"cell_{pos}", disabled=disabled, use_container_width=True):
                player_move(pos)
                st.rerun()

# Buttons row
st.markdown('<div class="btn-row">', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    if st.button("🔄 New Game", use_container_width=True):
        st.session_state.board.clear()
        st.session_state.game_over = False
        st.session_state.result_message = ""
        st.session_state.ai_turn = False
        st.rerun()
with c2:
    if st.button("🗑️ Reset All", use_container_width=True):
        st.session_state.board.clear()
        st.session_state.game_over = False
        st.session_state.result_message = ""
        st.session_state.ai_turn = False
        st.session_state.player_wins = 0
        st.session_state.ai_wins = 0
        st.session_state.draws = 0
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)
