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

# Inject a bit of styling for a bolder look
st.markdown(
    """
    <style>
    :root {
        --bg: #0b1220;
        --panel: #0f172a;
        --border: #1f2937;
        --accent: #22d3ee;
        --accent-2: #a855f7;
        --text: #e5e7eb;
        --muted: #94a3b8;
    }
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 20% 20%, rgba(34,211,238,0.08), transparent 25%),
                    radial-gradient(circle at 80% 0%, rgba(168,85,247,0.12), transparent 25%),
                    var(--bg);
        color: var(--text);
    }
    [data-testid="stHeader"] { background: transparent; }
    .panel {
        background: linear-gradient(145deg, var(--panel), #0b1629);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 20px 50px rgba(0,0,0,0.35);
    }
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(34,211,238,0.12);
        border: 1px solid rgba(34,211,238,0.25);
        color: var(--text);
        font-weight: 600;
    }
    .legend-item { color: var(--muted); font-size: 0.95rem; }
    .legend-emoji { font-size: 1.1rem; }
    div.stButton > button {
        border-radius: 12px;
        border: 1px solid var(--border);
        background: linear-gradient(160deg, #111c32, #0c1424);
        color: #e2e8f0;
        font-size: 2.4rem;
        font-weight: 700;
        height: 120px;
        width: 100%;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 12px 24px rgba(0,0,0,0.3);
        transition: transform 120ms ease, border-color 120ms ease, box-shadow 120ms ease;
    }
    div.stButton > button:hover { border-color: var(--accent); box-shadow: 0 16px 32px rgba(34,211,238,0.14); transform: translateY(-1px); }
    div.stButton > button:active { transform: translateY(0); box-shadow: 0 10px 18px rgba(0,0,0,0.35); }
    div.stButton > button:disabled {
        background: linear-gradient(160deg, #162341, #0f1a2e);
        border-color: #1f2b44;
        color: #cbd5e1;
        opacity: 0.9;
        cursor: not-allowed;
    }
    .board-card { padding: 12px; border-radius: 16px; background: rgba(255,255,255,0.01); border: 1px solid var(--border); }
    .status-msg { font-size: 1rem; color: var(--muted); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎮 Tic-Tac-Toe vs AI")
st.caption("Play as X, challenge the neural net as O.")

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

# Status and legend
turn_label = "🤖 AI is thinking" if st.session_state.ai_turn else "🎯 Your move"
if st.session_state.game_over:
    game_state = st.session_state.result_message or "Game over"
    state_style = "background: rgba(168,85,247,0.16); border-color: rgba(168,85,247,0.35);"
else:
    game_state = "In progress"
    state_style = "background: rgba(34,211,238,0.12); border-color: rgba(34,211,238,0.3);"

info_col, legend_col = st.columns([1.6, 1])
with info_col:
    st.markdown(
        f"""
        <div class="panel">
            <div class="pill" style="margin-bottom: 6px;">{turn_label}</div><br>
            <div class="pill" style="margin-bottom: 6px; {state_style}">{game_state}</div><br>
            <span class="status-msg">Model: NeuralNet · Board size 3x3</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with legend_col:
    st.markdown(
        """
        <div class="panel">
            <strong>Legend</strong><br>
            <div class="legend-item"><span class="legend-emoji">❌</span> You (X)</div>
            <div class="legend-item"><span class="legend-emoji">⭕</span> AI (O)</div>
            <div class="legend-item"><span class="legend-emoji">⬜</span> Empty cell</div>
            <div class="legend-item">First move is yours, then AI replies.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Display game board
board_state = st.session_state.board.board_state

st.markdown('<div class="panel board-card">', unsafe_allow_html=True)
cols = st.columns(3)
for i in range(3):
    for j in range(3):
        position = i * 3 + j
        cell_value = board_state[position].item()
        
        if cell_value == 1:
            label = "❌"
            disabled = True
        elif cell_value == -1:
            label = "⭕"
            disabled = True
        else:
            label = "⬜"
            disabled = st.session_state.game_over or st.session_state.ai_turn
        
        with cols[j]:
            if st.button(label, key=f"cell_{position}", disabled=disabled, use_container_width=True):
                player_move(position)
                st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# Display result message
if st.session_state.result_message:
    st.success(st.session_state.result_message)

# Reset button row
left, right = st.columns([1, 1])
with left:
    if st.button("🔄 New Game", use_container_width=True):
        st.session_state.board.clear()
        st.session_state.game_over = False
        st.session_state.result_message = ""
        st.session_state.ai_turn = False
        st.rerun()
with right:
    st.caption("Tip: Corners often lead to faster wins for X.")

# Display game status
if not st.session_state.game_over:
    if st.session_state.ai_turn:
        st.info("🤖 AI is thinking...")
    else:
        st.info("🎯 Your turn! Click a cell to place X")
