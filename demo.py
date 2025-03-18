import time

import streamlit as st
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam

from ai2048.game import Game
from ai2048.model import Policy, Value


@st.cache_resource
def get_policy():
    policy = Policy()
    checkpoint = torch.load("./ckpt-2025-03-17T23:06:44.826040-2600.pt")
    policy.load_state_dict(checkpoint['policy'])
    return policy


def play_game():
    game = Game()
    states = [game.clone().state]
    while game.alive():
        valid_actions = torch.tensor([float(game.valid(i)) for i in range(4)])
        policy_output = get_policy()(torch.tensor(game.state), valid_actions)
        action = torch.multinomial(policy_output, 1).item()
        game.move(action)
        states.append(game.clone().state)
    return states


def display_2048_board(board):
    """
    Display a 2048 game board in Streamlit with proper styling.
    
    Parameters:
    board (numpy.ndarray or list): A 4x4 array representing the 2048 game board,
                                  where 0 represents an empty cell.
    """
    # Convert to numpy array if it's a list
    if isinstance(board, list):
        board = np.array(board)
    
    # Ensure the board is 4x4
    if board.shape != (4, 4):
        st.error("Board must be 4x4. Current shape: " + str(board.shape))
        return
    
    # Define colors for each tile value (using 2048 game colors)
    color_map = {
        0: "#CDC1B4",     # Empty tile
        2: "#EEE4DA",
        4: "#EDE0C8",
        8: "#F2B179",
        16: "#F59563",
        32: "#F67C5F",
        64: "#F65E3B",
        128: "#EDCF72",
        256: "#EDCC61",
        512: "#EDC850",
        1024: "#EDC53F",
        2048: "#EDC22E",
    }
    
    # For values not in our color map, use the 2048+ color
    default_color = "#3C3A32"
    
    # Convert board to DataFrame for easier styling
    df = pd.DataFrame(board)
    
    # Create a custom CSS for the game board
    st.markdown("""
    <style>
    .tile-container {
        background-color: #BBADA0;
        border-radius: 6px;
        padding: 10px;
        width: fit-content;
    }
    .tile {
        width: 80px;
        height: 80px;
        border-radius: 3px;
        display: flex;
        justify-content: center;
        align-items: center;
        font-family: 'Arial', sans-serif;
        font-weight: bold;
        font-size: 24px;
        text-align: center;
        color: #776E65;
    }
    .value-0 {
        color: transparent;
    }
    .value-high {
        color: #F9F6F2;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Start the tile container
    s = """<div class="tile-container">"""

    # Create the board
    for i in range(4):
        # Start a row
        s += '<div style="display: flex;">'
        for j in range(4):
            value = board[i][j]
            
            # Determine tile color
            bg_color = color_map.get(value, default_color)
            
            # Determine text color class (dark for low values, light for high values)
            text_class = "value-0" if value == 0 else "value-high" if value >= 8 else ""
            
            # Display text as empty for 0, and actual value for others
            display_text = "" if value == 0 else str(value)
            
            # Create the tile
            s += f'<div class="tile {text_class}" style="background-color: {bg_color};">{display_text}</div>'
        s += '</div>'
    
    s += "</div>"

    st.markdown(s, unsafe_allow_html=True)
    
    # Display score information
    total_score = np.sum(board)
    max_tile = np.max(board)
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Moves made", st.session_state.index)
    with col2:
        st.metric("Highest Tile", max_tile)
    with col3:
        st.metric("Game Length", len(states) - 1)

@st.cache_data
def winner_game():
    game = play_game()
    while max(np.array(game[-1]).flatten()) < 2048:
        game = play_game()
    return game

if __name__ == "__main__":
    # Streamlit UI
    st.title("2048 Game Viewer")

    if st.button("Regenerate") or "game" not in st.session_state:
        # game = play_game()
        # while max(np.array(game[-1]).flatten()) < 2048:
        #     game = play_game()
        st.session_state.game = winner_game()
        st.session_state.index = len(st.session_state.game) - 1
    
    states = st.session_state.game

    # Initialize session state
    if 'index' not in st.session_state:
        st.session_state.index = 0

    cols = st.columns(4)
    with cols[0]:
        if st.button("Start"):
            st.session_state.index = 0
    with cols[1]:
        if st.button("Previous"):
            st.session_state.index = max(0, st.session_state.index - 1)
    with cols[2]:
        if st.button("Next"):
            st.session_state.index = min(len(states) - 1, st.session_state.index + 1)
    with cols[3]:
        if st.button("End"):
            st.session_state.index = len(states) - 1
    if st.button("Auto play"):
        st.session_state.auto_play = True

    # Display the current state
    current_state = states[st.session_state.index]
    display_2048_board(current_state)
    
    if "auto_play" in st.session_state and st.session_state.auto_play:
        if st.session_state.index == len(states) - 1:
            st.session_state.auto_play = False
        else:
            st.session_state.index += 1
            time.sleep(0.01)
            st.rerun()
