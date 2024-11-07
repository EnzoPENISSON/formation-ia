from collections import defaultdict

import numpy as np
import random

# Board Representation
EMPTY = 0
PLAYER_X = 1
PLAYER_O = -1
PLAYER_PRINT = {PLAYER_X: 'X', PLAYER_O: 'O', EMPTY: ' '}

# Possible actions are the positions on the board (0 to 8 for a 3x3 board)
actions_space = list(range(9))

# Q-Learning Parameters
alpha = 0.8  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
num_episodes = 100000

# Q-Table
Q = {}

# Helper function to get the board as a tuple (hashable state)
def get_state(board):
    return tuple(board.flatten())

# Fonction pour choisir une action
def choose_action(state):
    # Initialize Q-values for new state
    if state not in Q:
        Q[state] = np.zeros(len(actions_space))

    # Epsilon-greedy strategy Exploration
    if np.random.uniform(0, 1) < epsilon:
        # Choose a random valid action
        available_actions = [a for a in actions_space if state[a] == EMPTY]
        return random.choice(available_actions)
    else:
        # Exploitation
        # Choose the action with max Q-value for the current state
        q_values = Q[state]
        available_actions = [a for a in actions_space if state[a] == EMPTY]
        best_action = max(available_actions, key=lambda x: q_values[x])
        return best_action

# Fonction pour vérifier le gagnant, le match nul ou le jeu en cours
def check_winner(board):
    win_states = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                  (0, 3, 6), (1, 4, 7), (2, 5, 8),
                  (0, 4, 8), (2, 4, 6)]
    for (x, y, z) in win_states:
        if board[x] == board[y] == board[z] != EMPTY:
            return board[x]
    if EMPTY not in board:
        return 0  # Match null
    return None

# Fonction de récompense
def get_reward(result):
    if result == PLAYER_X:
        return 100  # Victoire
    elif result == PLAYER_O:
        return -100  # Défaite
    elif result == 0:
        return 50  # Draw
    else:
        return -1  # Ongoing game

# Initialize opponent Q-table with the same structure as the main agent's Q-table
opponent_Q = defaultdict(lambda: np.zeros(len(actions_space)))

# Opponent parameters
opponent_epsilon = 0.1  # Probability for the opponent to explore
opponent_alpha = 0.5    # Learning rate
opponent_gamma = 0.9    # Discount factor

def opponent_move_probabilistic(state, board):
    # Choose an action for the opponent using ε-greedy policy
    if np.random.rand() < opponent_epsilon:
        # Random action (exploration)
        action = random.choice([a for a in actions_space if board[a] == EMPTY])
    else:
        # Best action based on the opponent's Q-table (exploitation)
        action_values = opponent_Q[state]
        action = np.argmax(action_values)
        if board[action] != EMPTY:  # Handle cases where best action is invalid
            action = random.choice([a for a in actions_space if board[a] == EMPTY])
    return action


# Entrainer l'agent
for episode in range(num_episodes):
    board = np.array([EMPTY] * 9)  # Reset jeu
    state = get_state(board)
    done = False
    step = 0

    # Use the probabilistic opponent in your main training loop
    while not done and step < 9:
        action = choose_action(state)
        board[action] = PLAYER_X
        new_state = get_state(board)
        result = check_winner(board)
        reward = get_reward(result)

        if new_state not in Q:
            Q[new_state] = np.zeros(len(actions_space))

        Q[state][action] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[state][action])

        if result is not None:
            done = True
        else:
            # Opponent plays using its Q-learning strategy
            opponent_action = opponent_move_probabilistic(new_state, board)
            board[opponent_action] = PLAYER_O
            result = check_winner(board)
            reward = get_reward(result)

            if result is not None:
                done = True
            else:
                # Update opponent Q-table based on outcome
                opponent_Q[state][opponent_action] += opponent_alpha * (
                        reward + opponent_gamma * np.max(opponent_Q[new_state]) - opponent_Q[state][opponent_action]
                )

        state = new_state
        step += 1


    if (episode + 1) % 1000 == 0:
        print(f"Episode {episode + 1}/{num_episodes} complété")

print("Entrainement terminé")

# Fonction pour jouer un jeu avec l'IA entrainée
def play_game():
    board = np.array([EMPTY] * 9)
    state = get_state(board)
    done = False

    Who_start = random.choice([PLAYER_X, PLAYER_O])

    while not done:

        # IA joue
        action = choose_action(state)
        board[action] = PLAYER_X
        print(f"IA joue: {action}")
        print(board.reshape(3, 3))

        result = check_winner(board)
        if result is not None:
            break

        # Joueur joue
        opponent_action = int(input("Entrez une action (0-8): "))
        board[opponent_action] = PLAYER_O
        print(board.reshape(3, 3))

        result = check_winner(board)
        if result is not None:
            break

        state = get_state(board)

    if result == PLAYER_X:
        print("IA à gagné!")
    elif result == PLAYER_O:
        print("Joueur à gagné!")
    else:
        print("Match nul")

play_game()
