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

    # Exploration
    if np.random.uniform(0, 1) < epsilon:
        available_actions = [a for a in actions_space if state[a] == EMPTY]
        return random.choice(available_actions)
    else:
        # Exploitation
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
        return 1  # Victoire
    elif result == PLAYER_O:
        return -1  # Défaite
    elif result == 0:
        return 0  # Draw
    else:
        return -1  # Ongoing game


def opponent_move_strategic(board):
    """Définit un coup pour l'adversaire en tenant compte des stratégies de défense et d'attaque."""
    # 1. Vérifier les possibilités de victoire de l'adversaire
    for action in actions_space:
        if board[action] == EMPTY:
            board[action] = PLAYER_O
            if check_winner(board) == PLAYER_O:
                board[action] = EMPTY  # Annuler le coup avant de retourner
                return action  # Coup gagnant pour l'adversaire
            board[action] = EMPTY  # Annuler le coup

    # 2. Bloquer une possible victoire de l'agent
    for action in actions_space:
        if board[action] == EMPTY:
            board[action] = PLAYER_X
            if check_winner(board) == PLAYER_X:
                board[action] = PLAYER_O  # Placer le coup bloquant
                board[action] = EMPTY  # Annuler le coup après blocage
                return action  # Bloque l'agent
            board[action] = EMPTY  # Annuler le coup

    # 3. Jouer au centre si disponible pour augmenter les chances de victoire future
    if board[4] == EMPTY:
        return 4

    # 4. Jouer dans les coins si disponibles
    for action in [0, 2, 6, 8]:
        if board[action] == EMPTY:
            return action

    # 5. En dernier recours, jouer de manière aléatoire
    return random.choice([a for a in actions_space if board[a] == EMPTY])

def trainning():
    # Intégrer cette stratégie dans la boucle de formation
    for episode in range(num_episodes):
        board = np.array([EMPTY] * 9)  # Réinitialiser le jeu
        state = get_state(board)
        done = False
        step = 0

        while not done and step < 9:
            # Agent joue son coup
            action = choose_action(state)
            board[action] = PLAYER_X
            new_state = get_state(board)
            result = check_winner(board)
            reward = get_reward(result)

            if new_state not in Q:
                Q[new_state] = np.zeros(len(actions_space))

            # Mettre à jour la valeur Q
            Q[state][action] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[state][action])

            if result is not None:
                done = True
            else:
                # L'adversaire joue de manière stratégique
                opponent_action = opponent_move_strategic(board)
                board[opponent_action] = PLAYER_O
                result = check_winner(board)
                reward = get_reward(result)

                if result is not None:
                    done = True

            state = new_state
            step += 1

        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{num_episodes} complété")

    print("Entrainement terminé")

    #Save the trained Q-table in a json file
    import json
    with open('Q_table.json', 'w') as f:
        json.dump(Q, f)


# Fonction pour jouer un jeu avec l'IA entrainée
def play_game():
    global result
    board = np.array([EMPTY] * 9)
    state = get_state(board)
    done = False

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

#play_game()
trainning()