import numpy as np
import random

# Board Representation
EMPTY = 0
PLAYER_X = 1
PLAYER_O = -1

# Possible actions are the positions on the board (0 to 8 for a 3x3 board)
actions_space = list(range(9))

# Q-Learning Parameters
alpha = 0.8  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
num_episodes = 10000

# Q-Table
Q = {}

# Helper function to get the board as a tuple (hashable state)
def get_state(board):
    return tuple(board.flatten())

# Function to choose an action
def choose_action(state):
    if state not in Q:
        Q[state] = np.zeros(len(actions_space))  # Initialize Q-values for new state
    if np.random.uniform(0, 1) < epsilon:
        # Choose a random valid action
        available_actions = [a for a in actions_space if state[a] == EMPTY]
        return random.choice(available_actions)
    else:
        # Choose the action with max Q-value for the current state
        q_values = Q[state]
        available_actions = [a for a in actions_space if state[a] == EMPTY]
        best_action = max(available_actions, key=lambda x: q_values[x])
        return best_action

# Function to check for a win, loss, or draw
def check_winner(board):
    win_states = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                  (0, 3, 6), (1, 4, 7), (2, 5, 8),
                  (0, 4, 8), (2, 4, 6)]
    for (x, y, z) in win_states:
        if board[x] == board[y] == board[z] != EMPTY:
            return board[x]
    if EMPTY not in board:
        return 0  # Draw
    return None  # Game ongoing

# Reward function
def get_reward(result):
    if result == PLAYER_X:
        return 100  # Win
    elif result == PLAYER_O:
        return -100  # Loss
    elif result == 0:
        return 50  # Draw
    else:
        return -1  # Ongoing game

# Training the agent
for episode in range(num_episodes):
    board = np.array([EMPTY] * 9)  # Reset board
    state = get_state(board)
    done = False
    step = 0

    while not done and step < 9:
        action = choose_action(state)

        # Perform action
        board[action] = PLAYER_X  # Agent plays X
        new_state = get_state(board)

        # Check game outcome
        result = check_winner(board)
        reward = get_reward(result)

        if new_state not in Q:
            Q[new_state] = np.zeros(len(actions_space))

        # Update Q-Value
        Q[state][action] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[state][action])

        # If the game is won, lost, or drawn, finish the episode
        if result is not None:
            done = True
        else:
            # Opponent plays randomly
            opponent_action = random.choice([a for a in actions_space if board[a] == EMPTY])
            board[opponent_action] = PLAYER_O
            result = check_winner(board)
            reward = get_reward(result)

            if result is not None:
                done = True

        state = new_state
        step += 1

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if (episode + 1) % 1000 == 0:
        print(f"Episode {episode + 1}/{num_episodes} completed")

print("Training completed")

# play the game with the trained agent
def play_game():
    board = np.array([EMPTY] * 9)
    state = get_state(board)
    done = False

    while not done:
        # Agent plays
        action = choose_action(state)
        board[action] = PLAYER_X
        print(f"Agent plays: {action}")
        print(board.reshape(3, 3))

        result = check_winner(board)
        if result is not None:
            break

        # Opponent plays
        opponent_action = int(input("Enter opponent's action (0-8): "))
        board[opponent_action] = PLAYER_O
        print(board.reshape(3, 3))

        result = check_winner(board)
        if result is not None:
            break

        state = get_state(board)

    if result == PLAYER_X:
        print("Agent wins!")
    elif result == PLAYER_O:
        print("Opponent wins!")
    else:
        print("It's a draw!")

play_game()