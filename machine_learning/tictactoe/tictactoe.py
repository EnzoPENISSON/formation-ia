import numpy as np
import random
import json

# Flatten the board to make it easier for Q-learning
flatten_board = lambda board: [item for row in board for item in row]

# Board Representation
EMPTY = 0
PLAYER_X = 1
PLAYER_O = -1
PLAYER_PRINT = {PLAYER_X: 'X', PLAYER_O: 'O', EMPTY: ' '}

# Q-Learning Parameters
alpha = 0.75 # Slightly lower learning rate to reduce overfitting
gamma = 0.85  # Focus a bit more on immediate rewards
epsilon = 1.0  # Initial exploration rate
epsilon_min = 0.1  # Prevent epsilon from decaying too much
epsilon_decay = 0.995  # Slightly slower decay for more exploration
num_episodes = 100000  # Increase episodes for more learning
ratio = num_episodes // 100  # Print progress every 1% of episodes

# Q-Table (Initialize with zeros for all state-action pairs)
Q = {}

# Define all possible actions (positions on the board)
actions = [(i, j) for i in range(3) for j in range(3)]

def find_winning_move(state, player):
    """Find if there's a winning move for a player"""
    for action in actions:
        i, j = action
        if state[i][j] == EMPTY:
            state[i][j] = player  # Simulate the move
            if check_winner(state) == player:
                return action  # Return the winning move
            state[i][j] = EMPTY  # Undo the move if it doesn't result in a win
    return None  # No winning move found

def encode_state(board):
    """ Encode the state as a unique integer based on the board configuration. """
    return sum((board[i][j] + 1) * (3 ** (3 * i + j)) for i in range(3) for j in range(3))

def decode_state(encoded_state):
    """ Decode the integer back to a board configuration. """
    board = [[EMPTY] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            board[i][j] = (encoded_state // (3 ** (3 * i + j))) % 3 - 1
    return board

# Function to choose an action
def choose_action(state):
    """ AI chooses an action based on Q-table and strategy."""
    # Check if AI can win or block opponent from winning
    offensive_move = find_winning_move(state, PLAYER_X)
    defensive_move = find_winning_move(state, PLAYER_O)

    if offensive_move is not None:
        return offensive_move  # AI wins if possible
    elif defensive_move is not None:
        return defensive_move  # AI blocks the opponent if needed

    # Otherwise, choose randomly or based on Q-values
    valid_actions = [(i, j) for i in range(3) for j in range(3) if state[i][j] == EMPTY]
    if not valid_actions:
        return None  # No valid actions left, game over

    return random.choice(valid_actions)  # Choose randomly from valid moves

# Function to check the winner
def check_winner(board):
    """ Check if a player has won the game on the board. """
    win_states = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]

    # Flatten the board for easier checking of winning combinations
    flat_board = flatten_board(board)

    for (x, y, z) in win_states:
        if flat_board[x] == flat_board[y] == flat_board[z] != EMPTY:
            return flat_board[x]  # Return the player (X or O) who has won

    # If no winner, check for draw
    if EMPTY not in flat_board:
        return 0  # Draw

    return None  # Game still ongoing

# Reward function
def get_reward(state, result):
    if result == PLAYER_X:
        return 100  # Win for AI
    elif result == PLAYER_O:
        return -100  # Loss for AI
    elif result == 0:
        return 10  # Draw (good result for AI as it avoids a loss)

    # Introduce small penalties for non-optimal moves
    offensive_move = find_winning_move(state, PLAYER_X)
    defensive_move = find_winning_move(state, PLAYER_O)

    if offensive_move or defensive_move:
        return 50  # AI is close to winning or blocking an opponent
    return -10  # Default penalty for intermediate states

# Function to save the Q-table to a JSON file
def save_model(Q, filename="q_table.json"):
    # Convert state-action pairs to a serializable format
    serializable_Q = {}
    for state, actions in Q.items():
        state_str = str(state)  # Convert the state tuple into a string to use as a key
        serializable_Q[state_str] = actions

    # Write the Q-table to a JSON file
    with open(filename, 'w') as f:
        json.dump(serializable_Q, f, indent=4)
    print(f"Model saved to {filename}")

# Training the agent
def train_agent():
    global Q, epsilon
    for episode in range(num_episodes):
        # Start with an empty board
        state = [[EMPTY] * 3 for _ in range(3)]
        step = 0
        done = False
        while not done:
            action = choose_action(state)
            # Update the board with the action (place the AI's move)
            new_state = [row[:] for row in state]  # Copy the state
            new_state[action[0]][action[1]] = PLAYER_X

            # Check if the game ended after the move
            result = check_winner(new_state)
            reward = get_reward(new_state, result)

            # Update the Q-table
            Q[(encode_state(state), action)] = Q.get((encode_state(state), action), 0) + alpha * (reward + gamma * max([Q.get((encode_state(new_state), a), 0) for a in actions]) - Q.get((encode_state(state), action), 0))

            state = new_state
            step += 1

            if reward != -10 or step > 9:
                done = True

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if (episode + 1) % ratio == 0:
            print(f"Episode {episode + 1}/{num_episodes} completed")

    print("Training completed")
    # Save the Q-table model
    save_model(Q)

def load_model(filename="q_table.json"):
    try:
        with open(filename, 'r') as f:
            Q = json.load(f)
        print(f"Model loaded from {filename}")
        return Q
    except FileNotFoundError:
        train_agent()  # Train a new agent if the model is not found
        return Q

# Function to simulate a game between the AI and a random opponent
def simulate_game(ai_player, opponent=None):
    board = [[EMPTY] * 3 for _ in range(3)]  # Start with an empty board
    current_player = ai_player  # AI starts first
    game_over = False
    winner = None

    while not game_over:
        # Get the move for the current player
        if current_player == ai_player:
            action = choose_action(board)  # AI chooses its action
            board[action[0]][action[1]] = PLAYER_X
        else:
            if opponent is not None:
                action = opponent(board)
                board[action[0]][action[1]] = PLAYER_O
            else:
                # Random opponent chooses a valid action
                valid_actions = [(i, j) for i in range(3) for j in range(3) if board[i][j] == EMPTY]
                action = random.choice(valid_actions)
                board[action[0]][action[1]] = PLAYER_O

        # Check for a winner
        winner = check_winner(board)
        if winner is not None or all(board[i][j] != EMPTY for i in range(3) for j in range(3)):
            game_over = True

        # Switch player
        current_player = PLAYER_O if current_player == PLAYER_X else PLAYER_X

    return winner

# Evaluate the AI against a random opponent
def evaluate_accuracy(num_games):
    ai_wins = 0
    ai_draws = 0
    ai_losses = 0

    for _ in range(num_games):
        result = simulate_game(PLAYER_X)  # AI plays as X, opponent as O
        if result == PLAYER_X:
            ai_wins += 1
        elif result == 0:
            ai_draws += 1
        else:
            ai_losses += 1

    accuracy = (ai_wins + ai_draws) / num_games
    print(f"AI wins: {ai_wins}, AI draws: {ai_draws}, AI losses: {ai_losses}")
    print(f"Accuracy: {accuracy}")

def play_game():
    """Function to play a single game between the human and AI."""
    board = [[EMPTY] * 3 for _ in range(3)]  # Start with an empty board
    current_player = PLAYER_X  # AI starts first
    game_over = False
    winner = None

    while not game_over:
        print_board(board)

        if current_player == PLAYER_O:
            # Human player's turn (player O)
            action = human_move(board)  # Ask the player to move
            if action is None:
                print("Invalid move. Try again.")
                continue  # If the move is invalid, ask for a new one
            board[action[0]][action[1]] = PLAYER_O
        else:
            # AI player's turn (player X)
            print("AI is making its move...")
            action = choose_action(board)  # AI chooses its action
            if action is None:
                print("AI has no valid move left!")  # This is for debugging purposes.
                break
            board[action[0]][action[1]] = PLAYER_X

        # Check for a winner after each move
        winner = check_winner(board)

        # Game ends when there's a winner or the board is full
        if winner is not None or all(board[i][j] != EMPTY for i in range(3) for j in range(3)):
            game_over = True

        # Switch player
        current_player = PLAYER_O if current_player == PLAYER_X else PLAYER_X

    # Final board display and result
    print_board(board)
    if winner == PLAYER_X:
        print("AI (X) wins!")
    elif winner == PLAYER_O:
        print("You (O) win!")
    else:
        print("It's a draw!")

    return winner

def print_board(board):
    """Prints the current board."""
    for row in board:
        print(" | ".join([PLAYER_PRINT[cell] for cell in row]))
        print("---------")

def human_move(board):
    """Gets the human player's move."""
    try:
        move = int(input("Enter your move (0-8): "))  # Get input from user (0-8)
        if move < 0 or move > 8:
            print("Invalid move! Choose a number between 0 and 8.")
            return None
        row, col = divmod(move, 3)  # Convert move to row, col
        if board[row][col] != EMPTY:
            print("That space is already taken! Try again.")
            return None
        return (row, col)
    except ValueError:
        print("Invalid input! Please enter a number.")
        return None



def play_again():
    """ Ask the player if they want to play another game. """
    while True:
        choice = input("Do you want to play again? (y/n): ").lower()
        if choice == 'y':
            return True
        elif choice == 'n':
            return False
        else:
            print("Invalid input! Please enter 'y' for yes or 'n' for no.")

def main():
    """ Main function to start the game and handle replay logic. """
    print("Welcome to Tic-Tac-Toe! You are playing as O and the AI is playing as X.")

    # Loop for multiple games
    while True:
        play_game()  # Play a single game
        if not play_again():
            print("Thank you for playing! Goodbye!")
            break


Q = load_model()  # Load the Q-table model

#train_agent()

# Call the function to evaluate the AI's performance
evaluate_accuracy(10000)


# Start the game
main()
