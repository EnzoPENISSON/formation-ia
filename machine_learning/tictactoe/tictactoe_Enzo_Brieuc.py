"""
Fait par: Enzo PENISSON et Brieuc Courapié
"""
import random
import json

# Aplatir le plateau pour faciliter l'apprentissage par renforcement (Q-learning)
flatten_board = lambda board: [item for row in board for item in row]

# Représentation du plateau
EMPTY = 0
PLAYER_X = 1
PLAYER_O = -1
PLAYER_PRINT = {PLAYER_X: 'X', PLAYER_O: 'O', EMPTY: ' '}

# Paramètres de Q-Learning
alpha = 0.75  # Taux d'apprentissage légèrement inférieur pour réduire le surapprentissage
gamma = 0.85  # Se concentrer un peu plus sur les récompenses immédiates
epsilon = 1.0  # Taux d'exploration initial
epsilon_min = 0.1  # Empêcher epsilon de diminuer trop
epsilon_decay = 0.995  # Décroissance légèrement plus lente pour plus d'exploration
num_episodes = 100000  # Augmenter le nombre d'épisodes pour plus d'apprentissage
ratio = num_episodes // 100  # Afficher la progression tous les 1% des épisodes

# Q-Table (Initialiser avec des zéros pour toutes les paires état-action)
Q = {}

# Définir toutes les actions possibles (positions sur le plateau)
actions = [(i, j) for i in range(3) for j in range(3)]

def find_winning_move(state, player):
    """Trouver s'il y a un coup gagnant pour un joueur"""
    for action in actions:
        i, j = action
        if state[i][j] == EMPTY:
            state[i][j] = player  # Simuler le coup
            if check_winner(state) == player:
                return action  # Retourner le coup gagnant
            state[i][j] = EMPTY  # Annuler le coup s'il ne mène pas à une victoire
    return None  # Aucun coup gagnant trouvé

def encode_state(board):
    """ Encoder l'état comme un entier unique basé sur la configuration du plateau. """
    return sum((board[i][j] + 1) * (3 ** (3 * i + j)) for i in range(3) for j in range(3))

def decode_state(encoded_state):
    """ Décoder l'entier en une configuration de plateau. """
    board = [[EMPTY] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            board[i][j] = (encoded_state // (3 ** (3 * i + j))) % 3 - 1
    return board

# Fonction pour choisir une action
def choose_action(state):
    """ L'IA choisit une action basée sur la Q-table et la stratégie."""
    # Vérifier si l'IA peut gagner ou bloquer l'adversaire de gagner
    offensive_move = find_winning_move(state, PLAYER_X)
    defensive_move = find_winning_move(state, PLAYER_O)

    if offensive_move is not None:
        return offensive_move  # L'IA gagne si possible
    elif defensive_move is not None:
        return defensive_move  # L'IA bloque l'adversaire si nécessaire

    # Sinon, choisir aléatoirement ou basé sur les valeurs Q
    valid_actions = [(i, j) for i in range(3) for j in range(3) if state[i][j] == EMPTY]
    if not valid_actions:
        return None  # Aucune action valide restante, jeu terminé

    return random.choice(valid_actions)  # Choisir aléatoirement parmi les coups valides

# Fonction pour vérifier le gagnant
def check_winner(board):
    """ Vérifier si un joueur a gagné la partie sur le plateau. """
    win_states = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]

    # Aplatir le plateau pour faciliter la vérification des combinaisons gagnantes
    flat_board = flatten_board(board)

    for (x, y, z) in win_states:
        if flat_board[x] == flat_board[y] == flat_board[z] != EMPTY:
            return flat_board[x]  # Retourner le joueur (X ou O) qui a gagné

    # Si aucun gagnant, vérifier pour un match nul
    if EMPTY not in flat_board:
        return 0  # Match nul

    return None  # Jeu toujours en cours

# Fonction de récompense
def get_reward(state, result):
    if result == PLAYER_X:
        return 100  # Victoire pour l'IA
    elif result == PLAYER_O:
        return -100  # Défaite pour l'IA
    elif result == 0:
        return 10  # Match nul (bon résultat pour l'IA car il évite une défaite)

    # Introduire de petites pénalités pour les coups non optimaux
    offensive_move = find_winning_move(state, PLAYER_X)
    defensive_move = find_winning_move(state, PLAYER_O)

    if offensive_move or defensive_move:
        return 50  # L'IA est proche de gagner ou de bloquer un adversaire
    return -10  # Pénalité par défaut pour les états intermédiaires

# Fonction pour sauvegarder la Q-table dans un fichier JSON
def save_model(Q, filename="q_table.json"):
    # Convertir les paires état-action en un format sérialisable
    serializable_Q = {}
    for state, actions in Q.items():
        state_str = str(state)  # Convertir le tuple d'état en chaîne de caractères pour l'utiliser comme clé
        serializable_Q[state_str] = actions

    # Écrire la Q-table dans un fichier JSON
    with open(filename, 'w') as f:
        json.dump(serializable_Q, f, indent=4)
    print(f"Modèle sauvegardé dans {filename}")

# Entraîner l'agent
def train_agent():
    global Q, epsilon
    for episode in range(num_episodes):
        # Commencer avec un plateau vide
        state = [[EMPTY] * 3 for _ in range(3)]
        step = 0
        done = False
        while not done:
            action = choose_action(state)
            # Mettre à jour le plateau avec l'action (placer le coup de l'IA)
            new_state = [row[:] for row in state]  # Copier l'état
            new_state[action[0]][action[1]] = PLAYER_X

            # Vérifier si le jeu s'est terminé après le coup
            result = check_winner(new_state)
            reward = get_reward(new_state, result)

            # Mettre à jour la Q-table
            Q[(encode_state(state), action)] = Q.get((encode_state(state), action), 0) + alpha * (reward + gamma * max([Q.get((encode_state(new_state), a), 0) for a in actions]) - Q.get((encode_state(state), action), 0))

            state = new_state
            step += 1

            if reward != -10 or step > 9:
                done = True

        # Décroître epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if (episode + 1) % ratio == 0:
            print(f"Épisode {episode + 1}/{num_episodes} terminé")

    print("Entraînement terminé")
    # Sauvegarder le modèle de la Q-table
    save_model(Q)

def load_model(filename="q_table.json"):
    try:
        with open(filename, 'r') as f:
            Q = json.load(f)
        print(f"Modèle chargé depuis {filename}")
        return Q
    except FileNotFoundError:
        train_agent()  # Entraîner un nouvel agent si le modèle n'est pas trouvé
        return Q

# Fonction pour simuler une partie entre l'IA et un adversaire aléatoire
def simulate_game(ai_player, opponent=None):
    board = [[EMPTY] * 3 for _ in range(3)]  # Commencer avec un plateau vide
    current_player = ai_player  # L'IA commence en premier
    game_over = False
    winner = None

    while not game_over:
        # Obtenir le coup pour le joueur actuel
        if current_player == ai_player:
            action = choose_action(board)  # L'IA choisit son action
            board[action[0]][action[1]] = PLAYER_X
        else:
            if opponent is not None:
                action = opponent(board)
                board[action[0]][action[1]] = PLAYER_O
            else:
                # L'adversaire aléatoire choisit une action valide
                valid_actions = [(i, j) for i in range(3) for j in range(3) if board[i][j] == EMPTY]
                action = random.choice(valid_actions)
                board[action[0]][action[1]] = PLAYER_O

        # Vérifier s'il y a un gagnant
        winner = check_winner(board)
        if winner is not None or all(board[i][j] != EMPTY for i in range(3) for j in range(3)):
            game_over = True

        # Changer de joueur
        current_player = PLAYER_O if current_player == PLAYER_X else PLAYER_X

    return winner

# Évaluer l'IA contre un adversaire aléatoire
def evaluate_accuracy(num_games):
    ai_wins = 0
    ai_draws = 0
    ai_losses = 0

    for _ in range(num_games):
        result = simulate_game(PLAYER_X)  # L'IA joue en tant que X, l'adversaire en tant que O
        if result == PLAYER_X:
            ai_wins += 1
        elif result == 0:
            ai_draws += 1
        else:
            ai_losses += 1

    accuracy = (ai_wins + ai_draws) / num_games
    print(f"Victoires de l'IA: {ai_wins}, Matchs nuls de l'IA: {ai_draws}, Défaites de l'IA: {ai_losses}")
    print(f"Précision: {accuracy}")

def play_game():
    """Fonction pour jouer une seule partie entre l'humain et l'IA."""
    board = [[EMPTY] * 3 for _ in range(3)]  # Commencer avec un plateau vide
    current_player = random.choice([PLAYER_X, PLAYER_O])  # Choisir aléatoirement qui commence
    game_over = False
    winner = None

    while not game_over:
        print_board(board)

        if current_player == PLAYER_O:
            # Tour du joueur humain (joueur O)
            action = human_move(board)  # Demander au joueur de jouer
            if action is None:
                print("Coup invalide. Réessayez.")
                continue  # Si le coup est invalide, demander un nouveau coup
            board[action[0]][action[1]] = PLAYER_O
        else:
            # Tour du joueur IA (joueur X)
            print("L'IA joue son coup...")
            action = choose_action(board)  # L'IA choisit son action
            if action is None:
                print("L'IA n'a plus de coup valide!")  # Ceci est à des fins de débogage.
                break
            board[action[0]][action[1]] = PLAYER_X

        # Vérifier s'il y a un gagnant après chaque coup
        winner = check_winner(board)

        # Le jeu se termine lorsqu'il y a un gagnant ou que le plateau est plein
        if winner is not None or all(board[i][j] != EMPTY for i in range(3) for j in range(3)):
            game_over = True

        # Changer de joueur
        current_player = PLAYER_O if current_player == PLAYER_X else PLAYER_X

    # Affichage final du plateau et résultat
    print_board(board)
    if winner == PLAYER_X:
        print("L'IA (X) gagne!")
    elif winner == PLAYER_O:
        print("Vous (O) gagnez!")
    else:
        print("C'est un match nul!")

    return winner

def print_board(board):
    """Affiche le plateau actuel."""
    for row in board:
        print(" | ".join([PLAYER_PRINT[cell] for cell in row]))
        print("---------")

def human_move(board):
    """Obtenir le coup du joueur humain."""
    try:
        move = int(input("Entrez votre coup (0-8): "))  # Obtenir l'entrée de l'utilisateur (0-8)
        if move < 0 or move > 8:
            print("Coup invalide! Choisissez un nombre entre 0 et 8.")
            return None
        row, col = divmod(move, 3)  # Convertir le coup en ligne, colonne
        if board[row][col] != EMPTY:
            print("Cet espace est déjà pris! Réessayez.")
            return None
        return (row, col)
    except ValueError:
        print("Entrée invalide! Veuillez entrer un nombre.")
        return None

def play_again():
    """ Demander au joueur s'il veut jouer une autre partie. """
    while True:
        choice = input("Voulez-vous rejouer? (y/n): ").lower()
        if choice == 'y':
            return True
        elif choice == 'n':
            return False
        else:
            print("Entrée invalide! Veuillez entrer 'y' pour oui ou 'n' pour non.")

def main():
    """ Fonction principale pour démarrer le jeu et gérer la logique de rejouer. """
    print("Bienvenue au Tic-Tac-Toe! Vous jouez en tant que O et l'IA joue en tant que X.")

    # Boucle pour plusieurs parties
    while True:
        play_game()  # Jouer une seule partie
        if not play_again():
            print("Merci d'avoir joué! Au revoir!")
            break

Q = load_model()  # Charger le modèle de la Q-table
#train_agent() # Ou entraîner un nouvel agent

# Appeler la fonction pour évaluer les performances de l'IA
evaluate_accuracy(1000)

# Démarrer le jeu
main()
