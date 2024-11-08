import numpy as np
import random
import matplotlib.pyplot as plt

# Définition du labyrinthe
# 0 : chemin libre
# 1 : mur
maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 1],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0]
]

n_rows = len(maze)
n_cols = len(maze[0])

# Définition des états
# Chaque état est une position (ligne, colonne) dans le labyrinthe
state_space = [(i, j) for i in range(n_rows) for j in range(n_cols) if maze[i][j] == 0]

# Actions possibles : haut, bas, gauche, droite
actions = ['haut', 'bas', 'gauche', 'droite']
action_space = list(range(len(actions)))  # 0: haut, 1: bas, 2: gauche, 3: droite

# Paramètres de Q-Learning
alpha = 0.8  # Taux d'apprentissage
gamma = 0.95  # Facteur de discount
epsilon = 1.0  # Taux d'exploration initial
epsilon_min = 0.01
epsilon_decay = 0.995
num_episodes = 500

# Initialisation de la table Q
Q = {}
for state in state_space:
    Q[state] = np.zeros(len(actions))


# Fonction pour choisir une action
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(action_space)
    else:
        return np.argmax(Q[state])


# Fonction pour vérifier si un mouvement est valide
def is_valid(state):
    i, j = state
    return 0 <= i < n_rows and 0 <= j < n_cols and maze[i][j] == 0


# Fonction pour obtenir le prochain état
def next_state(state, action):
    i, j = state
    if action == 0:  # haut
        i -= 1
    elif action == 1:  # bas
        i += 1
    elif action == 2:  # gauche
        j -= 1
    elif action == 3:  # droite
        j += 1
    new_state = (i, j)
    if is_valid(new_state):
        return new_state
    else:
        return state  # reste sur place si mouvement invalide


# Définition de l'état de départ et de l'objectif
start_state = (0, 0)
goal_state = (4, 4)


# Fonction de récompense
def get_reward(state):
    if state == goal_state:
        return 100
    else:
        return -1


# Entraînement de l'agent
for episode in range(num_episodes):
    state = start_state
    step = 0
    while state != goal_state and step < 100:
        action = choose_action(state)
        new_state = next_state(state, action)
        reward = get_reward(new_state)

        # Mise à jour de Q
        Q[state][action] = Q[state][action] + alpha * (reward + gamma * np.max(Q[new_state]) - Q[state][action])

        state = new_state
        step += 1

    # Décroissance de epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{num_episodes} complété")

print("Entraînement terminé.\n")

# Extraction du chemin optimal
state = start_state
path = [state]
while state != goal_state:
    action = np.argmax(Q[state])
    state = next_state(state, action)
    path.append(state)
    if len(path) > 50:  # Sécurité pour éviter les boucles infinies
        print("Chemin trop long, arrêt de la recherche.")
        break

print("Chemin trouvé :")
print(path)