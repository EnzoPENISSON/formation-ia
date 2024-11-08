## Apprentissage par renforcement 

L'apprentissage par renforcement est une méthode d'apprentissage automatique où un agent apprend à prendre des décisions en interagissant avec un environnement. L'agent agit de manière à maximiser une récompense cumulée à long terme. Pour mieux comprendre les concepts de base, utilisons un exemple simple d’un agent qui apprend à naviguer dans un labyrinthe.

Concepts Clés de l'Apprentissage par Renforcement

- État (State) :
    - Un état représente une situation dans laquelle l'agent se trouve à un moment donné. Dans notre exemple du labyrinthe, un état pourrait être la position de l'agent sur la grille du labyrinthe (par exemple, (x, y)).
    - Par exemple, si l'agent est en position (2, 3) dans le labyrinthe, cet état décrit où il se trouve par rapport à l'objectif et aux obstacles.

- Action (Action) :
    - Une action est une décision que l'agent peut prendre pour passer d'un état à un autre. Cela pourrait être "aller vers le haut", "aller vers le bas", "aller à gauche" ou "aller à droite" dans le cas du labyrinthe.
    - Chaque action permet à l'agent de se déplacer dans une direction spécifique sur la grille.

- Environnement (Environment) :
    - L’environnement est le monde dans lequel l'agent évolue. Il contient toutes les informations sur la situation actuelle, y compris la disposition du labyrinthe, les murs, et la position de la récompense finale.
    - L'environnement répond aux actions de l'agent en fournissant un nouvel état et une récompense.

- Épisode (Episode) :
    - Un épisode est une séquence complète d'actions prises par l'agent depuis un état de départ jusqu'à un état terminal (par exemple, l'agent atteint la récompense ou se retrouve bloqué).
    - Un épisode se termine lorsque l'agent atteint son objectif ou ne peut plus progresser.

- Récompense (Reward) :
    - Une récompense est un retour de l'environnement qui évalue l'action prise par l'agent. L'objectif de l'agent est de maximiser la somme des récompenses qu'il reçoit.
    - Par exemple, dans le labyrinthe :
        - L'agent reçoit une grande récompense positive (par exemple, +10) lorsqu'il atteint la sortie.
        - L'agent peut recevoir une petite pénalité (par exemple, -1) lorsqu'il se déplace vers une case vide pour encourager des déplacements efficaces.
        - S'il se heur``te à un mur, il pourrait recevoir une plus grande pénalité (par exemple, -5).  

## Explication de la méthode de Q-Learning

- Initialisation de la table Q :
    - Une table Q(s,a)Q(s,a) est initialisée avec des valeurs nulles. Elle est utilisée pour stocker les valeurs de récompense attendue pour chaque état ss et action aa.

  - Algorithme :
    - Pour chaque épisode :
        - Initialiser l'état ss
        - Jusqu'à ce que l'état soit terminal (la position finale est atteinte) :
        - Choisir une action aa en utilisant une politique (par exemple, epsilon-greedy pour explorer/exploiter).
        - Exécuter l'action aa et observer la récompense rr et le nouvel état s′s′.
        - Mettre à jour la valeur de QQ selon la règle de mise à jour :
        - Q(s,a)←Q(s,a)+α[r+gamma * max⁡aQ(s′,a)−Q(s,a)]
        - Q(s,a)←Q(s,a)+α[r+gamma * max​Q(s′,a)−Q(s,a)]
        - Mettre s=s′s=s′.