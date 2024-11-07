# Travail Pratique : Nettoyage de Données Films


Dans ce travail pratique, vous allez travailler sur un jeu de données contenant des informations sur différents films. L'objectif principal est d'apprendre à nettoyer et préparer des données pour une analyse ultérieure, en vous concentrant particulièrement sur la prédiction du rating des films.

Le dataset contient plusieurs types de données : des valeurs numériques, des catégories, et une colonne "One-line" qui contient des descriptions textuelles des films. Bien que cette dernière puisse être intéressante pour prédire le rating d'un film, son analyse nécessiterait des techniques de traitement du langage naturel (NLP) que nous n'avons pas encore abordées dans ce cours. Nous nous concentrerons donc uniquement sur les colonnes numériques et catégoriques.

Votre première tâche sera d'examiner les données manquantes dans le dataset. Il est rare d'avoir des données parfaites dans le monde réel, et la façon dont vous gérez ces valeurs manquantes peut avoir un impact significatif sur vos résultats. Vous devrez donc développer et tester différentes stratégies pour traiter ces valeurs manquantes. Par exemple, vous pourriez choisir de supprimer les lignes incomplètes, ou de remplacer les valeurs manquantes par la moyenne ou la médiane des valeurs existantes. Chaque approche a ses avantages et ses inconvénients qu'il faudra analyser et expliquer.

Ensuite, vous devrez vous occuper des outliers, ces valeurs aberrantes qui peuvent fausser vos analyses. Là encore, plusieurs approches sont possibles : vous pouvez choisir de les supprimer, de les plafonner à certaines valeurs, ou d'appliquer des transformations mathématiques pour atténuer leur impact. Il est important de justifier votre choix en expliquant pourquoi une méthode est plus appropriée qu'une autre dans ce contexte précis.

Un cas particulier à considérer est celui de la colonne "Rating", puisque c'est la variable que nous cherchons à prédire. Si cette colonne contient des valeurs manquantes, vous devrez décider soit de supprimer ces entrées, soit de les remplacer par une valeur estimée (moyenne, médiane, etc.). Cette décision est cruciale car elle affectera directement la qualité de vos prédictions.

Pour évaluer l'efficacité de vos différentes stratégies de nettoyage, un [algorithme](basic_movies_rating_classificaiton_algorithm.py) de test vous est fourni. Cet algorithme prend en entrée votre dataset nettoyé (au format CSV) et retourne un score de précision pour la prédiction des ratings. La précision est calculée en comparant les prédictions de l'algorithme avec les véritables valeurs de rating. Plus votre stratégie de nettoyage est pertinente, meilleure sera la précision.

Attention : Si l'algorithme ne focntionne pas alors, le nettoyage n'est pas bien fait. 

Pour le rendu de ce travail, vous avez deux options :
 - La première option consiste à rendre un notebook Jupyter (.ipynb) qui contiendra à la fois votre code, vos explications et vos analyses. Cette option permet une présentation intégrée et interactive de votre travail.
 - La seconde option est de séparer le code et les explications en deux fichiers distincts : un fichier Python (.py) pour le code, et un document Word (.docx) pour vos explications et analyses. Dans ce cas, vous devrez compresser ces deux fichiers dans une archive ZIP avant de la déposer sur Moodle.
 - Dans les deux cas, vos explications devront être détaillées et justifiées. Il ne suffit pas de dire qu'une stratégie fonctionne mieux qu'une autre, il faut expliquer pourquoi, en vous appuyant sur vos résultats et votre compréhension des données.