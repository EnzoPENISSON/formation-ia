# [Linear Rgression](https://mlu-explain.github.io/linear-regression/)


## Objectifs
- Comprendre le concept de la régression linéaire.
- Apprendre à utiliser la bibliothèque scikit-learn pour implémenter une régression linéaire.
- Analyser la performance du modèle.
- Visualiser les résultats.

## Introduction à la Régression Linéaire

La régression linéaire est une méthode statistique qui permet de modéliser la relation entre une ou plusieurs variables indépendantes (ou features) et une variable dépendante (ou target) en ajustant une ligne de meilleure ajustement.
Installation des bibliothèques nécessaires

```
pip install pandas scikit-learn matplotlib seaborn
```

## Étape 1 : Importer les Bibliothèques et Charger les Données

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Charger le dataset
df = pd.read_csv('house_price.csv')  # Assurez-vous d'avoir le fichier "house_price.csv"
print(df.head())
```

## Étape 2 : Exploration des Données

- Afficher les informations générales sur le dataset.
- Vérifier les valeurs manquantes.
- Visualiser les relations entre les variables.

```python
# Informations générales sur le dataset
print(df.info())

# Statistiques descriptives
print(df.describe())

# Vérification des valeurs manquantes
print(df.isnull().sum())

# Visualisation de la relation entre les variables
sns.pairplot(df)
plt.show()
```

## Étape 3 : Prétraitement des Données

- Gérer les valeurs manquantes.
- Sélectionner les features pertinentes.

## Étape 4 : Séparer les Données en Ensembles d'Entraînement et de Test

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Étape 5 : Implémenter la Régression Linéaire

```python
# Créer le modèle
model = LinearRegression()

# Entraîner le modèle
model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)
```
## Étape 6 : Évaluer le Modèle

- Calculer l'erreur quadratique moyenne (MSE).
- Calculer le coefficient de détermination (R²).

```python
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Coefficient de détermination (R²):", r2)

```

## Questions

- Question 1 : Ajouter d'autres features de votre choix et vérifier si le modèle s'améliore.
- Question 2 : Normaliser les données avant d'entraîner le modèle et observer l'impact sur la performance.
- Question 3 : Implémenter une validation croisée (cross-validation) et analyser les résultats.
- Question 4 : Calculer le score R² pour l'ensemble d'entraînement et l'ensemble de test. Le modèle est-il surajusté (overfitting) ?
- Question 5 : Implémenter un autre modèle de régression (comme Ridge ou Lasso) et comparer les performances avec le modèle de régression linéaire simple.