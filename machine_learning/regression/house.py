import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_data(chart=False, info=False):
    # Charger le dataset
    df = pd.read_csv('../../data/Housing.csv')  # Assurez-vous d'avoir le fichier "house_price.csv"

    # Afficher les premières lignes du dataset
    if info:
        print(df.head())

        # Informations générales sur le dataset
        print(df.info())

        # Statistiques descriptives
        print(df.describe())

        # Vérification des valeurs manquantes
        print(df.isnull().sum())

    # Visualisation de la relation entre les variables
    if chart:
        sns.pairplot(df)
        plt.show()

    return df

def preprocess_data(df):
    lisst = ["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea"]
    for i in lisst:
        df[i] = df[i].map({'yes': 1, 'no': 0})
    df['furnishingstatus'] = df['furnishingstatus'].map({'semi-furnished': 1, 'unfurnished': 0, 'furnished': 2})

    return df

def predict_price(df):
    # Selecting features and target variable
    X = df[[
        "area",
        "stories",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "parking",
        "prefarea",
    ]]

    y = df['price']

    # Handle missing values in the target variable
    X = X[~y.isna()]
    y = y[~y.isna()]

    # Split the data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Créer le modèle
    model = LinearRegression()

    # Entraîner le modèle
    model.fit(X_train, y_train)

    # Prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error (MSE):", mse)
    print("Coefficient de détermination (R²):", r2)


if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df)
    predict_price(df)