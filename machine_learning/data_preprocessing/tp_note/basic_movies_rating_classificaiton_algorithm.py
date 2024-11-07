import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import accuracy_score, classification_report


def classification_algorithm(csv_file_path):
    # Step 1: Load and preprocess the data
    df = pd.read_csv(csv_file_path)

    print(df.head())

    df['VOTES'] = df['VOTES'].str.replace(',', '').astype(int)
    df['RunTime'] = df['RunTime'].astype(int)

    # Step 2: Remove the 'ONE-LINE' column as it's not being used for prediction
    X = df.drop(['ONE-LINE', 'RATING', 'MOVIES', 'YEAR', 'GENRE', 'STARS', 'Gross'], axis=1)
    y = df['RATING']

    # Step 3: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Create and train the Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Step 5: Make predictions on the test set
    y_pred = rf_classifier.predict(X_test)

    # Step 6: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Step 7: Feature importance
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf_classifier.feature_importances_})
    print("\nFeature Importance:")
    print(feature_importance.sort_values('importance', ascending=False))


if __name__ == '__main__':
    classification_algorithm('path_to_csv_file')
