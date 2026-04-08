import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model_path='mental_health_model.pkl', label_encoder_path='label_encoder.pkl', csv_path='Mental_health_dset.csv'):
    """Evaluate the trained model on a new dataset."""
    import joblib
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    # Load the trained model and label encoder
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)

    # Load the evaluation dataset
    df = pd.read_csv(csv_path)

    # Extract features and labels, handling potential missing 'Disorder' column
    feature_columns = [c for c in df.columns if c != 'Disorder']
    X = df[feature_columns].apply(lambda x: (x == 'yes').astype(int))

    # Check if 'Disorder' column exists for evaluation
    if 'Disorder' in df.columns:
        y = df['Disorder']
        y_enc = label_encoder.transform(y)

        # Generate predictions
        y_pred = model.predict(X)

        # Evaluate the model
        accuracy = accuracy_score(y_enc, y_pred)
        report = classification_report(y_enc, y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:\n", report)
        return accuracy, report
    else:
        print("Warning: 'Disorder' column not found in the evaluation dataset.")
        # If no labels, you can still generate predictions but cannot evaluate accuracy.
        y_pred = model.predict(X)
        predicted_disorders = label_encoder.inverse_transform(y_pred)
        print("Predicted Disorders:\n", predicted_disorders)
        return None, None

# Example usage:
if __name__ == '__main__':
    accuracy, report = evaluate_model()
