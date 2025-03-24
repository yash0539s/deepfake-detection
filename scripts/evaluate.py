import tensorflow as tf
import numpy as np
import os
import logging
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score, f1_score

# Setup logging
logging.basicConfig(filename="error_log.txt", level=logging.ERROR, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# File paths
MODEL_PATH = "D:\\deep_fake_hacthon\\deepfake_detector.h5"
X_TEST_PATH = 'data/processed/balanced_data.npy'
Y_TEST_PATH = 'data/processed/balanced_labels.npy'

# Load model
def load_model():
    try:
        if os.path.exists(MODEL_PATH):
            print("[INFO] Loading model...")
            return tf.keras.models.load_model(MODEL_PATH, compile=False)
        else:
            raise FileNotFoundError("Model file not found!")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        print(f"[ERROR] Could not load model: {e}")
        exit(1)

# Load test data
def load_test_data():
    try:
        X_test = np.load(X_TEST_PATH)
        y_test = np.load(Y_TEST_PATH)
        X_test = X_test / 255.0  # Normalize
        return X_test, y_test
    except Exception as e:
        logging.error(f"Error loading test data: {str(e)}")
        print(f"[ERROR] Could not load test data: {e}")
        exit(1)

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(np.int32)

    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    avg_precision = average_precision_score(y_test, y_pred_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, zero_division=1)

    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"Average Precision Score: {avg_precision:.4f}")
    print("\nClassification Report:\n", class_report)
    print("\nConfusion Matrix:\n", conf_matrix)

    # Save report
    with open("evaluation_report.txt", "w") as f:
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write(f"Average Precision Score: {avg_precision:.4f}\n")
        f.write("Classification Report:\n" + class_report + "\n")
        f.write("Confusion Matrix:\n" + str(conf_matrix) + "\n")

    print("\n[INFO] Evaluation report saved to 'evaluation_report.txt'.")

# Main execution
if __name__ == "__main__":
    model = load_model()
    X_test, y_test = load_test_data()
    evaluate_model(model, X_test, y_test)
    print("\n Model Evaluation Completed Successfully!")