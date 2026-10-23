from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def create_models():
    return {
        "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=3000, random_state=42))]),
        "Support Vector Machine": Pipeline([("scaler", StandardScaler()), ("classifier", SVC(kernel="rbf", random_state=42))]),
        "Random Forest": Pipeline([("scaler", StandardScaler()), ("classifier", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))]),
    }

def main():
    print("=" * 60)
    print("HANDWRITTEN CHARACTER RECOGNITION - DIGITS 0 TO 9")
    print("=" * 60)
    digits = load_digits()
    X, y = digits.data, digits.target
    print("\nDataset loaded successfully!")
    print(f"Number of images: {len(X)}")
    print(f"Image dimensions: {digits.images.shape[1]} x {digits.images.shape[2]}")
    print(f"Features per image: {X.shape[1]}")
    print("Classes: digits 0 to 9")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"\nTraining images: {len(X_train)}")
    print(f"Testing images: {len(X_test)}")
    print("Model selection: stratified 5-fold cross-validation on training data")

    models = create_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results, trained_models = [], {}

    for name, model in models.items():
        print("\n" + "-" * 55)
        print(f"Evaluating: {name}")
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        row = {
            "Model": name,
            "Mean CV F1 (Macro)": cv_scores.mean(),
            "Test Accuracy": accuracy_score(y_test, predictions),
            "Test Precision (Macro)": precision_score(y_test, predictions, average="macro", zero_division=0),
            "Test Recall (Macro)": recall_score(y_test, predictions, average="macro", zero_division=0),
            "Test F1 (Macro)": f1_score(y_test, predictions, average="macro", zero_division=0),
        }
        results.append(row)
        trained_models[name] = model
        print("Fold F1 scores:", [round(float(v), 4) for v in cv_scores])
        print(f"Mean CV F1 (Macro): {row['Mean CV F1 (Macro)']:.4f}")
        print(f"Test Accuracy:       {row['Test Accuracy']:.4f}")
        print(f"Test F1 (Macro):     {row['Test F1 (Macro)']:.4f}")

    results_df = pd.DataFrame(results).sort_values("Mean CV F1 (Macro)", ascending=False).reset_index(drop=True)
    csv_path = OUTPUT_DIR / "model_comparison.csv"
    results_df.to_csv(csv_path, index=False)
    print("\nMODEL COMPARISON")
    print(results_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_name]
    model_path = MODEL_DIR / "best_handwritten_character_model.joblib"
    joblib.dump(best_model, model_path)
    print("\nSelected model:", best_name)
    print(f"Mean CV F1 (Macro): {results_df.iloc[0]['Mean CV F1 (Macro)']:.4f}")
    print(f"Test Accuracy: {results_df.iloc[0]['Test Accuracy']:.4f}")
    print("Saved model:", model_path)

    metrics = ["Mean CV F1 (Macro)", "Test Accuracy", "Test Precision (Macro)", "Test Recall (Macro)", "Test F1 (Macro)"]
    ax = results_df.set_index("Model")[metrics].plot(kind="bar", figsize=(12, 6))
    ax.set_title("Handwritten Digit Recognition - Model Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=12)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=300)
    plt.close()

    test_predictions = best_model.predict(X_test)
    display = ConfusionMatrixDisplay.from_predictions(y_test, test_predictions, labels=list(range(10)), cmap="Blues")
    display.ax_.set_title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_test[i].reshape(8, 8), cmap="gray_r")
        pred = int(best_model.predict(X_test[i].reshape(1, -1))[0])
        ax.set_title(f"Pred: {pred}\nActual: {y_test[i]}")
        ax.axis("off")
    fig.suptitle("Sample Handwritten Digit Predictions")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sample_predictions.png", dpi=300)
    plt.close()
    print("\nGenerated CSV, comparison graph, confusion matrix, and sample prediction image.")
    print("PROJECT COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
