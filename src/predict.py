from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "best_handwritten_character_model.joblib"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run src/main.py first.")
    model = joblib.load(MODEL_PATH)
    digits = load_digits()
    _, X_test, _, y_test = train_test_split(digits.data, digits.target, test_size=0.20, random_state=42, stratify=digits.target)
    sample = X_test[0].reshape(1, -1)
    predicted = int(model.predict(sample)[0])
    actual = int(y_test[0])
    print("=" * 50)
    print("HANDWRITTEN DIGIT PREDICTION")
    print("=" * 50)
    print(f"Predicted digit: {predicted}")
    print(f"Actual digit:    {actual}")
    print("Result:", "Correct prediction" if predicted == actual else "Prediction differs from actual label")
    plt.figure(figsize=(4, 4))
    plt.imshow(sample.reshape(8, 8), cmap="gray_r")
    plt.title(f"Predicted: {predicted} | Actual: {actual}")
    plt.axis("off")
    output_path = OUTPUT_DIR / "single_prediction.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print("Prediction image saved to:", output_path)

if __name__ == "__main__":
    main()
