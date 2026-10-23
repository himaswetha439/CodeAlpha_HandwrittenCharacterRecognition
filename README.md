# Handwritten Character Recognition

## 1. Project Overview
This project recognizes handwritten digits from 0 to 9 using machine learning. It compares Logistic Regression, Support Vector Machine (SVM), and Random Forest, then saves the model selected using cross-validation.

For a lightweight and reproducible beginner project, it uses scikit-learn's built-in handwritten digits dataset. Each image is 8 x 8 pixels and is represented by 64 numerical pixel features. No manual dataset download is required.

## 2. Objectives
- Load and inspect handwritten digit images.
- Compare three classification algorithms.
- Use stratified 5-fold cross-validation for model selection.
- Evaluate models on a separate test set.
- Save evaluation tables, graphs, a confusion matrix, and sample predictions.
- Save the selected model for future predictions.

## 3. Dataset
The project uses `sklearn.datasets.load_digits`.
- Number of images: 1,797
- Image size: 8 x 8 pixels
- Features per image: 64
- Classes: digits 0 through 9

Dataset documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html

## 4. Technologies
Python, scikit-learn, Pandas, Matplotlib, and Joblib.

## 5. Methodology
1. Load the built-in handwritten digits dataset.
2. Split it into 80% training data and 20% test data using stratification.
3. Evaluate Logistic Regression, SVM, and Random Forest using stratified 5-fold cross-validation on training data.
4. Select the model with the highest mean cross-validation macro F1-score.
5. Evaluate models on the held-out test set using Accuracy, macro Precision, macro Recall, and macro F1-score.
6. Save the comparison CSV, model comparison chart, confusion matrix, sample prediction grid, and trained model.

Macro-averaged metrics give each digit class equal weight.

## 6. Project Structure
```text
2_Handwritten_Character_Recognition/
├── data/
├── models/
│   └── best_handwritten_character_model.joblib
├── outputs/
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── sample_predictions.png
│   └── single_prediction.png
├── src/
│   ├── main.py
│   └── predict.py
├── .gitignore
├── README.md
└── requirements.txt
```

## 7. Installation and Execution
Run commands from inside the `2_Handwritten_Character_Recognition` directory.

Install dependencies:
```bash
python -m pip install -r requirements.txt
```

Train and evaluate:
```bash
python src/main.py
```

Run a sample prediction after training:
```bash
python src/predict.py
```

## 8. Generated Outputs
- `model_comparison.csv` — cross-validation and test metrics.
- `model_comparison.png` — model comparison chart.
- `confusion_matrix.png` — confusion matrix for the selected model.
- `sample_predictions.png` — ten test images with predicted and actual labels.
- `single_prediction.png` — image for a single sample prediction.
- `best_handwritten_character_model.joblib` — saved model pipeline.

## 9. Limitations
This project recognizes handwritten digits (0–9), not alphabetic letters. The dataset is small and contains 8 x 8 pixel images, so performance may differ on real-world photographs, scanned forms, or larger images. This is an educational project, not a production-ready OCR system.

## 10. Conclusion
This project demonstrates a machine learning workflow for handwritten digit recognition, including preprocessing, cross-validation, model comparison, test evaluation, visualization, and model saving.
