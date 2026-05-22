# Mouse Brain Genotype Classifier 

A machine learning pipeline that classifies mouse brain genotypes (Control vs Trisomic) based on protein expression levels from cortex samples. Generates synthetic test data using GPT-3.5-turbo.

## What it does

- Loads and preprocesses cortex protein expression data (77 features)
- Trains and compares three classifiers: Decision Tree, Random Forest, Gradient Boosting
- Selects the best model based on F1-score
- Uses Recursive Feature Elimination (RFE) to select the 30 most important features
- Generates a synthetic mouse brain sample via ChatGPT and predicts its genotype
- Outputs a confusion matrix and classification report

## Tech Stack

- Python 3
- scikit-learn
- pandas, numpy
- matplotlib
- OpenAI API (GPT-3.5-turbo)

## Output

- Console: model scores, cross-validation metrics, best model, prediction for generated sample
- `confusion_matrix.png`

## Models

| Model | Tuning |
|-------|--------|
| Decision Tree | GridSearchCV (depth, splits, leaves) |
| Random Forest | GridSearchCV (estimators, depth, splits) |
| Gradient Boosting | Fixed learning rate 0.1 |

Best model selected automatically by macro F1-score on the test set.
