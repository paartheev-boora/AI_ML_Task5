# Heart Disease Prediction using Decision Trees and Random Forests

This project involves building machine learning models to predict heart disease from patient data using **Decision Tree Classifiers** and **Random Forest Classifiers**. The analysis is conducted on a dataset of 1025 patient records, each containing 13 medical attributes and one target label.

---

## 📊 Dataset

- Loaded from: `heart.csv`
- Records: **1025**
- Features: `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`, `target`

The `target` column indicates the presence (1) or absence (0) of heart disease.

---

## 🌳 1. Decision Tree Classifier (DTC)

### 🧠 Training and Visualization

- **Library**: `sklearn.tree.DecisionTreeClassifier`
- **Split**: 80/20 train-test split
- **Model**: Trained on all features to predict the `target`.

**Tree Characteristics**:
- **Depth**: 10  
- **Leaves**: 50

📌 *"The depth 10 depicts that the model is not overfitting. But the number of splits seems to be more."*

The tree was visualized using `plot_tree()` with feature names and class labels colored to indicate the classification outcome.

---

### 📈 Evaluation & Learning Curve

- **Train Accuracy**: 1.00  
- **Test Accuracy**: 0.99

📌 *"High accuracy but potential overfitting indicated by perfect training score."*

A **learning curve** was plotted using `sklearn.model_selection.learning_curve`, confirming this variance.

---

### 🔁 Cross-Validation

- **CV Accuracy**: 0.77 ± 0.05

📌 *"Performance drops during cross-validation, which confirms generalization is weaker."*

---

### ⚙️ Regularization

A second Decision Tree was trained with:
- `max_depth=5`
- `min_samples_leaf=10`

**Tree Characteristics**:
- **Depth**: 5  
- **Leaves**: 21

📌 *"This tree is optimal to consider."*

- **Train Accuracy**: 0.90  
- **Test Accuracy**: 0.82  
- **CV Accuracy**: 0.73 ± 0.09

Learning curves show reduced overfitting and better generalization.

---

### ✂️ Cost Complexity Pruning

Used `cost_complexity_pruning_path()` to prune the tree and trained multiple models for different `ccp_alpha` values.

- Trees were evaluated iteratively on training and test sets.
- The model with:
  - **Depth**: 3  
  - **Leaves**: 4  
  - **Test Accuracy**: 0.78  
  was selected as the **final tree**.

📌 *"This pruned model raises test accuracy by 4% without affecting training accuracy, and thus seems most desirable."*

---

## 🌲 2. Random Forest Classifier

### 🧠 Training

- **Model**: `RandomForestClassifier(n_estimators=100)`
- **Test Accuracy**: 0.9854

📌 *"Outperformed decision trees with higher robustness and generalization."*

---

### 🔍 Hyperparameter Tuning (GridSearchCV)

Performed extensive tuning using `GridSearchCV` over the following:
```python
{
  'n_estimators': [100, 300, 500],
  'max_depth': [10, 20, None],
  'min_samples_split': [2, 5, 10],
  'min_samples_leaf': [1, 2, 4],
  'max_features': ['sqrt', 'log2'],
  'bootstrap': [True, False]
}
```

**Best Parameters**:
- `n_estimators`: 300  
- `max_depth`: 10  
- `min_samples_split`: 2  
- `min_samples_leaf`: 1  
- `max_features`: 'sqrt'  
- `bootstrap`: True

**Best Cross-Validated Accuracy**: 0.9817

---

### 🌲 Best Tree in the Forest

- Evaluated individual decision trees in the ensemble.
- Visualized the best-performing tree (based on test accuracy).
- Max depth shown: 3 (for interpretability)

📌 *"This helps understand internal logic of one of the strong-performing estimators in the forest."*

---

## 🧠 Concepts & Techniques Applied

- Decision Trees & Random Forests
- Train-test split
- Accuracy scoring
- Learning curves
- Cross-validation
- Tree pruning (`ccp_alpha`)
- Grid search with cross-validation for hyperparameter tuning
- Visualization with `plot_tree`

---

## 📌 Key Insights

- Unpruned Decision Trees overfit despite high accuracy.
- Regularization and pruning improved generalization.
- A pruned tree with just 4 leaves achieved 78% test accuracy.
- Random Forests provided robust and superior performance (~98.5%).
- Hyperparameter tuning further optimized model strength.
- Visualizing trees helped interpret model behavior.

---

## ✅ Conclusion

This project demonstrates the application of decision trees and ensemble models to a real-world health dataset. While individual trees offer interpretability, ensemble methods like Random Forests achieve superior performance. Pruning and hyperparameter tuning significantly affect accuracy and generalization, highlighting the importance of model refinement.
