# Decision Tree from Scratch

## 📚 Course Overview

This lesson demonstrates how to build a **Decision Tree Classifier** from scratch using only Python and NumPy. The implementation includes support for both categorical and numerical features, using the Gini impurity metric for optimal split selection.

### Learning Objectives
- Understand the fundamentals of decision tree algorithms
- Learn how to calculate Gini impurity for node splitting
- Implement recursive tree building with proper stopping criteria
- Handle both categorical and numerical feature types
- Visualize decision trees with custom plotting functions

---

## 🎯 Prerequisites

### Required Libraries
```bash
pip install numpy matplotlib
```

### Knowledge Requirements
- Basic Python programming
- Understanding of NumPy arrays
- Familiarity with classification concepts
- Basic knowledge of recursion

---

## 🔍 Code Structure

### Core Components

#### 1. **Node Class**
Represents each node in the decision tree:
- Stores data, children nodes, and split information
- Handles both leaf nodes (predictions) and decision nodes (splits)
- Supports numerical thresholds for continuous features

#### 2. **DecisionTreeClassifier Class**
Main classifier implementation:
- **`fit(X, Y)`**: Trains the model by recursively building the tree
- **`find_best_split(node)`**: Finds optimal feature and threshold for splitting
- **`calculate_gini_impurity(Y)`**: Computes Gini impurity for node evaluation
- **`split_on_categorical_feature()`**: Handles categorical feature splits
- **`split_on_numerical_feature()`**: Handles numerical feature splits with threshold selection
- **`predict(X)`**: Traverses the tree to make predictions
- **`plot_tree()`**: Visualizes the decision tree structure

### Key Algorithms

**Gini Impurity Calculation:**
```
Gini = 1 - Σ(p_i²)
```
Where p_i is the probability of class i in the node.

**Splitting Strategy:**
- For categorical features: Split into multiple branches (one per unique value)
- For numerical features: Find optimal threshold using midpoint evaluation
- Select split with minimum weighted Gini impurity

**Stopping Criteria:**
- Pure node (Gini impurity = 0)
- No features left to split
- No valid split found

---

## 🚀 Usage Example

The `main.py` demonstrates a medical diagnosis scenario:

```python
from tree import DecisionTreeClassifier
import numpy as np

# Medical symptom data (Cough, Smell, Temperature, Diagnosis)
data = np.array([
    ["Coughing", "Can Smell", 39, 0],
    ["Coughing", "Cannot Smell", 37.8, 1],
    # ... more samples
])

X, Y = data[:, :-1], data[:, -1]

# Train model with mixed feature types
model = DecisionTreeClassifier(
    feature_types=["categorical", "categorical", "numerical"]
)
model.fit(X, Y)

# Make predictions
predictions = model.predict(test_samples)

# Visualize the tree
model.plot_tree(
    feature_names=["Cough Status", "Smell Ability", "Body Temperature"],
    class_names={0: "Negative", 1: "Positive"},
    save_path="decision_tree.png"
)
```

---

## 📊 Visualization Features

The `plot_tree()` method creates professional tree diagrams with:
- **Blue boxes**: Decision nodes showing split conditions
- **Green/Red boxes**: Leaf nodes showing predictions
- **Edge labels**: Split values or conditions
- **Node information**: Sample counts, class distribution, Gini impurity

---

## 🎓 Key Takeaways

1. **Recursive Structure**: Decision trees are built recursively by finding optimal splits
2. **Impurity Metrics**: Gini impurity guides the splitting process
3. **Mixed Data Types**: The implementation handles both categorical and numerical features
4. **Overfitting Prevention**: Stopping criteria prevent excessive tree depth
5. **Interpretability**: Tree visualization makes the model's decisions transparent

---

## 📝 Additional Files

- **`test_iris.py`**: Tests the classifier on the Iris dataset
- **`test_iris_sklearn.py`**: Compares implementation with scikit-learn

---

## 🔗 Resources

For a detailed video explanation of this implementation, check out the [Noktali Virgul YouTube channel](https://www.youtube.com/@noktalıvirgul).
