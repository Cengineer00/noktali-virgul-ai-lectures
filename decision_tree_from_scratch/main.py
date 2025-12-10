from tree import DecisionTreeClassifier
import numpy as np

data = np.array(
    [
        ["Öksürüyor", "Koku Alabiliyor", 39, 0],
        ["Öksürüyor", "Koku Alamıyor", 37.8, 1],
        ["Öksürmüyor", "Koku Alabiliyor", 38.4, 0],
        ["Öksürmüyor", "Koku Alamıyor", 36.7, 1],
        ["Öksürüyor", "Koku Alabiliyor", 38.5, 0],
        ["Öksürüyor", "Koku Alamıyor", 38.9, 1],
        ["Öksürmüyor", "Koku Alabiliyor", 37.3, 1],
    ]
)

X, Y = data[:, :-1], data[:, -1]
Y = Y.reshape((-1, 1))

model = DecisionTreeClassifier(feature_types=["categorical", "categorical", "numerical"])
model.fit(X, Y)

preds = model.predict(
    [
        ["Öksürüyor", "Koku Alabiliyor",36.5],
        ["Öksürmüyor", "Koku Alamıyor", 38],
        ["Öksürüyor", "Koku Alabiliyor", 39.5],
    ]
)

# Visualize the tree
print("\nGenerating graphical plot...")
feature_names = ["Öksürük Durumu", "Koku Alma", "Vücut Sıcaklığı"]
class_names = {0: "Negatif", 1: "Pozitif"}
model.plot_tree(feature_names, class_names, save_path="decision_tree.png")

print("\nPredictions:")
for i, pred in enumerate(preds):
    pred_label = class_names.get(int(pred), pred)
    print(f"  Sample {i+1}: {pred_label}")
