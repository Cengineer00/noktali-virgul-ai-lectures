from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
#from data import data

from sklearn.datasets import load_iris

iris = load_iris()
# split the data to train and test sets


X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

#X, Y = data[:, :-1], data[:, -1]
Y_train = Y_train.reshape((-1, 1))
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)

preds = model.predict(X_test)
print("\nPredictions:")

# calculate accuracy
accuracy = (preds.flatten() == Y_test).mean()
print(f"Accuracy: {accuracy * 100:.2f}%")