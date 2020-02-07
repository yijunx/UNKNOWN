# this is the iris flower example

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()

features = iris.data.T

sepal_length = features[0]
sepal_width = features[1]
petal_length = features[2]
petal_width = features[3]

sepal_length_label = iris.feature_names[0]
sepal_width_label = iris.feature_names[1]
petal_length_label = iris.feature_names[2]
petal_width_label = iris.feature_names[3]

# print(iris.target)

# now lets do an initial plot
# plt.scatter(sepal_length, sepal_width, c=iris.target)
# plt.xlabel(sepal_length_label)
# plt.ylabel(sepal_width_label)
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)
X_new = np.array([[5, 2.9, 1, 0.2]])

prediction = knn.predict(X_new)

print(knn.score(X_test, y_test))
