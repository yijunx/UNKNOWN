import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib import style

style.use('ggplot')

x = [1, 5, 1.5, 8, 1, 9]
y = [2, 8, 1.8, 8, 0.6, 11]

X = np.array([x, y]).T
print(f'shape of x is {X.shape}')

y = [0, 1, 0, 1, 0, 1]

# clf = classifier
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y)

print(clf.predict([[0.5, 0.5]]))

w = clf.coef_[0]
print(w)

