from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split as tts

wine = datasets.load_wine()

features = wine.data
labels = wine.target

clf = svm.SVC(kernel='linear', C=1)

# test size means 20% of the data is for test
X_train, X_test, y_train, y_test = tts(features, labels, test_size=0.2)

print(len(y_train))

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
score = clf.score(X_test, y_test)

print(score)



