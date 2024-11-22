# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary libraries: We'll need libraries like scikit-learn for the machine learning model and dataset, and numpy for numerical operations.
2.Load the Iris dataset: We can easily load the Iris dataset using sklearn.datasets.load_iris().
3.Split the dataset: We'll divide the data into training and testing sets using train_test_split.
4.Train the model: Use SGDClassifier to train the model on the training data.
5.Evaluate the model: Test the model on the testing data and check its accuracy.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Karthick Raja K
RegisterNumber:  212223240066
*/

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()

df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
df['target'] = iris.target

print(df.head())

x = df.iloc[:, :-1]
y = df['target']
# print("X :")
# print(x)
# print("Y :")
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/4, random_state = 42)

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

sgd_clf.fit(x_train, y_train)

y_pred = sgd_clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy :",accuracy)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix : \n",cm)
```

## Output:

![image](https://github.com/user-attachments/assets/65f9a673-ce7d-4e59-92ba-fcf4193f76ba)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
