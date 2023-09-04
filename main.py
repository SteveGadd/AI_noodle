import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('seattle-weather.csv')
X = df.drop(columns="weather")
Y = df['weather']
for i in range(len(X)):
    date = X.iloc[i, 0]
    month = date[5:7]
    if int(month) == 1 or int(month) == 2 or int(month) == 12:
        X.iloc[i, 0] = 0
    elif int(month) == 3 or int(month) == 4 or int(month) == 5:
        X.iloc[i, 0] = 1
    elif int(month) == 6 or int(month) == 7 or int(month) == 8:
        X.iloc[i, 0] = 2
    else:
        X.iloc[i, 0] = 3
X.rename(columns={'date': 'season'}, inplace=True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.99)


model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)
print(accuracy)

tree.export_graphviz(model, out_file='ai-project.dot',
                     feature_names=list(X.columns),
                     class_names=sorted(Y.unique()), label='all', rounded=True, filled=True)
