''' Ridwan Ishola
    Thu 26/12/2019
    ###  This program uses linear regression ###
    linear prediction/number prediction for students grades
    based on either their first grade, (G1), second grade (G2)
    or their third grade(G3). Predictions can also be made based
    on their study time, previous failures (in tests) or absences.
    The models are then saved in a txt file and also visually presented
    in graph form'''

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

word = input("What would you like to get data presented on?\n"
             "'G1', 'G2', 'G3', 'studytime', 'failures', 'absences':\n")

data = pd.read_csv("student-mat.csv", sep=";")
data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

predict = 'G3'

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
'''
best = 0
for _ in range(60):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    #print(acc)
    if acc > best:
        best = acc
        with open('studentmodel.pickle', 'wb') as f:
            pickle.dump(linear, f)
print('final acc is: ', best)'''

pickle_in = open('studentmodel.pickle', 'rb')
linear = pickle.load(pickle_in)
print("Coefficient: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(round(predictions[x]), x_test[x], y_test[x])

p = word
style.use('ggplot')
pyplot.scatter(data[p], data['G3'])
pyplot.xlabel(p)
pyplot.ylabel('Final Grade')
pyplot.show()











#https://www.youtube.com/watch?v=WFr2WgN9_xE