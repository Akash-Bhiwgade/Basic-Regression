import numpy as np
import pandas as pd
from sklearn import preprocessing, linear_model, svm, model_selection

df = pd.read_csv('toy_data_regression - Sheet1 (1).csv')

X = np.array(df[['experience', 'age']])
y = np.array(df['salary per month'])

X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

reg = linear_model.LinearRegression()
#reg = svm.SVR(kernel='poly')

reg.fit(X_train, y_train)
accuracy = reg.score(X_test, y_test)


prediction = reg.predict([[12, 35]])