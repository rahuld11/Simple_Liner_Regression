import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("G:/Assignments/Simple linear regression/Data sets for practice/Salary_Data.csv")

x = data.iloc[:, :-1].values
y = data.iloc[:, :1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0, test_size=1/3) 

linear_regressor = LinearRegression()
linear_regressor.fit(x_train,y_train)

y_pred = linear_regressor.predict(x_test)

plt.scatter(x_train, y_train, color = "red")

plt.plot(x_train, linear_regressor.predict(x_train), color = "blue")

plt.title("salary vs Experience (Training set)")
plt.xlabel("Years of experence")
plt.ylabel("salary")
plt.show()

plt.scatter(x_test, y_test, color = "red")

plt.plot(x_train, linear_regressor.predict(x_train), color = "blue")

plt.title("salary vs Experience (Test set)")
plt.xlabel("Years of experence")
plt.ylabel("salary")
plt.show()