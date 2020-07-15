# For reading data set
# importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# reading a csv file using pandas library
saldata=pd.read_csv("G:\\Assignments\\Simple linear regression\\Data sets for practice\\Salary_Data.csv")
saldata.columns
saldata.describe()

plt.hist(saldata.Salary)
plt.boxplot(saldata.Salary)

plt.hist(saldata.YearsExperience)
plt.boxplot(saldata.YearsExperience)

saldata.Salary.corr(saldata.YearsExperience)# # correlation value between X and Y
np.corrcoef(saldata.Salary,saldata.YearsExperience)

# For preparing linear regression model we need to import the statsmodels.formula.api
import statsmodels.formula.api as smf
model=smf.ols("Salary~YearsExperience",data=saldata).fit()

# For getting coefficients of the varibles used in equation
model.params

# P-values for the variables and R-squared value for prepared model
model.summary()

model.conf_int(0.05)# 95% confidence interval

pred=model.predict(saldata.iloc[:,0]) # Predicted values of Salary using the model
pred

error=saldata.Salary-pred
error

rmse=np.sqrt(np.mean(error)**2)
rmse

pred.corr(saldata.Salary)#0.97 Salary~YearsExperience

import matplotlib.pyplot as plt
plt.scatter(x=saldata["YearsExperience"],y=saldata["Salary"],color='red');plt.plot(saldata["YearsExperience"],pred,color='Black');plt.xlabel("YearsExperience");plt.ylabel("Salary")
