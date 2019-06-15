#IMPORT LABRARIES

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#IMPORT DATASET
data=pd.read_csv('../DATASET/Boston train.csv')
x=data.iloc[:,[7,14]].values
y=data['medv']

#SPLITING DATA INTO TRAINING and TESTING SET
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=0)

# APPLYING MODELS

#LINEAR REGRESSION
model=LinearRegression()
model.fit(x_train,y_train)
pred_reg = model.predict(x_test)

#LASSO REGRESSION

lasso = Lasso(alpha=0)
lasso.fit(x_train,y_train)
pred_lasso = model.predict(x_test)

# RIDGE REGRESSION

ridge=Ridge(alpha=0.01)
ridge.fit(x_train,y_train)
pred_ridge=ridge.predict(x_test)

# Coefficents 
print("regression coef",abs(model.coef_))
print("regression coef",abs(ridge.coef_))
print("regression coef",abs(lasso.coef_))

# Intercepts
print('This is the value of y intercept in regrn',model.intercept_)
print('This is the value of y intercept in ridge',ridge.intercept_)
print('This is the value of y intercept in Lasso',lasso.intercept_)

# RMSE
print('This is our RMSE in regrn',np.sqrt(mean_squared_error(y_test,pred_reg)))
print('This is our RMSE in ridge',np.sqrt(mean_squared_error(y_test,pred_ridge)))
print('This is our RMSE in Lasso',np.sqrt(mean_squared_error(y_test,pred_lasso)))

# Mean Absolute Error
print('Mean Absolute Error Regression model',mean_absolute_error(y_test, pred_reg))
print('Mean Absolute Error Ridge Model',mean_absolute_error(y_test, pred_ridge))
print('Mean Absolute Error LASSO model',mean_absolute_error(y_test, pred_lasso))

# R-squared
print('R-squared error Regression model',r2_score(y_test, pred_reg))
print('R-squared error Ridge Model',r2_score(y_test, pred_ridge))
print('R-squared error LASSO model',r2_score(y_test, pred_lasso))


# Visusalize the Training set results
plt.scatter(y_test,pred_reg,c='ORANGE')
plt.plot(y_test,model.predict(x_test),c='blue')
plt.title('BOSTON [MODEL]      ')
plt.xlabel('-------- X ------>')
plt.ylabel('-------- Y ------>')
plt.show()

# Visusalize the Training set results
plt.scatter(y_test,pred_ridge,c='red')
plt.plot(y_test,ridge.predict(x_test),c='blue')
plt.title('BOSTON [RIDGE MODEL]      ')
plt.xlabel('-------- X ------>')
plt.ylabel('-------- Y ------>')
plt.show()

#Visusalize the Training set results
plt.scatter(y_test,pred_lasso,c='GREEN')
plt.plot(y_test,lasso.predict(x_test),c='blue')
plt.title('BOSTON  [LASSO MODEL]      ')
plt.xlabel('-------- X ------>')
plt.ylabel('-------- Y ------>')
plt.show()


