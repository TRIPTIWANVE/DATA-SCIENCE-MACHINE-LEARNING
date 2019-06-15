import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("../DATASET/CLV history.csv")
print(df.isnull().sum())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#X = df.drop(['CUST_ID','CLV'],axis =1)
X=df.iloc[:,1:7].values
Y = df['CLV']
print(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,random_state=0)
corr = df.corr()
sns.heatmap(corr,annot=True)
plt.show()

model=LinearRegression()
model.fit(X_train,Y_train)
pred_reg = model.predict(X_test)
print(pred_reg)
from sklearn.metrics import mean_squared_error
#from math import sqrt

rmse = np.sqrt(mean_squared_error(Y_test,pred_reg))
print('rmse:',rmse)
print(abs(model.coef_))
print('B0',model.intercept_)

print('Variance score: {}'.format(model.score(X_test, Y_test)))
r2_score = model.score(X_train, Y_train)
print("r2_Score", r2_score)
