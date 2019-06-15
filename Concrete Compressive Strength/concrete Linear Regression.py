import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
df=pd.read_csv("../DATASET/Concrete_Data.csv")
print(df.isnull().sum())

df.columns=['cement','blast','flyash','water','superplasticiser','coarse','fine','age','ccs']
col=list(df.columns)
print(col)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#X = df.drop(['ccs'],axis =1)
X=df.iloc[:,[1,5,8]].values
Y = df['ccs']
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,random_state=101)
corr = df.corr()
sns.heatmap(corr,annot=True)
plt.show()

model=LinearRegression()
model.fit(X_train,Y_train)
pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error
from math import sqrt


rmse = sqrt(mean_squared_error(Y_test,pred))
print('rmse:',rmse)

print('Abs model coefficent',abs(model.coef_))
print('B0',model.intercept_)
print('R-squared error Regression model',r2_score(Y_test, pred))
