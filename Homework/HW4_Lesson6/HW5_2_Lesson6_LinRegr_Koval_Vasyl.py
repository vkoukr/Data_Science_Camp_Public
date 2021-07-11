import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 

# Load the diabetes dataset
from sklearn import datasets
#from sklearn.datasets import load_boston
X, y = datasets.load_diabetes(return_X_y=True)
#X, y = datasets.load_boston(return_X_y=True)
#from sklearn.datasets import load_boston
#from sklearn.metrics import mean_squared_error, r2_score


#---- Split to train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=  train_test_split(X, y, train_size=0.8, random_state=2021, shuffle=True)

#---- Normalization
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled= scaler.fit_transform(X_train)
X_test_scaled= scaler.transform(X_test)

#---- Linear Regression
from sklearn.linear_model import LinearRegression
#lin_reg=LinearRegression(normalize=False,)
lin_reg=LinearRegression()
lin_reg.fit(X_train_scaled,y_train)
#lin_reg.fit(X_train,y_train)
regressor = lin_reg
print ('\n ====Linear Regression====')
print ('R2 train score =', regressor.score(X_train_scaled, y_train))
print ('R2 test score =', regressor.score(X_test_scaled, y_test))
#print ('R2 train score =', regressor.score(X_train, y_train))
#print ('R2 test score =', regressor.score(X_test, y_test))
print ('b: {}, \nw= {}'.format(regressor.intercept_, regressor.coef_)) 

plt.figure()
y_line= lin_reg.predict(X_test_scaled)
#plt.plot((y_test-y_line), label='Absolute Differences in Predictions of Diabetes Data for Testing SET', color= 'b')
plt.plot(y_test, label='Diabetes Data for Testing SET', color= 'b')
plt.plot(y_line, label='Predicted Data for Testing SET', color= 'r')
plt.title('Predictions of Diabetes Data for Testing SET using "Linear Regression"')
plt.legend() 
plt.show() 

# x_line=X_train[:,0]
# fig=plt.figure()
# plt.hold(True)
# #ax = fig.gca(projection='3d')
# ax=fig.add_subplot(111)
# y_line=range(0,len(X_train))*regressor.coef_[0]+regressor.intercept_
# plt.plot(range(0,len(X_train)), y_line, '-r')
# plt.plot(x_line,y_train, 'o')
# plt.scatter(x_line,y_train)
# plt.show()

from sklearn.linear_model import Ridge
ridge_reg=Ridge()
ridge_reg.fit(X_train_scaled,y_train)
regressor = ridge_reg
print ('\n ====Ridge====')
print ('R2 train score =', regressor.score(X_train_scaled, y_train))
print ('R2 test score =', regressor.score(X_test_scaled, y_test))
print ('b: {}, \nw= {}'.format(regressor.intercept_, regressor.coef_)) 

plt.figure()
y_line= ridge_reg.predict(X_test_scaled)
#plt.plot((y_test-y_line), label='Absolute Differences in Predictions of Diabetes Data for Testing SET', color= 'b')
plt.plot(y_test, label='Diabetes Data for Testing SET', color= 'b')
plt.plot(y_line, label='Preded Data for Testing SET', color= 'r')
plt.title('Accuracy of Predictions of Diabetes Data for Testing SET using "Ridge Regularizarion"')
plt.legend() 
plt.show() 

from sklearn.linear_model import Lasso
lasso_reg=Lasso()
lasso_reg.fit(X_train_scaled,y_train)
regressor = lasso_reg
print ('\n ====Lasso====')
print ('R2 train score =', regressor.score(X_train_scaled, y_train))
print ('R2 test score =', regressor.score(X_test_scaled, y_test))
print ('b: {}, \nw= {}'.format(regressor.intercept_, regressor.coef_)) 

plt.figure()
y_line= lasso_reg.predict(X_test_scaled)
#plt.plot((y_test-y_line), label='Absolute Differences in Predictions of Diabetes Data for Testing SET', color= 'b')
plt.plot(y_test, label='Diabetes Data for Testing SET', color= 'b')
plt.plot(y_line, label='Preded Data for Testing SET', color= 'r')
plt.title('Accuracy of Predictions of Diabetes Data for Testing SET using "Lasso"')
plt.legend() 
plt.show()



from sklearn.preprocessing import PolynomialFeatures
poly= PolynomialFeatures(degree=2,include_bias=False) # default is True means to return the first feature of all 1 as for degree 0 
X_train_poly= poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
print ('\n ====Polynomial + Linear Regression with Normalization (Degree 2)==== ')
print ('X_train.shape= ',X_train.shape)
print ('X_train_poly.shape= ',X_train_poly.shape)
# # X_train_poly[:5]

#---- Normalization
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_poly_scaled = scaler.fit_transform(X_train_poly)
X_test_poly_scaled= scaler.transform(X_test_poly)

poly_lin_reg = LinearRegression().fit (X_train_poly_scaled,y_train)
regressor = poly_lin_reg
#print ('\n ====Polynomial + Linear Regression with Normalization (Degree 2)==== ')
print ('R2 train score =', regressor.score(X_train_poly_scaled, y_train))
print ('R2 test score =', regressor.score(X_test_poly_scaled, y_test))
print ('b: {}, \nw= {}'.format(regressor.intercept_, regressor.coef_)) 

print ('\n ====Polynomial + Ridge with Normalization (Degree 2, alpha=0.001)==== ')
print ('X_train.shape= ',X_train_scaled.shape)
print ('X_train_poly.shape= ',X_train_poly_scaled.shape)
poly_ridge = Ridge(alpha=0.001, max_iter=1e5).fit (X_train_poly_scaled,y_train) # Increased max-iter and alpha
regressor = poly_ridge
print ('R2 train score =', regressor.score(X_train_poly_scaled, y_train))
print ('R2 test score =', regressor.score(X_test_poly_scaled, y_test))
w= regressor.coef_
print ('b: {}, \nw= {}'.format(regressor.intercept_, w)) 


print ('\n ====Polynomial + Lasso with Normalization (Degree 2)==== ')
print ('X_train.shape= ',X_train.shape)
print ('X_train_poly.shape= ',X_train_poly.shape)
poly_lasso = Lasso(max_iter=100000).fit (X_train_poly_scaled,y_train)
regressor = poly_lasso
print ('Polynomial + Lasso')
print ('R2 train score =', regressor.score(X_train_poly_scaled, y_train))
print ('R2 test score =', regressor.score(X_test_poly_scaled, y_test))

plt.figure()
plt.title('Polynomial Predictions of Diabetes Data for Testing SET')
plt.plot(y_test, label='Original Diabetes Data', color= 'b')
y_line= poly_lin_reg.predict(X_test_poly_scaled)
plt.plot(y_line, label='Polynomial + Linear', color= 'r')
y_line= poly_ridge.predict(X_test_poly_scaled)
plt.plot(y_line, label='Polynomial + Ridge', color= 'g')
y_line= poly_lasso.predict(X_test_poly_scaled)
plt.plot(y_line, label='Polynomial + Lasso', color= 'c')
plt.legend() 
plt.show()