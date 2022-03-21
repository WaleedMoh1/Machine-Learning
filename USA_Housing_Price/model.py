#import library 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error , mean_absolute_error , mean_squared_error , r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures

#--------------------------------------------------------------------------
#PART 1 Data Preprocssing 
#--------------------------------------------------------------------------

#read data from outside excel sheet
dataset = pd.read_csv("USA_Housing.csv")


#split features into x and target into y
x = dataset.iloc[: , 0:5]
y = dataset.iloc[: , 5:6]


#Get more information about featrues and target
x.info()
y.info()
x.head()
y.head()
describe_features = x.describe()


#check if there are missing value or not
print(x.isnull().sum())


#missing value processing
misvalue = SimpleImputer(missing_values=np.nan , strategy="mean")
x = misvalue.fit_transform(x)
x = pd.DataFrame(x)
print(x.isnull().sum())


#method to visualization
def compareByPricePlt(labelN , labelT):
    plt.scatter(x[labelN] , y , color = 'red')
    plt.xlabel(labelT)
    plt.ylabel('Price')
    plt.show()
    
    
compareByPricePlt(0, 'Avg. Area Income')
compareByPricePlt(1, 'Avg. Area House Age')
compareByPricePlt(2, 'Avg. Area Number of Rooms')
compareByPricePlt(3, 'Avg. Area Number of Bedrooms')
compareByPricePlt(4, 'Area Population')

#--------------------------------------------------------------------------
#Part 2 Training model and evalution 
#--------------------------------------------------------------------------

#split data into training set and testing set
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.3 , random_state=45)


#method to evalution model
def evalutionModel(y_pred , model , x_train):
    MAPE = mean_absolute_percentage_error(y_test, y_pred)
    MAE = mean_absolute_error(y_test , y_pred)
    MSE = mean_squared_error(y_test , y_pred)
    r2 = r2_score(y_test, y_pred)
    train = model.score(x_train , y_train)
    print('MAPE: ',MAPE)
    print('MAE' , MAE)
    print('MSE' , MSE)
    print('R2: ' , r2)
    print('training: ' , train)


#building linear regression model 
model_linear_regression = LinearRegression()
model_linear_regression.fit(x_train , y_train)
y_pred = model_linear_regression.predict(x_test)


#evalute linear model
evalutionModel(y_pred , model_linear_regression , x_train)


#builing polynomial 
poly = PolynomialFeatures(degree= 4)
x_poly = poly.fit_transform(x_train)
x_tpoly = poly.fit_transform(x_test)
model_Poly = LinearRegression()
model_Poly.fit(x_poly , y_train)
y_pred2 = model_Poly.predict(x_tpoly)


#evalute poly model
evalutionModel(y_pred2 , model_Poly , x_poly)