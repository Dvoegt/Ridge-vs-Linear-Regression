import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
import seaborn as sns



# PAART 1:  Load the dataset
#manually make column names and then set them as column titles whilst skipping first line of data holding the columns
column_names = ['index',  'logCancerVol',  'logCancerWeight',  'age',  'logBenighHP',  'svi',  'logCP',  'gleasonScore',  'gleasonS45',  'levelCancerAntigen',  'train']
data = pd.read_csv('Linear vs Regression/Task1_RegressionTask_CancerData.txt', sep='\t', skiprows=1, names=column_names)




#PART 2: Cleaning the data

#cleaning data with null values
data = data.dropna()  #removes rows with empty cells(doing this instead of replacing with mean as some columns it would not be appropriate)

#split training and testing data sets
training_set = data[data['train'] == 'T']
test_set = data[data['train'] == 'F']

#get x and y from training set
X_train = training_set[['logCancerVol',  'logCancerWeight',  'age',  'logBenighHP',  'svi',  'logCP',  'gleasonScore',  'gleasonS45']]
Y_train = training_set[['levelCancerAntigen']]

#get x and y from testing set
X_test = test_set[['logCancerVol',  'logCancerWeight',  'age',  'logBenighHP',  'svi',  'logCP',  'gleasonScore',  'gleasonS45']]
Y_test = test_set[['levelCancerAntigen']]

#scale data
scaler = StandardScaler() #when scaled becomes numpy array rather then pada dataset
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  
Y_train_scaled = scaler.fit_transform(Y_train)
Y_test_scaled = scaler.transform(Y_test)




#Part 3) Linear regression and Ridge regression comparison

#LINEAR REGRESSION

#build and train LRM model with training data
lm = LinearRegression(fit_intercept=True, copy_X=True)
lm.fit(X_train_scaled,Y_train_scaled)

#find predicted outcomes using trained LRM 
y_pred = lm.predict(X_test_scaled)
LSE_mse = mean_squared_error(Y_test_scaled, y_pred)



#RIDGE REGRESSION

#Function to find the optimal alpha value for Ridge regression using manual cross-validation split
def give_alpha_manual_split(X, y):
    #Define the range of alpha values to be tested
    alphas = [0.01, 0.1, 1, 10, 100, 1000]

    #Initialize a list to store the mean cross-validation scores for each alpha
    mean_scores = []
    k_part_length = len(X) // 10  #Calculate the size of each fold

    #Iterate over each alpha value
    for alpha in alphas:
        #list to store mse values during folds
        values_mse = []  

        #CV with 10 folds
        for fold in range(10):
            start = fold * k_part_length  #find start index of current fold in data(length of each part * the part number)
            finish = (fold + 1) * k_part_length  #find end index of the same fold

            X_test = X[start:finish]  #Define test data input for this fold using start and finish
            y_test = y[start:finish] 

            #Construct the training data and target values by excluding the current fold
            X_train = np.concatenate((X[:start], X[finish:]), axis=0)
            y_train = np.concatenate((y[:start], y[finish:]), axis=0)

            #Create a Ridge regression model with the current alpha and fit it to the training data
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_train, y_train)

            #Make predictions on the test data and calculate the mean squared error (MSE)
            y_pred = ridge.predict(X_test)
            mse = ((y_test - y_pred) ** 2).mean()
            values_mse.append(mse)  #Store the MSE for this fold

        #Calculate the mean MSE for the current alpha
        mean_mse = np.mean(values_mse)
        mean_scores.append(mean_mse)  #Store the mean MSE for this alpha

    #Find the best alpha with the lowest mean cross-validation score
    best_alpha = alphas[np.argmin(mean_scores)]  #Choose the alpha with the lowest mean MSE

    return best_alpha  #Return the optimal alpha value


#get optimal alpha value useing cross validation
alpha =  give_alpha_manual_split(X_train_scaled, Y_train_scaled)

#Create Ridge model
ridge = Ridge(alpha=alpha)

#Train the model
ridge.fit(X_train_scaled, Y_train_scaled)

#Predict outcomes with the trained model on test data
y_pred_ridge = ridge.predict(X_test_scaled)

#Evaluate performance by calculating MSE
ridge_mse = mean_squared_error(Y_test_scaled, y_pred_ridge)

#Get results
print('The mse for lr is : ',LSE_mse)
print('MSE for Ridge Regression is: ', ridge_mse,' using an alpha value of ',alpha)

coefficients = ridge.coef_
print(coefficients)



feature_names = X_train.columns
plt.bar(feature_names, coefficients[0])
plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('Coefficients')
plt.title('Ridge Regression Coefficients')
plt.show()

residuals = Y_test_scaled - y_pred_ridge
plt.scatter(y_pred_ridge, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()