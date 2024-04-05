import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
import pandas as pd
import matplotlib.pyplot as plt
import sys
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
import seaborn as sns



def give_alpha(X, y):
    # Generate some sample data, replace this with your actual data
    np.random.seed(42)
    #X = np.random.rand(65, 8)
    #y = np.random.rand(65)

    # Define the range of alpha values to be tested
    alphas = [0.01, 0.1, 1, 10, 100, 1000, 100000, 500000]

    # Initialize a list to store the mean cross-validation scores for each alpha
    mean_scores = []

    # Create a 5-fold cross-validation iterator
    kf = KFold(n_splits=5)

    # Perform cross-validation for each alpha
    for alpha in alphas:
        scores = cross_val_score(Ridge(alpha=alpha), X, y, cv=kf, scoring='neg_mean_squared_error')
        mean_scores.append(-scores.mean())  # Convert negative MSE back to positive

    # Find the best alpha with the lowest mean cross-validation score
    best_alpha = alphas[np.argmin(mean_scores)]
    best_score = min(mean_scores)

    print("Best alpha:", best_alpha)
    print("Best cross-validation score (MSE):", best_score)








def give_alpha_manual_split(X, y):
    # Define the range of alpha values to be tested
    alphas = [0.01, 0.1, 1, 10, 100, 1000, 100000, 500000]

    # Initialize a list to store the mean cross-validation scores for each alpha
    mean_scores = []

    num_folds = 5
    fold_size = len(X) // num_folds

    for alpha in alphas:
        mse_scores = []

        for fold in range(num_folds):
            start = fold * fold_size
            end = (fold + 1) * fold_size

            X_test = X[start:end]
            y_test = y[start:end]

            X_train = np.concatenate((X[:start], X[end:]), axis=0)
            y_train = np.concatenate((y[:start], y[end:]), axis=0)

            ridge = Ridge(alpha=alpha)
            ridge.fit(X_train, y_train)

            y_pred = ridge.predict(X_test)
            mse = ((y_test - y_pred) ** 2).mean()
            mse_scores.append(mse)

        mean_mse = np.mean(mse_scores)
        mean_scores.append(mean_mse)

    # Find the best alpha with the lowest mean cross-validation score
    best_alpha = alphas[np.argmin(mean_scores)]
    best_score = min(mean_scores)

    print("Best alpha:", best_alpha)
    print("Best cross-validation score (MSE):", best_score)

# You should uncomment this part to generate sample data or replace it with your actual data
# np.random.seed(42)
# X = np.random.rand(65, 8)
# y = np.random.rand(65)

# Call the function with your data
# give_alpha_manual_split(X, y)













# PAART 1:  Load the dataset
#manually make column names and then set them as column titles whilst skipping first line of data holding the columns(to be improved I will automatically get the list of column names from the data itself)
column_names = ['index',  'logCancerVol',  'logCancerWeight',  'age',  'logBenighHP',  'svi',  'logCP',  'gleasonScore',  'gleasonS45',  'levelCancerAntigen',  'train']
data = pd.read_csv('Linear vs Regression/Task1_RegressionTask_CancerData.txt', sep='\t', skiprows=1, names=column_names)




#PART 2: Cleaning the data

#cleaning data with null values
data = data.dropna()  #removes rows with empty cells(doing this instead of replacing with mean as some columns it would not be appropriate)
#print(data['train'])
#cleaning data with wrong format

#cleaning wrong data

#removing duplicates(make sure this is appropriate for the data before you do so)



#Part 3a) :  Linear regression
# Prepare your features and target variable



#split training and test sets
training_set = data[data['train'] == 'T']
test_set = data[data['train'] == 'F']

#get x and y from training set
X_train = training_set[['logCancerVol',  'logCancerWeight',  'age',  'logBenighHP',  'svi',  'logCP',  'gleasonScore',  'gleasonS45']]
Y_train = training_set[['levelCancerAntigen']]

#
scaler = scaler = StandardScaler() #when scaled becomes numpy array rather then pada dataset
X_train_scaled = scaler.fit_transform(X_train)
Y_train_scaled = scaler.fit_transform(Y_train)

lm = LinearRegression(fit_intercept=True, copy_X=True)
lm.fit(X_train_scaled,Y_train_scaled)


# Access the coefficients and intercept
coefficients = lm.coef_
intercept = lm.intercept_

#print("Coefficients:", coefficients)
#print("Intercept:", intercept)

'''
#This code plots the training data for linear regression
g2 = sns.pairplot(data=training_set, x_vars=['logCancerVol',  'logCancerWeight',  'age'], y_vars = 'levelCancerAntigen', height=3)
plt.show()
'''




#NOW FINDING PREDICTED VALUES OF TEST DATA INPUT USING THE LINEAR MODEL DEVELOPED BY TRAINING DATA 


#get x and y from training set
X_test = test_set[['logCancerVol',  'logCancerWeight',  'age',  'logBenighHP',  'svi',  'logCP',  'gleasonScore',  'gleasonS45']]
Y_test = test_set[['levelCancerAntigen']]

#scale x and y like we did in training data
X_test_scaled = scaler.fit_transform(X_test)
Y_test_scaled = scaler.fit_transform(Y_test)

#find predicted outcomes using trained LRM 
y_pred = lm.predict(X_test_scaled)
final_mse = mean_squared_error(Y_test_scaled, y_pred)

#print('The mse for lr is : ',final_mse)






'''

Part 3b) RIDGE REGRESSION

'''

#return mse for that value of alpha
def find_mse_cv(alpha, k_parts, y_train):  #takes alpha, kparts and y_train with 65 rows each

    # Splitting Y into 52  rows for training and 13 for testing 
    Y_training, Y_testing = train_test_split(y_train, test_size=13, random_state=42)
    
    cv_Xtraining_data = None
    cv_Xtest_data = None
    mse_list = []
    for i in k_parts:
        test_dataInList = []
        cv_Xtest_data = i  #this will be the 1 part acting as input test data

        #now form the training data(4 parts of 5)
        for j in k_parts:
            if j.equals(i):
                pass
            else:
                test_dataInList.append(j)  
        
        cv_Xtraining_data = pd.concat(test_dataInList)

        #now that we have test and training data split we run the Ridge resgression model with value of Alpha given
        scaled_x_training = scaler.fit_transform(cv_Xtraining_data)
        scaled_x_testing = scaler.fit_transform(cv_Xtest_data)
        scaled_y_training = scaler.fit_transform(Y_training)
        scaled_y_testing = scaler.fit_transform(Y_testing)

        mse = ridge_regression_and_mse(scaled_x_training, scaled_y_training, scaled_x_testing, scaled_y_testing, alpha)
        #mse = ridge_regression_and_mse(cv_Xtraining_data, Y_training, cv_Xtest_data, Y_testing, alpha)
        mse_list.append(mse)

    mse_average = find_average(mse_list)
    return(mse_average)
        
def find_average(float_list):
    #calculate the average
    average = sum(float_list) / len(float_list)

    #return the average
    return(average)     

def ridge_regression_and_mse(X_train, Y_train, X_test, Y_test, alpha):
    # Create Ridge model
    ridge = Ridge(alpha=alpha)

    # Train the model
    ridge.fit(X_train, Y_train)

    # Predict outcomes with the trained model on test data
    y_pred_ridge = ridge.predict(X_test)

    # Evaluate performance by calculating MSE
    mse = mean_squared_error(Y_test, y_pred_ridge)

    return mse
   
        
#Find optimal alpha value


# Shuffle the rows while keeping categories intact
shuffled_X_train = X_train.sample(frac = 1, random_state = 42)

#split rows in data into 5 parts
part_1 = X_train[0:13]
part_2 = X_train[13:26]
part_3 = X_train[26:39]
part_4 = X_train[39:52]
part_5 = X_train[52:65] #missed last 2 elements as 67 elements is a prime number and 65 elements is divisable by 5
k_parts = [part_1, part_2, part_3, part_4, part_5] #list of panda dataset elements
combined_df = pd.concat([part_1, part_2, part_3, part_4, part_5], axis=0)
copy_y65 = Y_train[2:] #make y same size as x



'''
Y_training, Y_testing = train_test_split(copy_y65, test_size=13, random_state=42)


scaled_x_training = scaler.fit_transform(combined_df)
scaled_x_testing = scaler.fit_transform(part_5)
scaled_y_training = scaler.fit_transform(Y_training)
scaled_y_testing = scaler.fit_transform(Y_testing)
my_mse = ridge_regression_and_mse(Y_training, scaled_x_training, Y_testing, scaled_x_testing, 100000)
full_mse = ridge_regression_and_mse(Y_train_scaled, X_train_scaled, Y_test_scaled, X_test_scaled, 100000)
print(my_mse)
print(full_mse)
'''

combined_df_scaled = scaler.fit_transform(combined_df)
copy_65_scaled = scaler.fit_transform(copy_y65)
give_alpha(combined_df_scaled, copy_65_scaled)
give_alpha_manual_split(combined_df_scaled, copy_65_scaled)








