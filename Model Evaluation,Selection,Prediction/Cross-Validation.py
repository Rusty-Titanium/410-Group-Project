import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv('/Users/jasonbradley/PycharmProjects/Data Science/VolatilityDataFrame')

# Split dataset
X = df['Sector']
y = df['Volatility']
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists
mse_lr = []
mae_lr = []
rmse_lr = []
r2_lr = []

mse_rf = []
mae_rf = []
rmse_rf = []
r2_rf = []

mse_knn = []
mae_knn = []
rmse_knn = []
r2_knn = []

mse_svm = []
mae_svm = []
rmse_svm = []
r2_svm = []

mse_br = []
mae_br = []
rmse_br = []
r2_br = []

# Loop through each fold, train/evaluate the models
for train_idx, test_idx in kf.split(X):
    # Split the data into training and testing sets
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Convert the categorical variable into binary vectors using one-hot encoding
    encoder = OneHotEncoder()
    X_train_encoded = encoder.fit_transform(X_train.values.reshape(-1, 1))
    X_test_encoded = encoder.transform(X_test.values.reshape(-1, 1))

    # Scale the continuous variable using StandardScaler
    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))

    # Train and evaluate the Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train_encoded, y_train_scaled)
    y_pred_lr = lr_model.predict(X_test_encoded)
    mse_lr.append(mean_squared_error(y_test_scaled, y_pred_lr))
    mae_lr.append(mean_absolute_error(y_test_scaled, y_pred_lr))
    rmse_lr.append(np.sqrt(mean_squared_error(y_test_scaled, y_pred_lr)))
    r2_lr.append(r2_score(y_test_scaled, y_pred_lr))

    # Train and evaluate the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_encoded, y_train_scaled.ravel())
    y_pred_rf = rf_model.predict(X_test_encoded)
    mse_rf.append(mean_squared_error(y_test_scaled, y_pred_rf))
    mae_rf.append(mean_absolute_error(y_test_scaled, y_pred_rf))
    rmse_rf.append(np.sqrt(mean_squared_error(y_test_scaled, y_pred_rf)))
    r2_rf.append(r2_score(y_test_scaled, y_pred_rf))

    # Train and evaluate the K-nearest neighbors model
    knn_model = KNeighborsRegressor()
    knn_model.fit(X_train_encoded, y_train_scaled.ravel())
    y_pred_knn = knn_model.predict(X_test_encoded)
    mse_knn.append(mean_squared_error(y_test_scaled, y_pred_knn))
    mae_knn.append(mean_absolute_error(y_test_scaled, y_pred_knn))
    rmse_knn.append(np.sqrt(mean_squared_error(y_test_scaled, y_pred_knn)))
    r2_knn.append(r2_score(y_test_scaled, y_pred_knn))

    # Train and evaluate the Support Vector Machine model
    svm_model = SVR()
    svm_model.fit(X_train_encoded, y_train_scaled.ravel())
    y_pred_svm = svm_model.predict(X_test_encoded)
    mse_svm.append(mean_squared_error(y_test_scaled, y_pred_svm))
    mae_svm.append(mean_absolute_error(y_test_scaled, y_pred_svm))
    rmse_svm.append(np.sqrt(mean_squared_error(y_test_scaled, y_pred_svm)))
    r2_svm.append(r2_score(y_test_scaled, y_pred_svm))

    # Train and evaluate the Bayesian Ridge model
    br_model = BayesianRidge()
    br_model.fit(X_train_encoded.toarray(), y_train_scaled.ravel())
    y_pred_br = br_model.predict(X_test_encoded.toarray())
    mse_br.append(mean_squared_error(y_test_scaled, y_pred_br))
    mae_br.append(mean_absolute_error(y_test_scaled, y_pred_br))
    rmse_br.append(np.sqrt(mean_squared_error(y_test_scaled, y_pred_br)))
    r2_br.append(r2_score(y_test_scaled, y_pred_br))


#Calculate the mean and standard deviation
print('Linear Regression - Mean MSE:', np.mean(mse_lr), 'Std MSE:', np.std(mse_lr))
print('Random Forest - Mean MSE:', np.mean(mse_rf), 'Std MSE:', np.std(mse_rf))
print('K-nearest Neighbors - Mean MSE:', np.mean(mse_knn), 'Std MSE:', np.std(mse_knn))
print('Support Vector Machine - Mean MSE:', np.mean(mse_svm), 'Std MSE:', np.std(mse_svm))
print('Bayesian Ridge - Mean MSE:', np.mean(mse_br), 'Std MSE:', np.std(mse_br))


print('Linear Regression - Mean MAE:', np.mean(mae_lr), 'Std MAE:', np.std(mae_lr))
print('Random Forest - Mean MAE:', np.mean(mae_rf), 'Std MAE:', np.std(mae_rf))
print('K-nearest Neighbors - Mean MAE:', np.mean(mae_knn), 'Std MAE:', np.std(mae_knn))
print('Support Vector Machine - Mean MAE:', np.mean(mae_svm), 'Std MAE:', np.std(mae_svm))
print('Bayesian Ridge - Mean MAE:', np.mean(mae_br), 'Std MAE:', np.std(mae_br))


print('Linear Regression - Mean RMSE:', np.mean(rmse_lr), 'Std RMSE:', np.std(rmse_lr))
print('Random Forest - Mean RMSE:', np.mean(rmse_rf), 'Std RMSE:', np.std(rmse_rf))
print('K-nearest Neighbors - Mean RMSE:', np.mean(rmse_knn), 'Std RMSE:', np.std(rmse_knn))
print('Support Vector Machine - Mean RMSE:', np.mean(rmse_svm), 'Std RMSE:', np.std(rmse_svm))
print('Bayesian Ridge - Mean RMSE:', np.mean(rmse_br), 'Std RMSE:', np.std(rmse_br))


print('Linear Regression - Mean R2:', np.mean(r2_lr), 'Std R2:', np.std(r2_lr))
print('Random Forest - Mean R2:', np.mean(r2_rf), 'Std R2:', np.std(r2_rf))
print('K-nearest Neighbors -Mean R2: ', np.mean(r2_knn), 'Std R2:', np.std(r2_knn))
print('Support Vector Machine - Mean R2:', np.mean(r2_svm), 'Std R2:', np.std(r2_svm))
print('Bayesian Ridge - Mean R2:', np.mean(r2_br), 'Std R2:', np.std(r2_br))




