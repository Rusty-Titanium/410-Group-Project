import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv('/Users/jasonbradley/PycharmProjects/Data Science/VolatilityDataFrame')

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['Sector'], df['Volatility'], test_size=0.2)

# Convert the categorical variable
encoder = OneHotEncoder()
X_train_encoded = encoder.fit_transform(X_train.values.reshape(-1, 1))
X_test_encoded = encoder.transform(X_test.values.reshape(-1, 1))

# Scale the continuous
scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))

# Train and evaluate the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train_encoded, y_train_scaled)
y_pred_lr = lr_model.predict(X_test_encoded)
mse_lr = mean_squared_error(y_test_scaled, y_pred_lr)
mae_lr = mean_absolute_error(y_test_scaled, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test_scaled, y_pred_lr)

# Train and evaluate the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_encoded, y_train_scaled.ravel())
y_pred_rf = rf_model.predict(X_test_encoded)
mse_rf = mean_squared_error(y_test_scaled, y_pred_rf)
mae_rf = mean_absolute_error(y_test_scaled, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test_scaled, y_pred_rf)

# Train and evaluate the K-nearest neighbors model
knn_model = KNeighborsRegressor()
knn_model.fit(X_train_encoded, y_train_scaled.ravel())
y_pred_knn = knn_model.predict(X_test_encoded)
mse_knn = mean_squared_error(y_test_scaled, y_pred_knn)
mae_knn = mean_absolute_error(y_test_scaled, y_pred_knn)
rmse_knn = np.sqrt(mse_knn)
r2_knn = r2_score(y_test_scaled, y_pred_knn)

# Train and evaluate the Support Vector model
svm_model = SVR()
svm_model.fit(X_train_encoded, y_train_scaled.ravel())
y_pred_svm = svm_model.predict(X_test_encoded)
mse_svm = mean_squared_error(y_test_scaled, y_pred_svm)
mae_svm = mean_absolute_error(y_test_scaled, y_pred_svm)
rmse_svm = np.sqrt(mse_svm)
r2_svm = r2_score(y_test_scaled, y_pred_svm)

# Train and evaluate the Bayesian Ridge model
br_model = BayesianRidge()
br_model.fit(X_train_encoded.toarray(), y_train_scaled.ravel())
y_pred_br = br_model.predict(X_test_encoded.toarray())
mse_br = mean_squared_error(y_test_scaled, y_pred_br)
mae_br = mean_absolute_error(y_test_scaled, y_pred_br)
rmse_br = np.sqrt(mse_br)
r2_br = r2_score(y_test_scaled, y_pred_br)

print('Linear Regression MSE:', mse_lr, 'MAE:', mae_lr, 'RMSE:', rmse_lr, 'R2:', r2_lr)
print('Random Forest MSE:', mse_rf, 'MAE:', mae_rf, 'RMSE:', rmse_rf, 'R2:', r2_rf)
print('K-nearest neighbors MSE:', mse_knn, 'MAE:', mae_knn, 'RMSE:', rmse_knn, 'R2:', r2_knn)
print('Support Vector Machine MSE:', mse_svm, 'MAE:', mae_svm, 'RMSE:', rmse_svm, 'R2:', r2_svm)
print('Bayesian Ridge MSE:', mse_br, 'MAE:', mae_br, 'RMSE:', rmse_br, 'R2:', r2_br)