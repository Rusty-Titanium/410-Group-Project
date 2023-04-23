import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

df = pd.read_csv('/Users/jasonbradley/PycharmProjects/Data Science/VolatilityDataFrame')

# Split dataset
X = df.drop('Volatility', axis=1)
y = df['Volatility']

# Split the data; training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Encode categorical variables
encoder = OneHotEncoder()
X_train_encoded = encoder.fit_transform(X_train['Sector'].values.reshape(-1, 1))
X_test_encoded = encoder.transform(X_test['Sector'].values.reshape(-1, 1))

# Scale continuous
scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))

# Train the linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train_encoded, y_train_scaled)

# Encode sectors
sector_encoder = OneHotEncoder()
sector_encoded = sector_encoder.fit_transform(df['Sector'].values.reshape(-1, 1))

# Scale the volatility values
volatility_scaled = scaler.fit_transform(df['Volatility'].values.reshape(-1, 1))

# Make predictions for the volatility 
volatility_predictions = scaler.inverse_transform(lr_model.predict(sector_encoded))

# Combine the predicted volatility values with the original DataFrame
df['Predicted Volatility`'] = volatility_predictions

# Find the sector with the lowest predicted volatility
safest_sector = df.loc[df['Predicted Volatility'].idxmin()]['Sector']

print('The safest sector to invest in is:', safest_sector)