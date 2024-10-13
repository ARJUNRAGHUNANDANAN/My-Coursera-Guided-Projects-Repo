import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from jupyterthemes import jtplot

# Task #1: Understand the Problem Statement and Business Case

# Task #2: Import Libraries and Datasets
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)

house_df = pd.read_csv('realestate_prices.csv', encoding='ISO-8859-1')

# Task #3: Perform Data Visualization
sns.scatterplot(x='sqft_living', y='price', data=house_df)
house_df.hist(bins=20, figsize=(30, 40), color='b')
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(house_df.corr(), annot=True)
house_df_sample = house_df[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']]
sns.pairplot(house_df_sample)

# Task #4: Perform Data Cleaning and Feature Engineering
selected_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement']
X = house_df[selected_features]
y = house_df['price']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y = y.values.reshape(-1, 1)
y_scaled = scaler.fit_transform(y)

# Task #5: Train a Deep Learning Model with Limited Number of Features
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.25)
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(100, input_dim=7, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='linear'))

model.summary()
model.compile(optimizer='Adam', loss='mean_squared_error')
epochs_hist = model.fit(X_train, y_train, epochs=100, batch_size=50, validation_split=0.2)

# Task #6: Evaluate Trained Deep Learning Model Performance
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training and Validation Loss')
plt.legend(['Training Loss', 'Validation Loss'])
X_test_1 = np.array([[4, 3, 1960, 5000, 1, 2000, 3000]])
scaler_1 = MinMaxScaler()
X_test_scaled_1 = scaler_1.fit_transform(X_test_1)
y_predict_1 = model.predict(X_test_scaled_1)
y_predict_1 = scaler.inverse_transform(y_predict_1)
y_predict = model.predict(X_test)
plt.plot(y_test, y_predict, "^", color='r')
plt.xlabel('Model Predictions')
plt.ylabel('True Values')
y_predict_orig = scaler.inverse_transform(y_predict)
y_test_orig = scaler.inverse_transform(y_test)
RMSE = float(format(np.sqrt(mean_squared_error(y_test_orig, y_predict_orig)), '.3f'))
MSE = mean_squared_error(y_test_orig, y_predict_orig)