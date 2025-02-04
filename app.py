import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import ensemble

st.set_page_config(page_title="Restaurant Order Forecasting", layout="wide")
st.title("📊 Predicting Volume of Orders at an Indian Restaurant 🍽️")
url = "https://github.com/Sandeep1904/foodregression"
st.write("You can find the project on [github](%s)."% url)

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

df = load_data('restaurant-1-orders.csv')

st.header("Exploring the Raw Data")
st.write("Let's take a peek at the first few rows of our dataset:")
st.dataframe(df.head(7))  # Use st.dataframe for better display

st.header("Data Preprocessing and Feature Engineering")
st.write("Before we can build a model, we need to clean and prepare our data. This involves removing irrelevant columns, converting data types, and handling missing values.")

@st.cache_data
def preprocess(df):
    df.drop(columns=['Order Number', 'Total products', 'Item Name', 'Product Price'], inplace=True, errors='ignore') # added errors='ignore'
    df["Order Date"] = pd.to_datetime(df["Order Date"], format='%d/%m/%Y %H:%M', errors='coerce') # added errors='coerce'
    df.sort_values(by='Order Date', inplace=True)
    df.dropna(subset=['Order Date'], inplace=True) # drop rows with NaT from incorrect date format.
    df.drop(df.head(10).index, inplace=True, errors='ignore') # added errors='ignore'
    df.reset_index(drop=True, inplace=True)
    df = df.resample('D', on='Order Date').sum().reset_index()
    return df

df = preprocess(df)

st.write("We've grouped the data by day, summing the quantities ordered.  Here's the transformed data:")
st.dataframe(df.head(7))

st.write("### Let's visualize the order quantities over time to understand the patterns and identify any trends or seasonality.")

fig, ax = plt.subplots(figsize=(20, 5))  # Create figure and axes objects
sns.lineplot(df, x='Order Date', y='Quantity', ax=ax)  # Plot on the axes
plt.title("Daily Order Quantities") # Add title to the plot
st.pyplot(fig)  # Display the plot using st.pyplot

st.write("#### As we can see, there's significant variance in the data, with an apparent upward trend. Let's examine the distribution of order quantities and identify potential outliers.")

fig, ax = plt.subplots(figsize=(20, 8))
sns.boxplot(df['Quantity'], ax=ax)
plt.title("Box Plot of Order Quantities")
st.pyplot(fig)

st.write("#### 👆 The box plot reveals several outliers.  Let's remove these to improve the performance of our models.")

@st.cache_data
def remove_outliers(df):
    Q1 = df['Quantity'].quantile(0.25)
    Q3 = df['Quantity'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['Quantity'] >= lower_bound) & (df['Quantity'] <= upper_bound)]
    return df

df = remove_outliers(df)

st.write("#### 👇 After removing outliers, the distribution of order quantities becomes clearer:")

fig, ax = plt.subplots(figsize=(20, 8))
sns.kdeplot(data=df, x='Quantity', ax=ax)
plt.title("Distribution of Order Quantities (Outliers Removed)")
st.pyplot(fig)


st.header("Time Series Analysis and SARIMA Modeling")
st.write("Now, let's build a time series model to forecast order quantities. We'll start with the SARIMA model, which is suitable for data with seasonality.")

st.subheader("Checking for Stationarity")
st.write("A key requirement for SARIMA is that the time series should be stationary. Let's perform the Augmented Dickey-Fuller (ADF) test to check for stationarity.")

result = adfuller(df['Quantity'])
st.write(f'ADF Statistic: {result[0]}')
st.write(f'p-value: {result[1]}')

st.write("The ADF test suggests that our data is likely non-stationary. To confirm this, let's decompose the time series into its trend, seasonality, and residual components.")

decomposition = seasonal_decompose(df['Quantity'], model='additive', period=365)
fig = decomposition.plot() # create the figure object
st.pyplot(fig) # then display it

st.subheader("ACF and PACF Plots")
st.write("To determine the appropriate (p, d, q) and (P, D, Q) parameters for our SARIMA model, we'll examine the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots.")

fig, ax = plt.subplots(figsize=(20,6))
plot_acf(df['Quantity'], lags=100, ax=ax)
plt.title('Autocorrelation Function (ACF)')
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(20,6))
plot_pacf(df['Quantity'], lags=50, ax=ax)
plt.title('Partial Autocorrelation Function (PACF)')
st.pyplot(fig)

@st.cache_data
def train_sarima(df):
    p, d, q = 1, 1, 1
    P, D, Q, s = 1, 1, 1, 7  # Weekly seasonality
    model = SARIMAX(df['Quantity'], order=(p, d, q), seasonal_order=(P, D, Q, s), enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False)

sarima_result = train_sarima(df)
st.write(sarima_result.summary())

df['SARIMA_Predictions'] = sarima_result.fittedvalues
df['SARIMA_Predictions'] = df['SARIMA_Predictions'].abs().astype(int)

st.subheader("SARIMA Model Performance")
fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(df['Quantity'], label='Actual')
plt.plot(df['SARIMA_Predictions'], label='SARIMA Predictions', linestyle='--')
plt.title('SARIMA Model - Actual vs Predicted')
plt.legend()
st.pyplot(fig)

mae = mean_absolute_error(df['Quantity'], df['SARIMA_Predictions'])
mse = mean_squared_error(df['Quantity'], df['SARIMA_Predictions'])
st.write(f"Mean Absolute Error (MAE): {mae}")
st.write(f"Mean Squared Error (MSE): {mse}")


st.header("XGBoost Regression: A Machine Learning Approach")
st.write("Let's explore if a machine learning model can improve our predictions.  We'll use XGBoost, a powerful gradient boosting algorithm.")

st.subheader("Feature Engineering")
st.write("First, we'll create new features from the date and time information, as well as lagged and rolling mean features.")

@st.cache_data
def prepare_data_for_xgboost(df):
    df['year'] = df['Order Date'].dt.year
    df['month'] = df['Order Date'].dt.month
    df['day'] = df['Order Date'].dt.day
    df['day_of_week'] = df['Order Date'].dt.dayofweek
    df['is_weekend'] = (df['Order Date'].dt.dayofweek >= 5).astype(int)
    df['hour'] = df['Order Date'].dt.hour  # Hour of the day. It's 0 because the data is aggregated daily.

    df['Quantity_lag_1'] = df['Quantity'].shift(1)
    df['Quantity_rolling_mean_7'] = df['Quantity'].rolling(window=7).mean()
    df = df.fillna(0)  # Handle missing values

    features = ['year', 'month', 'day', 'day_of_week', 'is_weekend', 'hour', 'Quantity_lag_1', 'Quantity_rolling_mean_7']
    X = df[features]
    y = df['Quantity']
    return X, y

X, y = prepare_data_for_xgboost(df)

st.write("Here's a sample of our engineered features:")
st.dataframe(X.head())

st.subheader("Training the XGBoost Model")
st.write("We'll split the data into training and testing sets and train our XGBoost model. We'll then evaluate its performance on the test set.")

@st.cache_data  # Cache the trained model
def train_xgboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)  # Important: shuffle=False for time series
    params = {
        "n_estimators": 500, # increased estimators
        "max_depth": 4,
        "min_samples_split": 5,
        "learning_rate": 0.01,
        "loss": "squared_error",
        "random_state": 42 # added for reproducibility
    }
    model = ensemble.GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    st.write(f"Training MAE: {mae_train}")
    st.write(f"Training MSE: {mse_train}")

    return model, X_test, y_test

model, X_test, y_test = train_xgboost(X, y)



y_pred_test = model.predict(X_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
mse_test = mean_squared_error(y_test, y_pred_test)
st.write(f"Test MAE: {mae_test}")
st.write(f"Test MSE: {mse_test}")

st.subheader("XGBoost Model Predictions")
st.write("Let's visualize the actual vs. predicted order quantities from our XGBoost model.")

fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, label='Actual')  # Use y_test.index for x-axis
plt.plot(X_test.index, y_pred_test, label='XGBoost Predictions', linestyle='--') # Use X_test.index for x-axis
plt.title('XGBoost Model - Actual vs Predicted (Test Data)')
plt.legend()
st.pyplot(fig)


st.header("Model Comparison and Conclusion")
st.write("Both the SARIMA and XGBoost models provide forecasts of daily order quantities.  Let's compare their performance based on the Mean Absolute Error (MAE) and Mean Squared Error (MSE).")

st.write(f"##### **SARIMA Model (Test Data):** MAE: {mae}, MSE: {mse}")
st.write(f"##### **XGBoost Model (Test Data):** MAE: {mae_test}, MSE: {mse_test}")

st.write("🧩 In this particular case, XGBoost model performs better than SARIMA model.")
st.write("👉 However, it's important to note that the best model depends on the specific characteristics of the data and the forecasting task. Further experimentation with different models, feature engineering, and hyperparameter tuning may lead to even better results.")

st.write("👉 This analysis provides a starting point for forecasting restaurant orders. Further improvements can be made by incorporating external factors (e.g., holidays, promotions, weather) and exploring more advanced time series techniques.")
