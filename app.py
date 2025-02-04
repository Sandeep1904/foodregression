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

st.title("Let's build a prediction model for quantity of dishes ordered at an Indian restaurant!")


@st.cache_data
def load_data(file):
    return pd.read_csv(file)

df = load_data('restaurant-1-orders.csv')
st.write("""
Here's how the data looks initially -
""")

st.write(df.head(7))

#-----------------------------------------------------------------
st.write("""
Let's clean the data and remove the unnecessary columns, modify some columns -
The final dataframe looks like this -
""")

@st.cache_data
def preprocess(df):
    df.drop(columns=['Order Number', 'Total products', 'Item Name', 'Product Price'], inplace=True)
    df["Order Date"] = pd.to_datetime(df["Order Date"], format='%d/%m/%Y %H:%M')
    df.sort_values(by='Order Date', inplace=True)
    # will have to remove the first 10 rows as they are wrong about the time stamp and total products. this will affect the plots.
    df.drop(df.head(10).index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.resample('D', on='Order Date').sum().reset_index()

    return df

df = preprocess(df)
st.write("We have grouped the data by the days.")
st.write(df.head(7))

#plots
st.write("## A few plots to understand the data better -")

st.write("We can see that there is a lot of variance in the data.")
st.write("We can see that the the order quantity has an upward trend.")

plot1 = plt.figure(figsize=(20,8))
sns.lineplot(df, x='Order Date', y='Quantity')
st.pyplot(plot1)
st.write(df.describe())

st.write("We should remove the outliers according to this plot -")
plot2 = plt.figure(figsize=(12,8))
sns.boxplot(df)
st.pyplot(plot2)

# removing outliers
@st.cache_data
def rem_outliers(df):
    Q1 = df['Quantity'].quantile(0.25)
    Q3 = df['Quantity'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['Quantity'] >= lower_bound) & (df['Quantity'] <= upper_bound)]
    
    return df

df = rem_outliers(df)
st.write("As you can see the mean and std have shifted -")
st.write(df.describe())

plot3 = plt.figure(figsize=(20,8))
sns.kdeplot(data=df, x = 'Quantity')
st.write("In the distribution graph below, you can see that most of the orders are concentrated around 10 and 50 no. of items.")
st.pyplot(plot3)


#-----------------------------------------------------------------

st.write("## Now let's build an autoregression model!")
st.write("For using SARIMA, we need the check for a few things, such as stationarity and seasonality -")
st.write("We shall perform the Augmented Dickey–Fuller test using the statsmodels library.")

result = adfuller(df['Quantity'])
st.write(f'ADF Statistic: {result[0]}')
st.write(f'p-value: {result[1]}')
st.write("This suggests that data is not stationary. Let's check a few plots to verify this -")

decomposition = seasonal_decompose(df['Quantity'], model='additive', period=365)  # Assuming yearly seasonality
plot4 = decomposition.plot()
st.pyplot(plot4)

# Plot the ACF and PACF

plot5 = plot_acf(df['Quantity'], lags=100)  # Adjust the lags based on your data
plt.title('Autocorrelation Function (ACF)')
st.pyplot(plot5)

plot6 = plot_pacf(df['Quantity'], lags=50)  # Adjust the lags based on your data
plt.title('Partial Autocorrelation Function (ACF)')
st.pyplot(plot6)

#-----------------------------------------------------------------

st.write("## Time for some SARIMA!")

@st.cache_data
def SARIMA(df):
    # Define SARIMA parameters
    p, d, q = 1, 1, 1  # Non-seasonal parameters
    P, D, Q, s = 1, 1, 1, 7  # Seasonal parameters (weekly seasonality)

    # Fit SARIMA model
    model = SARIMAX(df['Quantity'],
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    enforce_stationarity=False,
                    enforce_invertibility=False)

    return model.fit(disp=False)

sarima_result = SARIMA(df)

# Print summary of the model
st.write(sarima_result.summary())

# Generate predictions
df['SARIMA_Predictions'] = sarima_result.fittedvalues
df['SARIMA_Predictions'] = df['SARIMA_Predictions'].abs().astype(int) # so they are not negative and/or floats

# Plot actual vs fitted values
plot7 = plt.figure(figsize=(12, 6))
plt.plot(df['Quantity'], label='Actual')
plt.plot(df['SARIMA_Predictions'], label='SARIMA Predictions', linestyle='--')
plt.title('SARIMA Model - Actual vs Predicted')
plt.legend()
st.pyplot(plot7)

# Calculate metrics
mae = mean_absolute_error(df['Quantity'], df['SARIMA_Predictions'])
mse = mean_squared_error(df['Quantity'], df['SARIMA_Predictions'])
st.write(f"Mean Absolute Error (MAE): {mae}")
st.write(f"Mean Squared Error (MSE): {mse}")
st.write(df.sample(7))

st.write("I applied various other smoothening and transformation techniques, but failed to get the MAE under 25.")
st.write("Model's predictions are 25 quantities up or down on an average.")
st.write("This maybe due to the face that our data has high variations and also that it is not stationary.")
st.write("Let's implement a Boosting Regression model and compare if it can do any better.")

#-----------------------------------------------------------------

st.write("## XGBoost Regression")
st.write("First we need to create some new features in the dataset for the model to learn.")
st.write("""
We will start by converting timestamps to individual year, month, day, 
         day_of_week, hour, is_weekend columns. Then we will add some lag
         and rolling mean features. And finally we will separate our input 
         and target features.
""")

@st.cache_data
def prep_for_XGB(df):
    # Create new features from 'Order Date'
    df['year'] = df['Order Date'].dt.year
    df['month'] = df['Order Date'].dt.month
    df['day'] = df['Order Date'].dt.day
    df['day_of_week'] = df['Order Date'].dt.dayofweek  # 0: Monday, 6: Sunday
    df['is_weekend'] = (df['Order Date'].dt.dayofweek >= 5).astype(int)
    df['hour'] = df['Order Date'].dt.hour


    # Create lag features
    df['Quantity_lag_1'] = df['Quantity'].shift(1)

    # Create rolling mean
    df['Quantity_rolling_mean_7'] = df['Quantity'].rolling(window=7).mean()

    # Handle missing values (if any)
    df = df.fillna(0)  # Replace missing values with 0

    # 2. Feature Selection (For demonstration, using a subset of features)
    features = ['year', 'month', 'day', 'day_of_week', 'is_weekend', 'hour', 'Quantity_lag_1', 'Quantity_rolling_mean_7']
    X = df[features]
    y = df['Quantity']

    return X, y

X, y = prep_for_XGB(df)

st.write(X.sample(5))
st.write(y.sample(5))

@st.cache_data
def XGB(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


    params = {
        "n_estimators": 500,
        "max_depth": 4,
        "min_samples_split": 5,
        "learning_rate": 0.01,
        "loss": "squared_error",
    }

    # 4. Create and train XGBoost model
    model = ensemble.GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)

    # 5. Make predictions
    y_pred = model.predict(X_train)

    # 6. Evaluate model performance
    mae = mean_absolute_error(y_train, y_pred)
    mse = mean_squared_error(y_train, y_pred)
    st.write(f"TRAINING: MAE = {mae} and MSE = {mse}")

    return model, X_test, y_test

model, X_test, y_test = XGB(X,y)
y_tpred = model.predict(X_test)
mae = mean_absolute_error(y_tpred, y_test)
mse = mean_squared_error(y_tpred, y_test)
st.write(f"TESTING: MAE = {mae} and MSE = {mse}")



#-----------------------------------------------------------------

st.write("### Well, we can see that XGBoost Regressor has performed better than SARIMA by a slight margin.")

   






