import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

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

