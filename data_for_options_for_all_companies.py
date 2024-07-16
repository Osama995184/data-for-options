#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.stats as si 
from datetime import date
from scipy import stats
from math import log, sqrt, exp
from scipy.optimize import newton
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from bs4 import BeautifulSoup as bs
from py_vollib.black_scholes.implied_volatility import implied_volatility as iv
import re
import math
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime
import mibian
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


# In[2]:


def black_scholes_call(row):
    S, X, T, r, sigma = row['Stock_price'], row['Strike_Price'], 1, row['Rate'], row['Rolling_Std']
    d1 = (np.log(S / X) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(row):
    S, X, T, r, sigma = row['Stock_price'], row['Strike_Price'], 1, row['Rate'], row['Rolling_Std']
    d1 = (np.log(S / X) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return X * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def assign_premium_prices(df_option):
    for index, row in df_option.iterrows():
        if row['Option_type'] == 'Call':
            df_option.loc[index, 'Option_price'] = black_scholes_call(row)
        elif row['Option_type'] == 'Put':
            df_option.loc[index, 'Option_price'] = black_scholes_put(row)
    return df_option

def assign_strikes(df_option,strike_prices): 
#     strike_prices = compute_strike_prices(df_option['Stock_price'])
    df_list = []

    for strike_price in strike_prices:
        df_temp = df_option.copy()
        df_temp['Strike_Price'] = strike_price
        df_temp = assign_premium_prices(df_temp)
        df_temp['predect_option_Price'] = df_temp['Option_price'].shift(-1)
        df_list.append(df_temp)

    df_option = pd.concat(df_list, axis=0)
    df_option['Date'] = pd.to_datetime(df_option['Date'])
    df_option = df_option.sort_values(by='Date')
    return df_option
def compute_strike_prices(price_series):
        # Get the last price in the series
        last_price = price_series.iloc[-1]

        # Calculate the six required strike prices
        strike_prices = [
        max(last_price - 100, 1) if last_price >= 100 else 5,
        max(last_price - 50, 5) if last_price >= 50 else 20,
        max(last_price - 20, 10) if last_price >= 20 else 50,
        max(last_price + 20, 1),
        max(last_price + 80, 1),
        max(last_price + 110, 1)
    ]
        return strike_prices
def calculate_theta(S, K, T, r, sigma, option_type='Call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'Call':
        theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'Put':
        
        theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return theta / 7

def calculate_delta(S, K, T, r, sigma, option_type='Call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    if option_type == 'Call':
        delta = norm.cdf(d1)
    elif option_type == 'Put':
        delta = norm.cdf(d1)-1
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return delta

def calculate_vega(row):
    S, K, T, r, sigma = row['Stock_price'], row['Strike_Price'], 1, row['Rate'], (row['Rolling_Std']/100)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T)
    return vega 

def calculate_rho(S, K, T, r, sigma, option_type='Call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'Call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'Put':
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return rho / 100

def calculate_gamma(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

def calculate_greeks(row):
    stock_price = row['Stock_price']
    strike_price = row['Strike_Price']
    time_to_maturity = 1
    interest_rate = row['Rate']
    volatility = (row['Rolling_Std']/100)
    option_type = row['Option_type']
    row['theta'] = calculate_theta(stock_price, strike_price, time_to_maturity, interest_rate, volatility, option_type='Call')
    row['delta'] = calculate_delta(stock_price, strike_price, time_to_maturity, interest_rate, volatility, option_type='Call')
    row['gamma'] = calculate_gamma(stock_price, strike_price, time_to_maturity, interest_rate, volatility)
    row['rho'] = calculate_rho(stock_price, strike_price, time_to_maturity, interest_rate, volatility, option_type='Call')

    return row


# In[30]:


def calculate_option_data(company_name,date_to_filtered):
    path = f'D:\\Quantum\\Codes\\Option_data_from_historical\\SI\\{company_name}.csv'
    df = pd.read_csv(path)
    df = df.dropna()
    df = df.drop([f'{company_name}_Open'],axis=1)
    df['Date'] = pd.to_datetime(df['Date'])
    date_filtered = pd.to_datetime(date_to_filtered)
    df = df[df['Date'] > date_filtered]
    df["Date"] = df["Date"].dt.strftime('%m/%d/%Y')
    df["Sympol"] = company_name

    df_interest = pd.read_csv('Treasury_10y.csv')
    df_interest['Date'] = pd.to_datetime(df_interest['Date'])
    df_interest["Date"] = df_interest["Date"].dt.strftime('%m/%d/%Y')

    df_option = df.merge(df_interest,left_on=['Date'],right_on=['Date'],how='left')
    df_option['Date'] = pd.to_datetime(df_option['Date'])
    df_option = df_option.sort_values(by='Date')
    df_option=df_option.rename(columns={f'{company_name}_Close': 'Price'})

    if df_option['Price'].dtype != 'object':
        df_option['Price'] = df_option['Price'].astype(str)

    # Remove commas from 'Price' column
    df_option['Price'] = df_option['Price'].str.replace(',', '')

    # Convert 'Price' column to float
    df_option['Price'] = df_option['Price'].astype(float)
    df_option['std_dev'] = df_option['Price'].std()
    df_option['sharpe_ratio'] = ((df_option['Price'].mean() - df_option['Rate']) / df_option['std_dev']) * np.sqrt(252)
    df_option['Rolling_Std'] = df_option['Price'].rolling(window=len(df_option), min_periods=1).std().shift(-1)
#     df_option = df_option.drop(df_option.index[-1])

    df_option=df_option.rename(columns={'Price': 'Stock_price'})
    df_option['Predicted_price_stock'] = df_option['Stock_price'].shift(-1)
    df_option = df_option.dropna()

    df_option_call = df_option.copy()
    df_option_put = df_option.copy()
    df_option_call['Option_type'] = 'Call'
    df_option_put['Option_type'] = 'Put'
    
    path2 = f'D:\\Quantum\\Codes\\Option_data_from_historical\\options_data\\companies_train\\{company_name}_final_data_for_options.csv'
    df2 = pd.read_csv(path2)
    Date_counts = df2['Strike_Price'].value_counts().to_frame()
    Date_counts.rename(columns={'Strike_Price': 'value_counts'}, inplace=True)
    Date_counts.index.name = 'Strike_Price'
    Date_counts_sorted = Date_counts.sort_index()
    strike_prices = Date_counts_sorted.index.tolist()
    
    df_option_call = assign_strikes(df_option_call,strike_prices)
    df_option_call = df_option_call.dropna()
    df_option_call['Strike_Price'] = df_option_call['Strike_Price'].round(0)

    df_option_put = assign_strikes(df_option_put,strike_prices)
    df_option_put = df_option_put.dropna()
    df_option_put['Strike_Price'] = df_option_put['Strike_Price'].round(0)

    df_option_call.replace(0, 0.5, inplace=True)
    df_option_call.to_csv(f'df_option_call.csv',index=False)
    df_option_call =  pd.read_csv(f'df_option_call.csv')

    df_option_put.replace(0,0.5, inplace=True)
    df_option_put.to_csv(f'df_option_put.csv',index=False)
    df_option_put =  pd.read_csv(f'df_option_put.csv')

    def implied_volatility(df):
        results = []  # Store the results for each row
        for index, row in df.iterrows():
            P = float(row['Option_price'])
            S = float(row['Stock_price'])
            E = float(row['Strike_Price'])
            T = float(1)
            r = float(row['Rate'])
            sigma = float(row['Rolling_Std']/100)

            if sigma == 0:
                results.append(None)
                continue

            while sigma < 1:
                try:
                    d_1 = float((math.log(S/E) + (r + (sigma**2)/2) * T) / (sigma * math.sqrt(T)))
                    d_2 = float((math.log(S/E) + (r - (sigma**2)/2) * T) / (sigma * math.sqrt(T)))
                    P_implied = float(S * norm.cdf(d_1) - E * math.exp(-r * T) * norm.cdf(d_2))

                    if abs(P - P_implied) < 0.001:
                        results.append(sigma)
                        break  # Exit the loop once the desired accuracy is achieved
                    sigma += 0.001
                except ZeroDivisionError:
                    results.append(None)
                    break
            else:
                results.append(None)

        return results
    df_option_put['implied_volatility'] = implied_volatility(df_option_put)
    df_option_call['implied_volatility'] = implied_volatility(df_option_call)

    df_option = pd.concat([df_option_call, df_option_put], axis=0)
    df_option['Date'] = pd.to_datetime(df_option['Date'])
    df_option = df_option.sort_values(by='Date')
    
    df_option['implied_volatility'] = pd.to_numeric(df_option['implied_volatility'], errors='coerce')

    imputer = SimpleImputer(strategy='mean')
    df_option[['implied_volatility']] = imputer.fit_transform(df_option[['implied_volatility']])

    df_option.to_csv(f'data.csv',index=False)

    df_option =  pd.read_csv(f'data.csv')

    df_option['Vega'] = df_option.apply(calculate_vega, axis=1)
    calculate_greeks(df_option)

    df_option=df_option.rename(columns={'Predicted_price_stock': 'future_stock_one_day'})
    df_option=df_option.rename(columns={'predect_option_Price': 'future_option_one_day'})

    df_option.to_csv(f'D:\\Quantum\\Codes\\Option_data_from_historical\\options_data\\companies_train\\{company_name}_final_data_for_options.csv',index=False, mode='a', header=False)
    os.remove('data.csv')
    os.remove('df_option_call.csv')
    os.remove('df_option_put.csv')
    print("Done")


# In[32]:


date_to_filtered = '6/13/2024'
companies = [('TWLO',date_to_filtered),('ZM',date_to_filtered),
             ('TEAM',date_to_filtered),('GTEK_ETF',date_to_filtered),
             ('SYM',date_to_filtered), ('SQ',date_to_filtered),
             ('SMH_ETF',date_to_filtered),('MLTX',date_to_filtered),
             ('MRVL',date_to_filtered),('MSI',date_to_filtered),
             ('ORCL',date_to_filtered), ('PATH',date_to_filtered),
             ('LPLA',date_to_filtered),('KLAC',date_to_filtered),
             ('LLY',date_to_filtered),('FTI',date_to_filtered),
             ('HUBS',date_to_filtered), ('INTC',date_to_filtered),
             ('CYBR',date_to_filtered),('BILL',date_to_filtered),
             ('DDOG',date_to_filtered),('DKNG',date_to_filtered),
             ('DT',date_to_filtered), ('ELF',date_to_filtered),
             ('CELH',date_to_filtered),('CMG',date_to_filtered),
             ('COIN',date_to_filtered), ('COST',date_to_filtered),
             ('PANW',date_to_filtered),('GOOGL',date_to_filtered),
             ('TSLA',date_to_filtered),('ADBE',date_to_filtered),
             ('NVDA',date_to_filtered),('AAPL',date_to_filtered),
             ('SMCI',date_to_filtered), ('AMZN',date_to_filtered),
             ('MSFT',date_to_filtered),('META',date_to_filtered),
             ('ANET',date_to_filtered),('ARKG',date_to_filtered),
             ('ARKK_ETF',date_to_filtered), ('ASML',date_to_filtered),
             ('AEYE',date_to_filtered),('CRWD',date_to_filtered),
             ('MA',date_to_filtered), ('MELI',date_to_filtered),
             ('SOUN',date_to_filtered),('ROKU',date_to_filtered),
             ('ROIV',date_to_filtered), ('RBLX',date_to_filtered),
             ('XLE_ETF',date_to_filtered),('XLF_ETF',date_to_filtered),
             ('WDAY',date_to_filtered), ('VRT',date_to_filtered),
             ('OXY',date_to_filtered),('SPCE',date_to_filtered),
             ('RIVN',date_to_filtered),('LCID',date_to_filtered),
             ('TSM',date_to_filtered),('VKTX',date_to_filtered),
             ('V',date_to_filtered),('UNH',date_to_filtered),
             ('UBER',date_to_filtered), ('U',date_to_filtered),
             ('CRM',date_to_filtered),('FTNT',date_to_filtered),
             ('AMD',date_to_filtered), ('NIO',date_to_filtered)
]
now = datetime.now()
# Format date and time
formatted = now.strftime("%Y-%m-%d %H:%M:%S")
print("Formatted date and time:", formatted)
for idx, (company_name,file_for_historical_data) in enumerate(companies):
    calculate_option_data(company_name,file_for_historical_data)
    print (f'done {company_name}')
    now = datetime.now()
    # Format date and time
    formatted = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Formatted date and time:", formatted)


# In[ ]:





# In[ ]:




