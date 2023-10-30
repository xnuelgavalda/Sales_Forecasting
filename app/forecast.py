#-------------------------------------------
# Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image

import scipy.stats as scs

# plotly family
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import (download_plotlyjs,
                            init_notebook_mode,
                            plot, iplot)

import logging
from prophet import Prophet
logging.getLogger().setLevel(logging.ERROR)

from sklearn.metrics import (mean_absolute_error,
                            mean_squared_error,
                            mean_squared_log_error,
                            mean_absolute_percentage_error,
                            r2_score)

from joblib import load
import joblib
import base64
import sys



def load_data(url):
    df = pd.read_csv(url,
                     index_col='Date')
    df.index = pd.to_datetime(df.index)
    df.reset_index(inplace=True)
    return df


sales = load_data('./data/sales.csv')

print('sales', sales.head())

sales['year'] = [d.year for d in sales.Date]
sales['month'] = [d.strftime('%b') for d in sales.Date]
sales['dayofweek'] = sales['Date'].dt.day_name()


# ------------------------------------
# Trained prophet Model

# Convert dataframe into the format required by Prophet
df = sales[['Date', 'Sales_Dollars']]
print("df:", df.head())
print(df.info())
df.columns = ["ds", "y"]
prediction_size = 21  # 21 month are considered as training split and to compute metrics
train_df = df[:-prediction_size]
train_df2 = train_df.copy().set_index("ds")

# Apply Box-Cox Transf.
train_df2["y"], lambda_prophet = scs.boxcox(train_df2["y"])
train_df2.reset_index(inplace=True)


# Box-Cox invers transformation
def inverse_boxcox(y, lambda_):
    return np.exp(y) if lambda_ == 0 else np.exp(np.log(lambda_ * y + 1) / lambda_)

def make_comparison_dataframe(historical, forecast):
    """Join the history with the forecast.

       The resulting dataset will contain columns 'yhat', 'yhat_lower', 'yhat_upper' and 'y'.
    """
    return forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]].join(
        historical.set_index("ds")
    )

def forecast_prophet(forecasting_period=''):

    model = Prophet(changepoint_prior_scale= 0.9,
                    holidays_prior_scale = 0.5,
                    seasonality_mode = 'multiplicative',
                    seasonality_prior_scale= 10,
                    #weekly_seasonality=True,
                    #daily_seasonality = True,
                    yearly_seasonality = True,
                    interval_width=0.95)
    model.fit(train_df2)
    future = model.make_future_dataframe(periods=forecasting_period,
                                         freq='MS')
    forecast = model.predict(future)
    fig_comp = model.plot_components(forecast)

    # Invers Box-Cox Transf.
    for column in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast[column] = inverse_boxcox(forecast[column], lambda_prophet)

    df_merged = make_comparison_dataframe(df, forecast)

    model_acc = r2_score(df.y[-prediction_size:], forecast.yhat[-prediction_size:])
    model_mse = mean_squared_error(df.y[-prediction_size:], forecast.yhat[-prediction_size:])
    model_mae = mean_absolute_error(df.y[-prediction_size:], forecast.yhat[-prediction_size:])
    model_mape = mean_absolute_percentage_error(df.y[-prediction_size:], forecast.yhat[-prediction_size:])
    model_mape = model_mape.astype(float)

    return df_merged, fig_comp, model_acc, model_mse, model_mae, model_mape






