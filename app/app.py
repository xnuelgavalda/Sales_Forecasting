"""
Created on October 16 2023

@author: xavier Nuel Gavaldà
"""

#-------------------------------------------
# Imports
#-------------------------------------------

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
                             mean_absolute_percentage_error,
                             r2_score)

from joblib import load
import joblib
import base64
import sys

# Import functions from forecast.py
from forecast import forecast_prophet

# ---------------------------------------------------
# Building the Streamlit app
# ---------------------------------------------------

# Header

def main():
    st.write("<h1 style='text-align: center;'>Sales Forecasting Project</h1>",
             unsafe_allow_html=True)

    # Load data
    @st.cache_data
    def load_data(url):
        df = pd.read_csv(url,
                         index_col='Date')
        df.index = pd.to_datetime(df.index)
        df.reset_index(inplace=True)
        return df

    sales = load_data('./data/sales.csv')

    sales['year'] = [d.year for d in sales.Date]
    sales['month'] = [d.strftime('%b') for d in sales.Date]
    #sales['dayofweek'] = [d.strftime('%w') for d in sales.Date]
    sales['dayofweek'] = sales['Date'].dt.day_name()

    # Checkbox to show row data
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(sales)

    # Figure1: Sales vs Time
    fig1 = px.line(sales,
                   x='Date',
                   y='Sales_Dollars',
                   title='Sales vs Date',
                   labels={
                       'Sales_Dollars':'Sales ($)'
                          }
                   )

    fig1.update_layout(xaxis_tickangle=0,
                       hovermode='closest',
                       title_x=0.5)  # center the title

    fig1.update_xaxes(showline=True,
                      linewidth=0,
                      showgrid=True,
                      gridwidth=0.5)

    fig1.update_yaxes(showline=True,
                      linewidth=0,
                      showgrid=True,
                      gridwidth=0.5)

    st.plotly_chart(fig1,
                    use_container_width=True)

    #Figure2: Bar chart
    sales['year'] = sales['year'].astype(str)
    custom_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']

    fig2 = px.bar(sales,
                  x='month',
                  y='Sales_Dollars',
                  color='year',
                  barmode='group',
                  labels={
                      'Sales_Dollars':'Sales ($)',
                      'month':'Month',
                      'year':'Year'
                         },
                  title='Monthly Sales by Year'
                  )

    fig2.update_layout(xaxis=dict(categoryorder='array',
                                  categoryarray=custom_order),
                       xaxis_tickangle=0,
                       hovermode='closest',
                       title_x=0.5)

    fig2.update_xaxes(showline=True,
                      linewidth=0,
                      showgrid=True,
                      gridwidth=0.5)

    fig2.update_yaxes(showline=True,
                      linewidth=0,
                      showgrid=True,
                      gridwidth=0.5)

    st.plotly_chart(fig2,
                    use_container_width=True)

    # Statistics
    st.write("<h3 style='text-align: center;'>Statistics</h3>",
             unsafe_allow_html=True)

    # Figure3: Yearly Box plot
    fig3 = px.box(sales,
                  x='year',
                  y='Sales_Dollars',
                  labels={
                      'Sales_Dollars': 'Sales ($)',
                      'month': 'Month',
                      'year': 'Year'
                         },
                  title='Yearly Trend'
                  )

    fig3.update_layout(xaxis_tickangle=0,
                       hovermode='closest',
                       title_x=0.5)  # center the title

    fig3.update_xaxes(showline=True,
                      linewidth=0,
                      showgrid=True,
                      gridwidth=0.5)

    fig3.update_yaxes(showline=True,
                      linewidth=0,
                      showgrid=True,
                      gridwidth=0.5)

    st.plotly_chart(fig3,
                    use_container_width=True)

    # Figure4: Monthly Box plot
    fig4 = px.box(sales,
                  x='month',
                  y='Sales_Dollars',
                  labels={
                      'Sales_Dollars': 'Sales ($)',
                      'month': 'Month',
                      'year': 'Year'
                         },
                  title='Monthly Trend'
                  )

    fig4.update_layout(xaxis_tickangle=0,
                       hovermode='closest',
                       title_x=0.5)  # center the title

    fig4.update_xaxes(showline=True,
                      linewidth=0,
                      showgrid=True,
                      gridwidth=0.5)

    fig4.update_yaxes(showline=True,
                      linewidth=0,
                      showgrid=True,
                      gridwidth=0.5)

    st.plotly_chart(fig4,
                    use_container_width=True)

    # Figure5: Daily Box plot
    fig5 = px.box(sales,
                  x='dayofweek',
                  y='Sales_Dollars',
                  labels={
                      'Sales_Dollars': 'Sales ($)',
                      'dayofweek': 'Day of Week',
                         },
                  title='Daiy Trend'
                  )

    fig5.update_layout(xaxis_tickangle=0,
                       hovermode='closest',
                       title_x=0.5)  # center the title

    fig5.update_xaxes(showline=True,
                      linewidth=0,
                      showgrid=True,
                      gridwidth=0.5)

    fig5.update_yaxes(showline=True,
                      linewidth=0,
                      showgrid=True,
                      gridwidth=0.5)

    st.plotly_chart(fig5,
                    use_container_width=True)

    # Figure6: Sales vs Time

    # Define the custom order for months
    custom_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']

    fig6 = px.line(sales,
                   x='month',
                   y='Sales_Dollars',
                   color='year',
                   title='Seasonality - Monthly Sales',
                   labels={
                       'Sales_Dollars': 'Sales ($)',
                       'month': 'Month',
                       'year':'Year'
                          }
                   )

    fig6.update_layout(xaxis=dict(categoryorder='array',
                                  categoryarray=custom_order),
                       xaxis_tickangle=0,
                       hovermode='closest',
                       title_x=0.5)

    fig6.update_layout(xaxis_tickangle=0,
                       hovermode='closest',
                       title_x=0.4)  # center the title

    fig6.update_xaxes(showline=True,
                      linewidth=0,
                      showgrid=True,
                      gridwidth=0.5)

    fig6.update_yaxes(showline=True,
                      linewidth=0,
                      showgrid=True,
                      gridwidth=0.5)

    st.plotly_chart(fig6,
                    use_container_width=True)

    # ------------------------------------
    # Forecast Section

    df = sales[['Date','Sales_Dollars']]
    df.columns = ["ds", "y"]
    df["y"], lambda_prophet = scs.boxcox(df["y"])

    st.write("<h3 style='text-align: center;'>Forecast</h3>",
             unsafe_allow_html=True)

    # Custom CSS to set the size of the number input field
    custom_css = """
    <style>
    /* Adjust the size of the number input */
    input[type=number] {
    width: 50px; /* Set your preferred width */
    height: 30px; /* Set your preferred height */
    padding: 5px; /* Set padding as needed */
    }
    </style>
    """

    st.markdown(custom_css, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown("[Xavier Nuel Gavaldà](https://portfolio-xaviernuelgavalda-148e3f5b58a0.herokuapp.com/)")

    # Forecast

    st.write("<h4 style='text-align: center;'>Note: The training data set is setup for the last 21 months of the dataset, that is from January 2014</h4>",
             unsafe_allow_html=True)

    default_value = 21
    forecasting_month = st.number_input(label="Forecast period months desired (from January 2014)",min_value=0, max_value=42, value=default_value)
    prediction_size = 21

    # Plotting interactive results with confidence intervals
    def show_forecast_with_confidence(cmp_df, num_predictions, num_values, title):
        """
        Objective: Visualize the forecast with confidence intervals
        cmp_df: dataset
        num_predictions: trained months
        num_values: forecast months
        title: plot title in string
        """
        def create_go(name, column, num, **kwargs):
            points = cmp_df.tail(num)
            args = dict(name=name, x=points.index, y=points[column], mode="lines")
            args.update(kwargs)
            return go.Scatter(**args)

        lower_bound = create_go(
            "Lower Bound 80% Confidence Interval",
            "yhat_lower",
            num_predictions,
            line=dict(width=0),
            marker=dict(color="gray"),
        )

        upper_bound = create_go(
            "Upper Bound 80% Confidence Interval",
            "yhat_upper",
            num_predictions,
            line=dict(width=0),
            marker=dict(color="gray"),
            fillcolor="rgba(68, 68, 68, 0.3)",
            fill="tonexty",
        )

        forecast = create_go(
            "Forecast",
            "yhat",
            num_predictions,
            line=dict(color="rgb(31, 119, 180)")
        )

        actual = create_go(
            "Actual",
            "y",
            num_values,
            marker=dict(color="red")
        )

        # In this case the order of the series is important because of the filling
        data = [lower_bound, upper_bound, forecast, actual]
        layout = go.Layout(yaxis=dict(title="Sales ($)"),
                           xaxis=dict(title="Date"),
                           title=title,
                           showlegend=True)
        fig = go.Figure(data=data,
                        layout=layout)
        fig.update_layout(width=1000,
                          height=600,
                          xaxis_tickangle=0,
                          hovermode='closest',
                          legend=dict(yanchor="top",y=0.95,
                                      xanchor="right", x=0.4),
                          title_x=0.4)  # center the title

        fig.update_xaxes(showline=True,
                         linewidth=0,
                         showgrid=True,
                         gridwidth=0.5)

        fig.update_yaxes(showline=True,
                         linewidth=0,
                         showgrid=True,
                         gridwidth=0.5)
        return fig

    # Plotting interactive results with confidence intervals
    def show_forecast(cmp_df, num_predictions, num_values, title):
        """
        Objective: Visualize the forecast without confidence intervals
        cmp_df: dataset
        num_predictions: trained months
        num_values: forecast months
        title: plot title in string
        """
        def create_go(name, column, num, **kwargs):
            points = cmp_df.tail(num)
            args = dict(name=name, x=points.index, y=points[column], mode="lines")
            args.update(kwargs)
            return go.Scatter(**args)

        forecast = create_go(
            "Forecast",
            "yhat",
            num_predictions,
            line=dict(color="rgb(31, 119, 180)")
        )

        actual = create_go(
            "Actual",
            "y",
            num_values,
            marker=dict(color="red")
        )

        # In this case the order of the series is important because of the filling
        data = [forecast, actual]

        layout = go.Layout(yaxis=dict(title="Sales ($)"),
                           xaxis=dict(title="Date"),
                           title=title,
                           showlegend=True)
        fig = go.Figure(data=data,
                        layout=layout)

        fig.update_layout(width=1000,
                          height=600,
                          xaxis_tickangle=0,
                          hovermode='closest',
                          legend=dict(yanchor="top", y=0.95,
                                      xanchor="right", x=0.2),
                          title_x=0.4)  # center the title

        fig.update_xaxes(showline=True,
                         linewidth=0,
                         showgrid=True,
                         gridwidth=0.5)

        fig.update_yaxes(showline=True,
                         linewidth=0,
                         showgrid=True,
                         gridwidth=0.5)
        return fig


    forecast_result = forecast_prophet(forecasting_period=forecasting_month)

    # Metrics
    def plot_metrics():
        st.write("<h4 style='text-align: center;'>Metrics</h4>",
                 unsafe_allow_html=True)

        # Create a container to hold the metrics
        metrics_container = st.empty()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(label="Accuracy", value='{:2.2f}%'.format(forecast_result[2]))
        col2.metric(label="MAE", value='{:.1f}'.format(forecast_result[4]))
        col3.metric(label="MAPE", value='{:2.2%}'.format(forecast_result[5]))
        col4.metric(label="MSE", value='{:.0f}'.format(abs(forecast_result[3])))

        # Update the metrics container with custom CSS
        metrics_container.markdown(
            """
            <style>
            .metrics-row {
                display: flex;
                justify-content: space-between;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        return col1, col2, col3, col4


    if st.button("Show Forecast without Confidence Intervals"):
        title ='Forecast without Confidence Intervals'
        fig = show_forecast(forecast_result[0],
                            prediction_size,
                            forecasting_month,
                            title)
        st.plotly_chart(fig,
                        use_container_width=True)
        plot_metrics()

        st.write("<h5 style='text-align: center;'>Forecast components</h5>",
                 unsafe_allow_html=True)
        st.pyplot(fig=forecast_result[1])

    if st.button("Show Forecast with Confidence Intervals"):
        title='Forecast with Confidence Interval'
        fig = show_forecast_with_confidence(forecast_result[0],
                                            prediction_size,
                                            forecasting_month,
                                            title)
        st.plotly_chart(fig,
                            use_container_width=True)
        plot_metrics()

        st.write("<h5 style='text-align: center;'>Forecast Components</h5>",
                 unsafe_allow_html=True)
        st.pyplot(fig=forecast_result[1])


if __name__ == '__main__':
    main()