import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

def plot_ts(df, color='RegionName'):
    '''
    Docstring:     plot time series with plotly express.
    Signature:     plot_ts(
                   df=none,
                   color='RegionName'
                   )
    Parameters:    df: pandas dataframe
    Return:        plotly express figure
    '''
    # plot time series
    fig = px.line(df, x='time', y="value", color=color,
                  title='Home Value by Zip Code')
    fig.update_xaxes(showline=True, linewidth=1, linecolor='gray', mirror=True, title='year')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='gray', mirror=True)
    fig.show()
    

def get_data_by_zc(df, zip_code):
    '''
    Docstring:     Subset dataframe by zip code and convert to time series.
    Signature:     get_data_by_zc(
                   df = none,
                   zip_code = none
                   )
    Parameters:    df: pandas dataframe
                   zip_code: int, zip code     
    Return:        Pandas dataframe
    '''

    tmp = (df[(df.RegionName == zip_code)][['time','value']])\
           .set_index('time').copy() 
    return tmp


def px_displot(df, stdized=False):
    '''
    Docstring:     Use plotly express figure factory to plot data distribution.
    Signature:     px_displot(
                   df = none,
                   stdizd=False
                   )
    Parameters:    df: pandas dataframe
                   stdized: bool.  If true plot data after normalizing with
                   MinMaxScaler.
    Return:        Plotly express figure_factory displot.                   
    '''

    zip_codes = list(set(df.RegionName))
       
    x1 = (df[(df.RegionName == zip_codes[0])][['time','value']])\
                           .set_index('time').copy()
    x2 = (df[(df.RegionName == zip_codes[1])][['time','value']])\
                           .set_index('time').copy()
    x3 = (df[(df.RegionName == zip_codes[2])][['time','value']])\
                           .set_index('time').copy()
    x4 = (df[(df.RegionName == zip_codes[3])][['time','value']])\
                           .set_index('time').copy()
    x5 = (df[(df.RegionName == zip_codes[4])][['time','value']])\
                           .set_index('time').copy()
    
    hist_data = [x1.values.ravel(), x2.values.ravel(), x3.values.ravel(), 
                x4.values.ravel(), x5.values.ravel()]
        
    if stdized:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        x1_s = scaler.fit_transform(x1.values)
        x2_s = scaler.fit_transform(x2.values)
        x3_s = scaler.fit_transform(x3.values)
        x4_s = scaler.fit_transform(x4.values)
        x5_s = scaler.fit_transform(x5.values)
        
        hist_data = [x1_s.ravel(), x2_s.ravel(), x3_s.ravel(),
                     x4_s.ravel(), x5_s.ravel()]
                     
    # Create distplot with curve_type set to 'normal'
    fig = ff.create_distplot(hist_data, group_labels=zip_codes,
                             show_hist=False,
                             curve_type='normal',
                             show_rug=False)
    # Add title
    fig.update_layout(title_text='Home Values Data Distribution by Zip Code',
                      xaxis_title_text='home value',
                      yaxis_title_text='density')
    fig.update_xaxes(showline=True, linewidth=1, linecolor='gray', mirror=True,
                     title_text='zip code')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='gray', mirror=True)

    if stdized:
        fig.update_layout(title_text='Normalized Home Values Data Distribution by Zip Code',
                          xaxis_title_text='normalized home value')
        
    fig.show()     
    
    
def adfuller_test_df(ts, zc, txts):
    """
    Docstring:     Returns the AD Fuller Test Results and p-values for the null hypothesis
                   that there the data is non-stationary (that there is a unit root in the data)
    Signature:     adfuller_test_df(
                   ts=none
                   zc=none
                   txts=none
                   )
    Parameters:    ts: dataframe with time as index
                   zc: int, zip code
                   txts: string as part of index
    Return:        Dataframe                      
    """
    
    df_res = adfuller(ts)
    names = ['Test Statistic','p-value','#Lags Used','# of Observations Used']
    res  = dict(zip(names,df_res[:4]))
    res['p<.05'] = res['p-value']<.05
    res['Stationary?'] = res['p<.05']
    
    return pd.DataFrame(res,index=[f'{zc}, {txts}'])


def stationarity_diff_chk(df, zipcode, periods=1, plot=True, diff_n=1):
    '''
    Docstring:     Check if data is stationary and plot before and after differencing data.
    Signature:     stationarity_diff_chk(
                   df=none,
                   cities=none
                   periods=1,
                   plot=True,
                   diff_n=1
                   )
    Parameters:    df: dataframe
                   cities: list of cities
                   periods: int set to 1
                   plot: Boolean, if False will not plot data
                   diff_n: int, number of time to difference the data
    Return:        Adfuller test result and line plots.                   
    '''
    
    adf_res = []
    
    for zc in zipcode:
        tmp = get_data_by_zc(df, zc)
        
        # calculate differencing statistics
        for i in range(1, diff_n+1):
            if i == 1:
                df_diff = tmp.diff(periods=periods)
                df_diff.dropna(inplace=True)
            elif i > 1:
                df_diff = df_diff.diff(periods=periods)
                df_diff.dropna(inplace=True)
        
        adf_results = adfuller_test_df(df_diff, zc, 'Differencing')
        adf_res.append(adf_results)
    
        if plot:
            # plot rolling statiistic
            if adf_results['p-value'].values < 0.05:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=tmp.index, y=tmp['value'], mode='lines',name='Observed'))
                fig.add_trace(go.Scatter(x=df_diff.index, y=df_diff.value, mode='lines',
                                         name='Differencing'))
                if diff_n == 1:
                    title = f'First Order Differencing and Observed Home Values in {zc}'
                elif diff_n ==2:
                    title = f'Second Order Differencing and Observed Home Values in {zc}'
            
                fig.update_layout(width=750,
                                  height=400,
                                  title=title,
                                  xaxis_title='year',
                                  yaxis_title='value'
                                 )
                fig.update_xaxes(showline=True, linewidth=1, linecolor='gray', mirror=True)
                fig.update_yaxes(showline=True, linewidth=1, linecolor='gray', mirror=True)
                fig.show()
    
    return pd.concat(adf_res)

def plot_acf_pacf(df, diff_1_zc, diff_2_zc, diff_3_zc, zip_codes, p_type='acf'):
    '''
    Docstring:    Plot ACF or PACF
    Signature:    plot_acf_pacf(
                  df=none,  
                  diff_1_zc=none,
                  diff_2_zc=none,
                  diff_3_zc=none,
                  zip_codes=none,
                  p_type='acf'
                  )
    Parameters:   df:        dataframe
                  diff_1_zc: list, zip codes that require 1st order differencing  
                  diff_2_zc: list, zip codes that require 2nd order differencing  
                  diff_3_zc: list, zip codes that require 2nd order differencing  
                  zipcodes:  list of zip codes
                  type:      string, either 'acf' or 'pacf'
    Returns:      ACF or PACF plots                  
    '''
    # plot the autocorrelation by zip code
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(15, 15))
        plot_number = 1
        
        # set font size
        title_s = 15
        xlabel_s = 13
        ylabel_s = 14
        xyticks_s = 12
        # apply differencing and plot ACF
        for zc in zip_codes:
            if zc in diff_1_zc:
                tmp = get_data_by_zc(df, zc)
                axn = plt.subplot(3, 2, plot_number)
                if p_type == 'acf':
                    plot_acf(tmp.diff().dropna(), ax=axn, lags=40)
                    plt.title(f'Autocorrelation First Order Differencing, {zc}', size=title_s)
                    plt.ylabel('Autocorrelation', size=ylabel_s)
                else:
                    plot_pacf(tmp.diff().dropna(), ax=axn, lags=40)
                    plt.title(f'Partial Autocorrelation First Order Differencing, {zc}', size=title_s)
                    plt.ylabel('Partial Autocorrelation', size=ylabel_s)
            elif zc in diff_2_zc:
                tmp = get_data_by_zc(df, zc)
                axn = plt.subplot(3, 2, plot_number)
                if p_type == 'acf':
                    plot_acf(tmp.diff().diff().dropna(), ax=axn, lags=40)
                    plt.title(f'Autocorrelation Second Order Differencing, {zc}', size=title_s)
                    plt.ylabel('Autocorrelation', size=ylabel_s)
                else:
                    plot_pacf(tmp.diff().diff().dropna(), ax=axn, lags=40)
                    plt.title(f'Partial Autocorrelation Second Order Differencing, {zc}', size=title_s)
                    plt.ylabel('Partial Autocorrelation', size=ylabel_s)
            elif zc in diff_3_zc:
                tmp = get_data_by_zc(df, zc)
                axn = plt.subplot(3, 2, plot_number)
                if p_type == 'acf':
                    plot_acf(tmp.diff().diff().diff().dropna(), ax=axn, lags=40)
                    plt.title(f'Autocorrelation Third Order Differencing, {zc}', size=title_s)
                    plt.ylabel('Autocorrelation', size=ylabel_s)
                else:
                    plot_pacf(tmp.diff().diff().diff().dropna(), ax=axn, lags=40)
                    plt.title(f'Partial Autocorrelation Third Order Differencing, {zc}', size=title_s)
                    plt.ylabel('Partial Autocorrelation', size=ylabel_s)
            plot_number = plot_number + 1
            plt.xlabel('Lag', size=xlabel_s)
            plt.xticks(size=xyticks_s)
            plt.yticks(size=xyticks_s)
        plt.subplots_adjust(hspace=.4)
        plt.show()


def top_output(model_output, zip_codes):
    '''
    Docstring:     Return top 5 zip codes with the lowest AIC score.
    Signature:     top_output(
                   model_output=none,
                   zip_codes=none
                   )
    Parameters:    model_output: dataframe    .                  
                   zip_codes: list of zip codes. 
    Return:        dataframe
    '''
    
    df_model_top_res = []
    for zc in zip_codes:
        # extract top combination of pdq and PDQ for each zip code
        tmp = model_output[model_output['zc'] == zc].sort_values('AIC').head(1)
        df_model_top_res.append(tmp)
    
    return pd.concat(df_model_top_res)


def gen_arima_params(df, zip_codes, pdq, enf_stationarity=False):
    '''
    Docstring:     Grid search pdq for arima model.
    Signature:     gen_arima_params(
                   df=none,
                   zip_codes=none,
                   pdq=none,
                   enf_stationarity=False
                   )
    Parameters:    df: dataframe,
                   zip_codes: list of zip codes,
                   pdq: list of combinations of pdq,
                   enf_staionarity: boolean
    Return:        Dataframe with list of AIC from the arima model.                   
    '''
    import warnings
    warnings.filterwarnings("ignore")
    
    # define empty list to store results
    ans=[]
   
    for zc in zip_codes:
        tmp = get_data_by_zc(df, zc)

        for i in pdq:
            try:
                # fit model
                model = sm.tsa.SARIMAX(tmp, 
                                   order=i,
                                   enforce_stationarity=enf_stationarity,
                                   enforce_invertibility=False)
                output = model.fit()
                ans.append([zc, i, output.aic])
                    
            except:
                continue
                    
    # convert results to dataframe
    df_tmp = pd.DataFrame(ans, columns=['zc', 'pdq', 'AIC'])
    
    return top_output(df_tmp, zip_codes)


def fit_arima_model(df, arima_params, zc, enf_stationarity=False, summ=True,
                    plot=True, model_out=False):
    '''
    Docstring:     Fit the arima model with the optimum pdq, print model summary
                   and plot model diagnostic results.
    Signature      fit_arima_model(
                   df=none,
                   arima_params: none,
                   zc: none,
                   enf_stationarity=False,
                   summ=True,
                   plot=True,
                   model_out=False
                   )
    Parameters:    df: dataframe.
                   arima_params: dataframe which contains optimum pdq for each zip code. 
                   zc: int, zip code.
                   enf_stationarity: Boolean.
                   summ: Boolean.  If True print out model summary.
                   plot: Boolean.  If True plot diagnostic figures.
                   model_out: If True return model output.
    Return:        model summary, model diagnostic plots and model output.                   
    '''
    
    import warnings
    warnings.filterwarnings("ignore")
    
    tmp = get_data_by_zc(df, zc)
    
    # get model parameters
    model_param = arima_params[arima_params['zc'] == zc]
        
    # assign model parameters
    for i in model_param.iterrows():
        pdq = i[1]['pdq']
        
    # fit model    
    model = sm.tsa.SARIMAX(tmp,
                           order=pdq,
                           enforce_stationarity=enf_stationarity,
                           enforce_invertibility=False)
    
    output = model.fit()
    
    if summ:
        print(f'\033[1m*** ARIMA Parameters, {pdq} ***')
        print(f'\033[1m*** Coefficients Statistics, {zc} ***')
        display(output.summary().tables[1])
        
    if plot:
        # plot diagnostics 
        print(f'\033[1m*** ARIMA Diagnostics Plot, {zc} ***')
        with plt.style.context('seaborn-darkgrid'):
            fig = output.plot_diagnostics(figsize=(10,10))
            fig.tight_layout(pad=5.0)
            axes = fig.get_axes()
            for ax in axes:
                if 'Standardized' in str((ax.title)):
                    ax.tick_params(axis='x', labelrotation=90, size=11)
                else:
                    ax.tick_params(axis='both', size=11)
            plt.show()

    # return model output
    if model_out:
        return output
    
    
    
def one_step_fc_arima(df, top_model_params, zc, pred_start_date,
                       enf_stationarity=False, dynamic=False):
    '''
    Docstring:     Fit arima model and get predictions.  In addition, calculate
                   MSE and RMSE and plot the actual vs predicted values with confident intervals.
    Signature:     one_step_fc_arima(
                   df=none,
                   top_model_params=none,
                   zc=none,
                   pred_start_date=none,
                   enf_stationarity=False,
                   dynamic=False
                   )
    Parameters:    df: dataframe
                   top_model_params: df of best pdq for each zip code
                   zc: int, zip code
                   pred_start_date: string of date
                   enf_stationarity: boolean
                   dynamic: boolean
    Return:        MSE, RMSE and plotly go figure of actual vs predicted plot.                   
    '''
    
    output = fit_arima_model(df,top_model_params, zc, enf_stationarity=enf_stationarity,
                             summ=False,plot=False,model_out=True)
    
    # get prediction
    pred = output.get_prediction(start=pd.to_datetime(pred_start_date), dynamic=dynamic)
    
    # get prediction confident intervals
    pred_conf = pred.conf_int()
    
    # observed data
    tmp = get_data_by_zc(df, zc)
    
    # calculate mean square error
    predicted = pred.predicted_mean
    observed = tmp[pred_start_date:].values.ravel()
    from sklearn.metrics import mean_squared_error
    mse = round(mean_squared_error(observed,predicted),2)
    print(f'The mean square error of our forcast is : {mse}.')
    print(f'The root mean square error of our forcast is {round(np.sqrt(mse),2)}.')

    # plot one-step ahead forcast
    fig = go.Figure()
    # plot observed data
    fig.add_trace(go.Scatter(x=tmp.index, y=tmp['value'], mode='lines', name='Observed'))

    # plot upper confident interval
    fig.add_trace(go.Scatter(name='Upper Bound', x=pred_conf.index, y=pred_conf.iloc[:, 0],
                         mode='lines', opacity=.3, 
                         marker=dict(color="#73BD92"),# line=dict(width=2),
                         showlegend=False))
    # plot lower confident interval
    fig.add_trace(go.Scatter(name='Lower Bound', x=pred_conf.index, y=pred_conf.iloc[:, 1],
                         mode='lines', opacity=.3,
                         marker=dict(color="#73BD92"), #line=dict(width=1),
                         fill='tonexty',
                         showlegend=False,   
                         fillcolor='rgba(68, 68, 68, 0.3)'))
    
    # plot prediction
    fig.add_trace(go.Scatter(x=pred.predicted_mean.index,y=pred.predicted_mean.values,
                         mode='lines', name='One-step ahead forecast'))

    # update figure layout
    fig.update_xaxes(rangeslider_visible=True, showline=True, linewidth=1,
                 linecolor='gray', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='gray', mirror=True)
    fig.update_layout(legend=dict(orientation="h",
                              yanchor="bottom",
                              y=1.02,
                              xanchor="right",
                              x=1
                             ),
                     xaxis_title='Date', 
                     yaxis_title='Home Value',
                     title=f'One-Step Ahead Forcast, {zc}')
    fig.show()
    
    return {'zc': zc, 'dynamic':dynamic,'output':output,
            'mse':mse, 'rmse':round(np.sqrt(mse),2)}


def plot_forecast_rmse(df, x, y, title, color):
    '''
    Docstring:     Plot zip code rmse from dynamice one-step ahead forecast.
    Signature:     plot_forecast_rmse(
                   df=: none,
                   x=none,
                   y=none',
                   color='dynmaice'
                   )
    Parameters:    df: pandas dataframe
                   x:  str, col. name
                   y:  list\array
                   color: string
    Return:        plotly express scatter figure
    '''
    fig = px.scatter(df, x=x, y=y, color=color, 
                     title=title)
    fig.update_xaxes(showline=True, linewidth=1, linecolor='gray', mirror=True,
                     type='category', title_text='zip code')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='gray', mirror=True)
    fig.update_traces(marker=dict(size=12,
                                  line=dict(width=2,
                                  color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()

    
def gen_sarima_params(df, zip_codes, pdq, PDQ, enf_stationarity=False):
    '''
    Docstring:     Grid search pdq and PDQ for sarima model.
    Signature:     gen_sarima_params(
                   df=none,
                   =none,
                   pdq=none,
                   PDQ=none,
                   enf_stationarity=False
                   )
    Parameters:    df: dataframe,
                   zip_codes: list of zip codes,
                   pdq: list of combinations of pdq,
                   PDQ: list of combinations of PDQ
                   enf_staionarity: boolean
    Return:        Dataframe with list of AIC from the arima model.                   
    '''    
    import warnings
    warnings.filterwarnings("ignore")
    
    # define empty list to store results
    ans=[]
   
    for zc in zip_codes:
        tmp = get_data_by_zc(df, zc)
        for i in pdq:
            for j in PDQ:
                try:
                    # fit model
                    model = sm.tsa.SARIMAX(tmp, 
                                   order=i,
                                   seasonal_order=j,
                                   enforce_stationarity=enf_stationarity,
                                   enforce_invertibility=False)
                    output = model.fit()
                    ans.append([zc, i, j, output.aic])
                    
                except:
                    continue
                    
    # convert results to dataframe
    df_tmp = pd.DataFrame(ans, columns=['zc', 'pdq', 'PDQ', 'AIC'])
    
    return top_output(df_tmp, zip_codes)   
    
    
def fit_sarima_model(df, sirama_params, zc, enf_stationarity=False, summ=True,
                     plot=True, model_out=False):
    '''
    Docstring:     Fit the arima model with the optimum pdq, print model summary
                   and plot model diagnostic results.
    Signature      fit_arima_model(
                   df=none,
                   sarima_params: none,
                   zc: none,
                   enf_stationarity=False,
                   summ=True,
                   plot=True,
                   model_out=False
                   )
    Parameters:    df: dataframe.
                   sarima_params: dataframe which contains optimum pdq for each zip code. 
                   zc: int, zip code.
                   enf_stationarity: Boolean.
                   summ: Boolean.  If True print out model summary.
                   plot: Boolean.  If True plot diagnostic figures.
                   model_out: If True return model output.
    Return:        model summary, model diagnostic plots and model output.                   
    '''    
    import warnings
    warnings.filterwarnings("ignore")
    
    tmp = get_data_by_zc(df, zc)
    
    # get model parameters
    model_param = sirama_params[sirama_params['zc'] == zc]
        
    # assign model parameters
    for i in model_param.iterrows():
        pdq = i[1]['pdq']
        PDQ = i[1]['PDQ']
        
    # fit model    
    model = sm.tsa.SARIMAX(tmp,
                           order=pdq,
                           seasonal_order=PDQ,
                           enforce_stationarity=enf_stationarity,
                           enforce_invertibility=False)
    
    output = model.fit()
    
    if summ:
        print(f'\033[1m*** ARIMA Parameters, {pdq}, {PDQ} ***')
        print(f'\033[1m*** Coefficients Statistics, {zc} ***')
        display(output.summary().tables[1])
        
    if plot:
        # plot diagnostics 
        print(f'\033[1m*** SARIMAX Diagnostics Plot, {zc} ***')
        with plt.style.context('seaborn-darkgrid'):
            fig = output.plot_diagnostics(figsize=(12,12))
            fig.tight_layout(pad=5.0)
            axes = fig.get_axes()
            for ax in axes:
                if 'Standardized' in str((ax.title)):
                    ax.tick_params(axis='x', labelrotation=90, size=11)
                else:
                    ax.tick_params(axis='both', size=11)
#             output.plot_diagnostics(figsize=(10, 8))
            plt.show()
        
    # return model output
    if model_out:
        return output    
    
    
def one_step_fc_sarima(df, sarima_params, zc, pred_start_date,
                       enf_stationarity=False, dynamic=False):
    '''
    Docstring:     Fit sarima model and get predictions.  In addition, calculate
                   MSE and RMSE and plot the actual vs predicted values with confident intervals.
    Signature:     one_step_fc_sarima(
                   df=none,
                   sarima_params=none,
                   zc=none,
                   pred_start_date=none,
                   enf_stationarity=False,
                   dynamic=False
                   )
    Parameters:    df: dataframe
                   top_model_params: df of best pdq for each zip code
                   zc: int, zip code
                   pred_start_date: string of date
                   enf_stationarity: boolean
                   dynamic: boolean
    Return:        MSE, RMSE and plotly go figure of actual vs predicted plot.                   
    '''    
    output = fit_sarima_model(df,sarima_params, zc,
                             enf_stationarity=enf_stationarity,
                             summ=False,plot=False,model_out=True)
    
    # get prediction
    pred = output.get_prediction(start=pd.to_datetime(pred_start_date), dynamic=dynamic)
    
    # get prediction confident intervals
    pred_conf = pred.conf_int()
    
    # observed data
    tmp = get_data_by_zc(df, zc)
    
    # calculate mean square error
    predicted = pred.predicted_mean
    observed = tmp[pred_start_date:].values.ravel()
    mse = ((predicted - observed)**2).mean()
    print(f'The mean square error of our forcast is {round(mse,2)}.')
    print(f'The root mean square error of our forcast is {round(np.sqrt(mse),2)}.')
    
    # plot one-step ahead forcast
    fig = go.Figure()
    # plot observed data
    fig.add_trace(go.Scatter(x=tmp.index, y=tmp['value'], mode='lines', name='Observed'))

    # plot upper confident interval
    fig.add_trace(go.Scatter(name='Upper Bound', x=pred_conf.index, y=pred_conf.iloc[:, 0],
                         mode='lines', opacity=.3, 
                         marker=dict(color="#73BD92"),# line=dict(width=2),
                         showlegend=False))
    # plot lower confident interval
    fig.add_trace(go.Scatter(name='Lower Bound', x=pred_conf.index, y=pred_conf.iloc[:, 1],
                         mode='lines', opacity=.3,
                         marker=dict(color="#73BD92"), #line=dict(width=1),
                         fill='tonexty',
                         showlegend=False,   
                         fillcolor='rgba(68, 68, 68, 0.3)'))
    
    # plot prediction
    fig.add_trace(go.Scatter(x=pred.predicted_mean.index,y=pred.predicted_mean.values,
                         mode='lines', name='One-step ahead forecast'))

    # update figure layout
    fig.update_xaxes(rangeslider_visible=True, showline=True, linewidth=1,
                     linecolor='gray', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='gray', mirror=True)
    fig.update_layout(legend=dict(orientation="h",
                              yanchor="bottom",
                              y=1.02,
                              xanchor="right",
                              x=1
                             ),
                     xaxis_title='Date', 
                     yaxis_title='Home Value',
                     title=f'One-Step Ahead Forcast, {zc}')
    fig.show()    
    
    return {'zc': zc, 'dynamic':dynamic, 'output':output,
            'mse':mse, 'rmse':round(np.sqrt(mse),2)}
    
    
def get_a_s_fc(df, zc, start, end, afc_param=None, sfc_param=None, sarima=False, cost=20000):
    '''
    Docstring:    Get and plot model forecast results.
    Signature:    get_a_s_fc(
                  df=none,
                  zc=none,
                  start=none,
                  end=none,
                  afc_param=None,
                  sfc_param=None,
                  sarima=False,
                  cost=20000
    Parameters:   df: Pandas dataframe
                  zc: int, zip code
                  start: date
                  end: date
                  afc_param: Pandas dataframe
                  s_fc_param: dataframe
                  sarima:False                  
                  cost: int, general repair costs
                  )
    Return:       Forecast values, confident interval values and plotly figure                      
    '''
    import warnings
    warnings.filterwarnings("ignore")
    tmp = get_data_by_zc(df, zc)
    
    # fit model
    if sarima:
        pdq = sfc_param[sfc_param['zc'] == zc]['pdq'].values[0]
        PDQ = sfc_param[sfc_param['zc'] == zc]['PDQ'].values[0]
        model = sm.tsa.SARIMAX(tmp,
                           order=pdq,
                           seasonal_order=PDQ,
                           enforce_stationarity=False,
                           enforce_invertibility=False)
    
        output = model.fit()
    else:
        pdq = afc_param[afc_param['zc'] == zc]['pdq'].values[0]
        model = sm.tsa.SARIMAX(tmp,
                           order=pdq,
                           enforce_stationarity=False,
                           enforce_invertibility=False)
        output = model.fit()
    
    
    # Get forecast x steps ahead in future
    steps = int(round((pd.to_datetime(end)-pd.to_datetime(start))/np.timedelta64(1,'M'), 0))
    fc = output.get_forecast(steps=steps)
    

    # Get confidence intervals of forecasts
    fc_conf = fc.conf_int()
    
    l_conf_v = round(fc_conf[end[0:7]]['lower value'][0],0)
    u_conf_v = round(fc_conf[end[0:7]]['upper value'][0],0)
    
    # plot forcast
    fig = go.Figure()
    # plot observed data
    fig.add_trace(go.Scatter(x=tmp.index, y=tmp['value'], mode='lines', name='Observed'))

    # plot upper confident interval
    fig.add_trace(go.Scatter(name='Upper Bound', x=fc_conf.index, y=fc_conf.iloc[:, 0],
                             mode='lines', opacity=.3, 
                             marker=dict(color="#73BD92"),
                             showlegend=False))
                  
    # plot lower confident interval
    fig.add_trace(go.Scatter(name='Lower Bound', x=fc_conf.index, y=fc_conf.iloc[:, 1],
                             mode='lines', opacity=.3,
                             marker=dict(color="#73BD92"),
                             fill='tonexty',
                             showlegend=False,   
                             fillcolor='rgba(68, 68, 68, 0.3)'))
    
    # plot forecast
    fig.add_trace(go.Scatter(x=fc.predicted_mean.index,y=fc.predicted_mean.values,
                             mode='lines', name='Forecast'))

    # update figure layout
    fig.update_xaxes(rangeslider_visible=True, showline=True, linewidth=1,
                     linecolor='gray', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='gray', mirror=True)
    fig.update_layout(legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1.02,
                                  xanchor="right",
                                  x=1),
                      xaxis_title='date', 
                      yaxis_title='home value',
                      title=f'{zc} Forecast')
    fig.show()    
    
    tmp = cal_roi(df, zc, fc.predicted_mean.values, fc_conf, start, end,  l_conf_v, u_conf_v, cost)
   
    return tmp


def cal_roi(df, zc, fc_res, fc_conf, purchase_date, sale_date, l_conf_v, u_conf_v, cost=20000):
    '''
    Doctring:    Caluculate ROI (fc_price - (.035*(org_price) + 20000) / (.035*(org_price) + 20000)
    Signature:   cal_roi(
                 df=none,
                 zc=int,
                 fc_res=none,
                 fc_conf=none,
                 purchase_date=none,
                 sale_date=none,
                 l_conf_v=none
                 u_conf_v=none
                 cost=20000
                 )
    Parameters:  df: pandas dataframe
                 fc_res: list/array
                 fc_conf: pandas dataframe
                 purchase_date: string
                 sale_date: string
                 l_conf_v: float, forecast lower conf value at end date
                 u_conf_v: float, forecast upper conf value at end date
                 cost: int, genearl repair cost
    Return:      pandas dataframe                 
    '''
    # get purchase price
    tmp = get_data_by_zc(df, zc)
    p_price = tmp[purchase_date:]
    cost=cost
            
    # calculate mean roi
    s_price = round(fc_res[-1], 0)
    mean_roi = round((s_price - ((0.035* p_price)+cost+p_price)) / ((0.035* p_price)+cost+p_price),2)
    
    # calculate roi based on upper confident interval
    s_price_u = fc_conf[sale_date:]['upper value'].values
    upper_roi = round((s_price_u - ((0.035* p_price)+cost+p_price)) / ((0.035* p_price)+cost+p_price),2)
    
    # calculate roi based on lower confident interval
    s_price_l = fc_conf[sale_date:]['lower value'].values
    lower_roi = round((s_price_l - ((0.035* p_price)+cost+p_price)) / ((0.035* p_price)+cost+p_price), 2)                       

    # put results in dict
    roi_dict = {'zip_code': zc, 'mean_roi': mean_roi.value[0], 'upper_roi': upper_roi.value[0],
                'lower_roi': lower_roi.value[0], f'forecast_price_{sale_date}': s_price,
                f'actual_price_{purchase_date}': p_price.value[0], 'lower_conf': l_conf_v, 'upper_conf': u_conf_v}
                
            
    return pd.DataFrame([roi_dict.values()], columns=roi_dict.keys())


def plot_roi_by_model(df, x, y, color, title, orient='h'):
    '''
    Docstring:    Plot ROI by model
    Signature:    plot_roi_by_model(
                  df=none,
                  x=none,
                  y=none,
                  color=none,
                  title=none,
                  orient='h'
                  )
    Parameters:   df: dataframe
                  x: string, panda column name 
                  y: string, panda column name 
                  color: string, panda column name
                  title: string
                  orient: string, 'h' or 'v'
    Return:       Plotly grouped bar plot       
    '''
    # set vertical line
    if x == 'mean_roi':
        shapes = [dict(type='line', xref= 'x', x0= 0.1, x1= 0.1, yref= 'y', y0= -.5, y1= 4.5,
                      line= {'color':'green', 'width': 1, 'dash':'dashdot'})]
    elif x =='lower_roi':
        shapes = [dict(type='line', xref= 'x', x0= -0.4, x1= -0.4, yref= 'y', y0= -.5, y1= 4.5,
                      line= {'color':'green', 'width': 1, 'dash':'dashdot'})]        
    
    # plot bar chart
    fig = px.bar(df, x, y, color=color, orientation=orient, title=title)
    
    # update layout
    fig.update_layout(yaxis_type='category', barmode='group', shapes= shapes, showlegend=True)
    fig.update_xaxes(showline=True, linewidth=1, linecolor='gray', mirror=True)
    
    if x == 'mean_roi':
        fig.add_annotation(x=.1, y=4.5,
                       text='Minimum expected ROI (10%)',
                       showarrow=True,
                       arrowhead=1)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='gray', mirror=True)
    elif x == 'lower_roi':
        fig.add_annotation(x=-.4, y=4.5,
                       text='Negative ROI threshold (-40%)',
                       showarrow=True,
                       arrowhead=1)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='gray', mirror=True, side='right')
    fig.show()


def model_rmse(dfs, mdl_name):
    '''
    Docstring:     Compare model RMSE
    Signature:     model_rmse_comp(
                   dfs=none,
                   mdl_name=none
                   )
    Parameters:    dfs: list, list of dataframe
                   mdl_name: list, model name
    Return:        dataframe                   
    '''
    filter_rmse = pd.DataFrame()

    for df , model in list(zip(dfs,  ['A', 'B', 'C'])):
        tmp = df[df['dynamic'] == True][['zc','rmse']]
        tmp['model'] = model
        filter_rmse = pd.concat([filter_rmse, tmp])
        
    return filter_rmse

# def plot_roi_by_model(df, x, y, color, title, orient='h'):
#     '''
#     Docstring:    Plot ROI by model
#     Signature:    plot_roi_by_model(
#                   df=none,
#                   x=none,
#                   y=none,
#                   color=none,
#                   title=none,
#                   orient='h'
#                   )
#     Parameters:   df: dataframe
#                   x: string, panda column name 
#                   y: string, panda column name 
#                   color: string, panda column name
#                   title: string
#                   orient: string, 'h' or 'v'
#     Return:       Plotly grouped bar plot       
#     '''
#     # set vertical line
#     if x == 'mean_roi':
#         shapes = [dict(type='line', xref= 'x', x0= 0.1, x1= 0.1, yref= 'y', y0= -.5, y1= 4.5,
#                       line= {'color':'green', 'width': 1, 'dash':'dashdot'})]
#     elif x =='lower_roi':
#         shapes = [dict(type='line', xref= 'x', x0= -0.4, x1= -0.4, yref= 'y', y0= -.5, y1= 4.5,
#                       line= {'color':'green', 'width': 1, 'dash':'dashdot'})]        
    
#     # plot bar chart
#     fig = px.bar(df, x, y, color=color, orientation=orient, title=title)
    
#     # update layout
#     fig.update_layout(yaxis_type='category', barmode='group', shapes= shapes)
#     fig.update_xaxes(showline=True, linewidth=1, linecolor='gray', mirror=True)
#     fig.update_yaxes(showline=True, linewidth=1, linecolor='gray', mirror=True)
#     if x == 'mean_roi':
#         fig.add_annotation(x=.1, y=4.5,
#                        text='Minimum expected ROI (10%)',
#                        showarrow=True,
#                        arrowhead=1)
#     elif x == 'lower_roi':
#         fig.add_annotation(x=-.4, y=4.5,
#                        text='Negative ROI threshold (-40%)',
#                        showarrow=True,
#                        arrowhead=1)    
#     fig.show()


# def model_rmse(dfs, mdl_name):
#     '''
#     Docstring:     Compare model RMSE
#     Signature:     model_rmse_comp(
#                    dfs=none,
#                    mdl_name=none
#                    )
#     Parameters:    dfs: list, list of dataframe
#                    mdl_name: list, model name
#     Return:        dataframe                   
#     '''
#     filter_rmse = pd.DataFrame()

#     for df , model in list(zip(dfs,  ['A', 'B', 'C'])):
#         tmp = df[df['dynamic'] == True][['zc','rmse']]
#         tmp['model'] = model
#         filter_rmse = pd.concat([filter_rmse, tmp])
        
#     return filter_rmse



































    