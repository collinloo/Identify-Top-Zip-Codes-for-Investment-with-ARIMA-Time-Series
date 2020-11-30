## Table of Contents
* [Repo Structure](#Repo-Structure)
* [How to Run the Codes](#How-to-Run-the-Codes)
* [Project Roadmap](#Project-Roadmap)
* [Obtain Data](#Obtain-Data)
* [Preprocess Data](#Preprocess-Data)
* [Intermediate Model Fitting](#Intermediate-Model-Fitting)
* [Explore Data](#Explore-Data)
* [Stationarity Check](#Stationarity-Check)
* [Auto and Partial Correlation Plot](#Auto-and-Partial-Correlation-Plot)
* [Decompose Trend](#Decompose-Trend)
* [Train and Validation Data Set](#Train-and-Validation-Data-Set)
* [Model A](#Model-A)
* [Model B](#Model-B)
* [Model C](#Model-C)
* [Model Interpretation](#Model-Interpretation)
* [Models 5 Years Forecast](#Models-5-Years-Forecast)
* [Conclusion](#Conclusion)
* [Future Work](#Future-Work)


## Repo Structure
* Root of the repo contains the main jupyter notebook and a python file of my fuctions
* A PDF file delivers the presentation of this project
* img folder holds all the images for the repo README.md and presentation.
* csv_files folder houses the data source and mapping of States to Region file.

## How to Run the  Codes
<ul>
    <li>import pandas as pd</li>
    <li>import numpy as np</li>
    <li>import itertools</li>
    <li>import statsmodels.api as sm</li>
    <li>from statsmodels.tsa.seasonal import seasonal_decompose</li>
    <li>from statsmodels.tsa.stattools import adfuller</li>
    <li>from statsmodels.graphics.tsaplots import plot_acf, plot_pacf</li>
    <li>import matplotlib.pyplot as plt</li>
    <li>from matplotlib.pylab import rcParams</li>
    <li>import plotly.express as px</li>
    <li>import plotly.graph_objects as go</li>
    <li>import plotly.figure_factory as ff</li>
    <li>import seaborn as sns</li>
    <li>from my_func import *</li>
</ul>

## Project Roadmap
#### Objective
The purpose of this project is to develop a time series model to evaluate if a zip code meet the investment strategy criteria of a real estate investment company.

#### Business Case
The comppay's goals is to invest in a property, rent it and sell it after five years.  The properties in the selected top five zip code should meet the two following requirements after the sale of the property.
1. A minimum of 10% return on investment (ROI).
2. A lost of ROI no greater than 40%.  Lost is calculated using the lowest forecast for the period which comes from the lower confidence interval of the forecast.  Since real estate investments are not as volatile as stock or commodity investments, it is unlikely for home value to depreciate sharply within five years.  Unless there are unforeseen natural disasters or major financial crisis like the 2008 real estate bubble.

Prior to modeling, we filter the data based on the following investment strategies:
1. ROI on the sale of the property should be greater than 10%. 

ROI=(*sale price after 5 years-((purchase price(0.035))+repair+purchase price)/(purchase price(0.035))+20000+purchase price*   
where 0.035 = closing cost as a percent of the purchase price  
repair = 20,000   


2. Property must have a price to rent ratio between 12 and 25 because this is a good indicator to tell whether people are incline to rent or own a home.

PTR=*median home price / median annual rental*   
where Median Annual Rental = Zillow Rental Index x 12  
    
   
3. Cash on Cash ratio should be greater than 8%.  

COC=*((Rental * 12) - (Mort. Payment + Insurance + Vacancy Allowance)) / (Down Payment + Closing Cost)*  
where Rental = Zillow Rental Index  
Mort. payment = Loan x ((Interest rate/12) x (1+Interest rate/12))sq(Mort. Term) / (1+Interest rate/12))sq(Mort.Term)-1)  
Loan = Property purchase price x 0.80   
Interest rate = 2.5%  
Mort.Term = 360 (30yrs mortgage)  
Insurance = (Property purchase price / 100,000) x 40  
Vacancy Allowance = Rental Index x 0.10  
Down Payment = Property purchase price x 0.20  
Closing Cost = Property purchase price x 0.035  
    
4. Standard deviation of Rental Index should be in the range of 30% to 60 % quantile.  This is to reduce rental market volatility.  
    
#### Project Approach
We will follows the OSEMN framework during the project execution.  Firstly, we will create a base model, using the EDA process to derive the order parameter for the ARIMA model.  Secondly, utilize the grid search process to find the optimum order parameter for the ARIMA model.  Lastly, We will choose the best model based on the two requirements stated in the business case.

## Obtain Data
1. Import zillow_data.csv as pandas dataframe.
2. Import invest_metrics.csv as pandas dataframe.  
    <ul>
        <li>ROI, price to rent ratio, cash on cash and rental standard deviation are calculated using the rental index and the median home value from the main dataset. Detail calculation can be found in the Excel spreadsheet in the data directory (Metro_ZORI.xlsx).</li>
        <li>The Zillow Observed Rent Index can be downloaded from https://www.zillow.com/research/data/ </li>
    </ul>
3. Merge the two dataframes.

## Preprocess Data
1. Subset the dataframe based on investment criteria.
    <ul>
        <li>Return on investment should be greater than 10%.</li>
        <li>Properties must have a price to rent ration between 12 and 25 as this is a good indicator to use to identify the rental market location.</li>
        <li>Cash on cash return rate should be greater than 8%.</li>
        <li>Rental index standard deviation should be within the 30% to 60% quantile to minimize risk and volatility.</li>
    </ul>
2. Select 3 zip codes from each state where ROI are the highest.

## Intermediate Model Fitting
1. Run each top 3 zip codes from each state through the ARIMA grib search.
2. Select the top 5 zip codes from the results, where ROI is more than 10% and lower ROI is less than -40%.
3. Top 5 zip codes
    <ul>
        <li>28227, Mint Hill, NC</li>
        <li>29472, Ridgeville, SC</li>
        <li>28273, Charlotte, NC</li>
        <li>75233, Dallas, TX</li>
        <li>02746, New Bedford, MA</li>
    </ul>

## Explore Data
General trend    
![](/img/ppt_home_value.png?raw=true)

Data distribution
![](/img/ppt_distplot.png?raw=true)

Sample: Box & Whisker Zip Code 28227
![](/img/w_b.png?raw=true)

## Stationarity Check
ARMA model requires time series to be stationary. Since our time series exhibits a linear upward trend, we will need to remove such trend using one of the detrending methods, such as differencing, and validate our results with Augmented Dickey Fuller (ADF) test.

WE had to perform 2nd order differencing to remove stationary.

## Auto and Partial Correlation Plot
Plot ACF and PACF to determine the p and q for AR and MA models.

Autocorrelation
![](/img/acf.png?raw=true)

Partial Autocorrelation
![](/img/pacf.png?raw=true)

## Decompose Trend
Check seasonality in the series.

Sample: Seasonality for 3 zip codes
![](/img/ppt_seasonal.png?raw=true)

## Train and Validation Data Sets
Create train and validation data sets, 75/25 split.
<ul>
    <li> 2008-12-01 < Train < 2016-01-01</li>
    <li> validation > 2015-12-01</li>
</ul>

## Model A
1. Create ARIMA model A based on observations from the ACF and the PACF plot. 
    <ul>
        <li>28227	(1, 2, 1)</li>
        <li>29472	(1, 2, 2)</li>
        <li>28273	(1, 2, 2)</li>
        <li>75233	(2, 2, 0)</li>
        <li>02746	(1, 2, 2)</li>
    </ul>
2. Fit the model and plot the diagnostics.

Sample: Training Summary and Diagnotic Plot
![](/img/mdlA_28227.png?raw=true)

3. Fit the model on the validation data set.

Sample: Validation Summary and Diagnotic Plot
![](/img/mdlA_val_28227.png?raw=true)

4. Plot the RMSE results from the dynamic vs non-dynamic forecast for training and validation.

Sample: Dynamic vs Non-Dynamic Forecast RMSE (Model A Validation)
![](/img/mdlA_val_rmse.png?raw=true)

## Model B
1.  For model B, we performed grid search and selected ARIMMA model with the lowest AIC score and fit the data.
2.  Carry out same analysis as in Model A.

## Model C
1.  For model C, we introduced seasonality into the model.  Ran the grid search and fit the data.
2.  Carry out same analysis as in Model A

## Model Interpretation
1. Combined and plot the RMSEs from the 3 models.

RMSE Training
![](/img/rmse_t.png?raw=true)

RMSE Validation
![](/img/rmse_v.png?raw=true)

2. Observations:
    <ul>
            <li>Model C seems  to perform better in all the zip codes with the lowest RMSE in the dynamic forecast of the training data set, except in the case of zip code 28227 where it is off by 1,000.</li>
            <li>Model A and B are almost identical in performance, since the grid search returns the same parameters as the manual one.</li>
    </ul>

## Models 5 Years Forecast
1.  We chose 5 years period because that was the intention of the investors.  Buy the properties, rent it out for 5 years than sell them.

2. Plot the forecast and the confidence interval values for the 3 models.

Sample: 5 years forecast from model B
![](/img/ppt_fc_02746.png?raw=true)

3. Check if the zip codes if they meet the two final investment requirements.

Mean ROI
![](/img/ppt_roi.png?raw=true)

Lower ROI
![](/img/ppt_lower_roi.png?raw=true)

## Conclusion
After we filtered the data using the 4 investment strategies, namely ROI, Price-to-Rent, Cash-on-Cash and Rental Index standard deviation, we were left with 179 zip codes.  From this subset of data, we furthered narrowing down our range by selecting top 3 zip codes with the highest ROI from each state.  The final data set contained 23 zip codes.  

In the intermediate stage, we ran a baseline ARIMA model on the top 20 zip codes from the final data set.  From the results, we recommended the following top 5 zip codes:
1. 28227, Mint Hill, NC
2. 29472, Ridgeville, SC
3. 28273, Charlotte, NC
4. 75233, Dallas, TX
5. 02746, New Bedford, MA

In the subsequent stage, we developed and analyzed 3 models.  Finally, we performed a 5 five forecast and computed the various ROI.

Model B is the only model that meet the investment requirements, a ROI that is greater than 10% and a risk factor where the lower ROI is less than -40%.  Model A came close to Model B performance, except in forecasting zip code 28227, where the risk might be too high.

## Future Work
1.  We could have approach the project differently by focusing on a particular state and run all the zip codes in the selected state through the model.  The advantage is that all the data get evaluated.  The disadvantage is that it is computational expensive.
2. Develop a process to parameterize the investment strategies to generate a new data set.  In this case, we can evaluate the effect of loosing or tightening the investment requirements quickly.
3. Try Facebook Prophet forecasting method and compare the end results with our models.