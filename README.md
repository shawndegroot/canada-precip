# canada-precip
A Dash
app to predict precipitation in Canada. Hosted online at https://canada-precip.herokuapp.com/.

Based off idea of [Rain in Australia](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package) on Kaggle, this app makes a prediction in real-time whether or not there will be precipitation tomorrow, for any of weather 326 stations, by training a binary classifiation model on the target RainTomorrow created from Canadian Historical Station Data downloaded from [ClimateData.ca](https://climatedata.ca/). 

The following 5 variables are scraped off the Canadian Weather Office Pages for the station selected (past 24 hr conditions page): Mean Temperature, Maximum Temperature, Minimum Temperature, Maximum Relative Humidity and Minimum Relative Humidity.

Precipitation is defined as >= 1.1 mm in a day. 

Decision Tree Classifier trained on 10 years of daily station data (2010-2019) for 83 stations (670000 samples), filtered to include days with humidity records. Accuracy of 85%.

If you have any suggestions or ideas for improvements, please don't hesitate to contact me. 
