import pandas as pd
import pickle as pickle
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import re
import dash_bootstrap_components as dbc
from textwrap import dedent

mapbox_style = "mapbox://styles/plotlymapbox/cjyivwt3i014a1dpejm5r7dwr"
mapbox_access_token = 'pk.eyJ1IjoiZGVncm9vdHMiLCJhIjoiY2p3MDF5aGZ0MDZrcjN5bHA0aXU4M3R1aCJ9.RXDDdwYcaj-wJ7frURO4ZQ'
PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

app= dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

#####5 inputs######
# trained model
fld_p='DECISION_TREE_FINAL.sav'
# scaling for prediction
fld_SCALE = 'SCALING_FINAL.sav'
# list of 326 cities for dropdown
fld_c='tables.txt'
# for 3 letter code-->past 24 hrs page
fld_s='stations_new.csv'
# for initial map with stations
fld_sn='stations_clean.csv'


with open(fld_c, 'r') as f:
    cities= [line.strip() for line in f]
    cities.sort()
    
loaded_clf = pickle.load(open(fld_p, 'rb'))
scaler=pickle.load(open(fld_SCALE, 'rb'))
stations=pd.read_csv(fld_s)
stations_sn=pd.read_csv(fld_sn)

##################FUNCTION "LEARN MORE" BOX############
def markdown_popup():    
    return html.Div(
        id="markdown",
        className="modal",
        style={"display": "none"},
        children=(
            html.Div(
                className="markdown-container",
                children=[
                    html.Div(
                        className="close-container",
                        children=html.Button(
                            "Close",
                            id="markdown_close",
                            n_clicks=0,
                            className="closeButton",
                        ),
                    ),
                    html.Div(
                        className="markdown-text",
                        children=[
                            dcc.Markdown(
                                children=dedent(
                                    """
                                This app scrapes data off the [Canadian Weather Office Pages](https://weather.gc.ca/canada_e.html), and makes a 
                                prediction whether or not there will be precipitation tomorrow for 
                                the weather station selected, based upon the following five variables for the current day: 
                                Mean Temperature, Maximum Temperature, 
                                Minimum Temperature, Maximum Relative Humidity and Minimum Relative Humidity. Precipitation defined as >= 1.1 mm. Decision Tree Classifier trained
                                on 10 years of daily station data (2010-2019) for 83 stations, downloaded
                                from [ClimateData.ca] (https://climatedata.ca/). Accuracy of 85%. Idea from [Rain in Australia] (https://www.kaggle.com/jsphyg/weather-dataset-rattle-package) on Kaggle.
                                Code can be found [here] (https://github.com/shawndegroot/canada-precip).
                                
                                Created by Shawn DeGroot.
                                """
                                )   
                            )
                        ],
                style = {'marginLeft': 40, 'marginRight': 40, 'marginTop': 10, 'marginBottom': 10, 
               'backgroundColor':'#F7FBFE',
               'border': 'thin lightgrey solid', 'padding': '6px 0px 0px 8px'})
                ],
            )   
        ),
    )

    
##############NAVBARS###############################
NAVBAR = dbc.Navbar(
children=[
    html.A(
        dbc.Row(
            [
                dbc.Col(html.Img(src=PLOTLY_LOGO, height="40px")),
                dbc.Col(
                    dbc.NavbarBrand("WILL THERE BE PRECIPITATION TOMORROW?", className="ml-2")
                ),
            ],
            align="center",
            no_gutters=True,
        ),
        href="https://plot.ly",
    )
],
color="dark",
dark=True,
sticky="top",
)

NAVBAR_BOTTOM = dbc.Navbar(
children=[
    html.A(
        dbc.Row(
            [
                dbc.Col(),
                dbc.Col(
                ),
            ],
            align="center",
            no_gutters=True,
        ),
        href="https://plot.ly",
    )
],
color="dark",
dark=True,
sticky="bottom",
)
################MAIN APP###################
options=[{'label': i, 'value': i} for i in cities]

app.layout=html.Div(        
              children=[
                    html.Div([
                        # child 1- Navbar
                            NAVBAR,
                            html.Br(),
                         #child 2- Main Text  
                            html.Div([
                                html.H5(
                                    """This app uses a machine learning algorithm to predict whether or not 
                                    there will be precipitation tomorrow, for any of 325 active weather stations in Canada. 
                                    """
                                        )],
                                style = {'marginLeft': 40},
                                ),                                             
                            # child 3- learn more button
                            html.Div([
                                html.Button(
                                    "Learn More", id="learn-more-button", n_clicks=0
                                            )],
                                style = {'marginLeft': 40},
                                ),                
                            html.Br(),    
                            html.Br(),    
                            # child 4- enter locatoin                
                            html.Div([
                                html.H5(
                               "Enter a Location:"
                                )], 
                                style = {'marginLeft': 40},
                                ),                                           
                            #child 5- Dropdown
                            html.Div([
                                dcc.Dropdown(
                                    id='my-dropdown',
                                    options=options,
                                     style={"border": "0px solid black",
                                            },
                                      placeholder="Select a Location in Canada",       
                                              )],
                                style = {'marginLeft': 40},
                                ),                         
                            # child 6- map- putting all stations on map
                             dcc.Graph(
                                      style={'height': '550px', 'width' : '1440px'},
                                            id="mapbox_map",
                                            figure=dict(    
                                                data = [    
                                                    dict(
                                                        type= 'scattermapbox',
                                                        lat= list(stations_sn['Latitude']),
                                                        lon= list(stations_sn['Longitude']),
                                                        hoverinfo= 'text',
                                                        hovertext = [["{}, {} <br>Latitude = {} <br>Longitude = {}".format(i,j,k,l)]
                                                                        for i,j,k,l in zip(stations_sn['Name'], stations_sn['Province/Territory_x'],stations_sn['Latitude'],stations_sn['Longitude']
                                                                        )],
                                                        mode= 'markers',
                                                        name= list(stations_sn['Name']),
                                                        marker= dict(
                                                            size= 6,
                                                            opacity= 0.4,
                                                            color= 'red'),
                                                    )
                                                    ],
                                                layout=dict(
                                                    mapbox=dict(
                                                            layers=[],
                                                            accesstoken=mapbox_access_token,
                                                            style=mapbox_style,
                                                            center=dict(
                                                                lat=71.12490, lon=-95.61446
                                                            ),
                                                            zoom=1.5,
                                                            margin=dict(r=0, l=90, t=10, b=0),
                                                            ),
                                                autosize=True,
                                                margin=dict(r=90, l=90, t=10, b=0),
                                                ),
                                            ),
                                        ),
                            html.Br(),
                            ]
                        ),            
html.Div(id='output-container'),
html.Br(),
html.Br(),
NAVBAR_BOTTOM,
 # calling markdown popup function:   
markdown_popup(),
])   
####################CALLBACKS########################################   
#first callback: dropdown-->text (and scraping wxoffice pages)
@app.callback(
    dash.dependencies.Output(component_id='output-container', component_property='children'),
    [dash.dependencies.Input(component_id='my-dropdown',component_property='value')])
    
def update_output(value):
    try:
        city=value    
        city_station=stations[[any([a, b]) for a, b in zip(stations['Name'].str.contains(city,flags=re.IGNORECASE,na=False),
        stations['Name2'].str.contains(city,flags=re.IGNORECASE,na=False))]]                   
        city_station=city_station.sort_values("ID_NEW",ascending=[False])  
        city_station=city_station[:1]   
        city_station_ID=city_station['ID_NEW']
        city_station_ID=city_station_ID.to_string()
        city_station_ID=city_station_ID[4:]
        city_station_ID=city_station_ID.strip().lower()
    except Exception as e:
        print(str(e))
                    
    try:    
        # go to past 24 hr conditions page for city
        dfs=pd.read_html("https://weather.gc.ca/past_conditions/index_e.html?station="+city_station_ID)
        dfs2 = pd.DataFrame(dfs)
        dfs2=dfs[0]
        dfs2_new = dfs2.fillna('remove')
        dfs2_new = dfs2_new.iloc[1:]
        row = dfs2_new.loc[dfs2_new['Conditions'].str.contains("2020")].index.tolist()    
        dfs3 = dfs2_new.iloc[:row[0]]
        dfs3.drop(dfs3.tail(1).index,inplace=True)
        dfs3 = dfs3.replace("remove", np.nan)       
        #tmax
        tmaxC=dfs3['Temperature (°C)']
        tmaxC = pd.DataFrame(tmaxC)
        tmaxC['new_col'] = tmaxC['Temperature (°C)'].astype(str).str[:2]
        tmaxC = pd.to_numeric(tmaxC['new_col'])
        MAX = tmaxC.max()
        #tmin
        MIN = tmaxC.min()
        #tmean
        MEAN = int(round(tmaxC.mean()))
        #max hum
        hum_max = dfs3['Relativehumidity(%)'].max()
        # min hum
        hum_min = dfs3['Relativehumidity(%)'].min()    
        df_city=pd.DataFrame()    
        df_city= { 'MEAN_TEMPERATURE': [MEAN],
                   'MAX_TEMPERATURE': [MAX],
                   'MIN_TEMPERATURE': [MIN],
                   'MAX_REL_HUMIDITY': [hum_max],
                   'MIN_REL_HUMIDITY': [hum_min],
                   }          
        X_new=pd.DataFrame(df_city)
        array=np.asarray(X_new)    
        array_scaled=scaler.transform(array)  
        # make prediction and apply scaling
        pred=loaded_clf.predict(array_scaled)
        
        if pred==1:
            return html.Div([
        html.H4(className="h3-title", children="There WILL be precipitation in "+city+ " tomorrow!")],
                style = {'marginLeft': 90},
                ),                                                   
        else:
            return html.Div([
        html.H4(className="h3-title", children="There will NOT be precipitation in "+city+ " tomorrow!")],
                style = {'marginLeft': 90},
                ),               
    except Exception as e:
        print(str(e))       

    #####################################
# second callback: dropdown-->symbol on map
@app.callback(
    dash.dependencies.Output(component_id='mapbox_map', component_property='figure'),
    [dash.dependencies.Input(component_id='my-dropdown', component_property='value')
    ])
 
def update_map(VALUE):
    city = VALUE
    city = str(city)
    # getting LAT/LON for city 
    city_station_new=stations[[any([c, d]) for c, d in zip(stations['Name'].str.contains(city,flags=re.IGNORECASE,na=False),
    stations['Name2'].str.contains(city,flags=re.IGNORECASE,na=False))]]   
    city_station_new=city_station_new.sort_values("ID_NEW",ascending=[False])  
    city_station_new=city_station_new[:1]    
    latitude = round(city_station_new['Latitude'].values[0], 2)
    longitude = round(city_station_new['Longitude'].values[0], 2)          
     # building DF
    col_names = ['city', 'latitude','longitude']
    list_new=[city,latitude,longitude]
    df = pd.DataFrame(columns=col_names)
    df.loc[len(df)] = list_new    
    hovertext = [["{} <br>Latitude = {} <br>Longitude = {}".format(city,latitude,longitude)]]                                                                                                          
    hovertext = ", ".join(map(str, hovertext))
    hovertext = hovertext.strip("[]").strip("()")
    hovertext = hovertext.strip(" ' ")
    hovertext = hovertext.split(',')

    data = [
            dict(
                type = "scattermapbox",
                lat = [float(latitude)],
                lon = [float(longitude)],
                hoverinfo = 'text',
                hovertext = hovertext,  
                mode = 'markers',
                name = [city],
                marker = dict(size=16,opacity=0.8, color="red"),
                )
            ]       
    layout = dict(
                mapbox=dict(
                layers=[],
                accesstoken=mapbox_access_token,
                style= mapbox_style,
                center=dict(lat=latitude, lon=longitude),
                zoom=4.0,
                margin=dict(r=0, l=90, t=10, b=0),
                ),
            hovermode="closest",
            margin=dict(r=90, l=90, t=10, b=0),
            dragmode="lasso",
                )    
    figure = dict(data=data, layout=layout)    
    return figure

#####################################
# third callback- Learn more popup
@app.callback(
    dash.dependencies.Output("markdown", "style"),
    [dash.dependencies.Input("learn-more-button", "n_clicks"),
     dash.dependencies.Input("markdown_close", "n_clicks")],
     )

def update_click_output(button_click, close_click):
    ctx = dash.callback_context
    prop_id = ""
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if prop_id == "learn-more-button":
        return {"display": "block"}
    else:
        return {"display": "none"}

############################
if __name__ == '__main__':
    app.run_server(debug=True)


