import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from flask import Flask
import os

import pandas as pd
import numpy as np
import re
import pickle
import plotly.graph_objs as go
import base64

server = Flask(__name__)
server.secret_key = os.environ.get('secret_key', 'secret')
app = dash.Dash(name = __name__, server = server)
app.config.supress_callback_exceptions = True

my_model = pickle.load(open('ub_house_price_model.pkl','rb'))
image_filename = 'ub_pic.jpeg' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())
myimg = 'data:image/jpeg;base64,{}'.format(encoded_image.decode())

dash_pred_input = pd.read_csv('dash_pred_input.csv', index_col=0)
dash_dummies_cat_features = pd.get_dummies(dash_pred_input[['location','district','dis_horoo']])
dash_numerical_features = dash_pred_input.drop(['location','district','dis_horoo'], axis=1)
dash_numerical_features['rooms']=1

all_locs = ['10-р хороолол', '100 айл', '11-р хороолол', '120 мянгат', '13-р хороолол', '14-р хороолол', '15-р хороолол', '16-р хороолол', '19-р хороолол', '220 мянгат', '5-р хороолол', '6-р бичил', 'Japan town', 'King Tower', 'Marshall town', 'Olympic village', 'River Garden', 'UB Town', 'Zaisan luxury village', 'Амгалан', 'Бага тойрог', 'Баянбүрд', 'Баянмонгол хороолол', 'Зайсан', 'Зурагт', 'Зүүн 4 зам', 'Их Монгол хороолол', 'Нисэх', 'Нэхмэлийн шар', 'Оддын хороолол', 'Сансар', 'Тээврийн товчоо', 'Төмөр зам', 'Улиастай', 'Ханбүргэдэй', 'Хос-Өргөө', 'Чингэлтэй', 'Шар хад', 'Эрдэнэ толгой', 'Яармаг']

dash_locs_options = [{"label":i, "value":i} for i in all_locs]

app.css.append_css({'external_url':'https://use.fontawesome.com/releases/v5.7.2/css/all.css'})
app.layout = html.Div([
    html.Img(src=myimg, height='200px',width='100%'),
    html.Div([
    html.Div([
        dcc.Dropdown(
        id='av_locs',
        options=dash_locs_options,
        placeholder='Choose ur location'
        ),
        html.Div([
            html.Label('Number of rooms', style={'display':'block', 'float':'left', 'height':'30px','line-height':'30px', 'padding':'1% 0', 'font-size':'16px', 'font-family':'serif'}),
            dcc.Input(
                id='num_rooms',
                type='number', 
                inputmode='numeric', 
                max=10, 
                min=0,
                placeholder='Number of rooms',
                style={'display':'block', 'float':'right', 'height':'30px','line-height':'30px', 'width':'25%', 'border-radius':'12px','text-align':'center'}
            )
        ], style={'height':'40px', 'padding':'0 5px', 'background':'#91f5a4'}),
        html.Div([
            html.Label('Number of square meter', style={'display':'block', 'float':'left', 'height':'30px','line-height':'30px', 'padding':'1% 0', 'font-size':'16px', 'font-family':'serif'}),
            dcc.Input(
                id='sqmtr',
                type='number', 
                inputmode='numeric', 
                max=600, 
                min=0,
                placeholder='Choose your place size in square meter',
                style={'display':'block', 'float':'right', 'height':'30px','line-height':'30px', 'width':'25%', 'border-radius':'12px','text-align':'center'}
            )
        ], style={'height':'40px', 'padding':'0 5px', 'background':'#91f5a4'}),
        html.Button(id='predict_button', type='submit', children='Predict', style={'width':'100%', 
                                                                                   'height':'30px', 
                                                                                   'display':'block',
                                                                                   'border-radius':'0 0 9px 9px',
                                                                                   'font-size':'16px',
                                                                                  'font-family':'serif'}),
        html.Div([
            html.H3(children='Predictions are made on all locations using the user input'),
            html.Ul([
                html.Li(children='Large screen devices are recommended to see the outputs',style={"padding":'6px'}),
                html.Li(children='You are able to compare prices of different places',style={"padding":'6px'}),
                html.Li(children='Additionally appartment prices may depend on the surrounding places such as: number of kindergartens, schools, crimes per 100 thousand people, and distance from city center',style={"padding":'6px'}),
                html.Li(children="A few core determinents such as age of the appartment and side it faces are not used in the model thus can't be displayed",style={"padding":'6px'}),
                html.Li(children="Main dataset(3863 rows, 2018-12-25:2019-02-25) has been scrapped from 'Unegui.mn' for educational purpose",style={"padding":'6px'}),
                html.Li(children="Resources and work here: https://github.com/robertritz/mongol.ai/tree/W_branch")
            ]),
        ], style={"border":"1px solid red", 'border-radius':'10px','padding':'10px'}),
        
        
    ], style={'float':'left', 'width':'450px'}),
    html.Div([],id="dash_prediction_output", style={'display':'grid', 
                                              'grid-template-columns':'1fr 1fr', 
                                              'grid-template-areas':'"first_con second_con""third_con third_con"'}),
    ], style={'display':'grid','grid-template-columns': '1fr 2fr'}),

    html.Div(id='output')
])

 

@app.callback(Output('dash_prediction_output', 'children'),
             [Input('predict_button', 'n_clicks')],
             [State('av_locs', 'value'),
              State('num_rooms', 'value'),
              State('sqmtr', 'value')])

def get_inputs(submit, av_locs, num_rooms, sqmtr):
    if submit is not None:
        # use input for every location
        dash_numerical_features['rooms']= num_rooms
        dash_numerical_features['sqmtr']= sqmtr
        
        # finding location of user input
        user_input_loc_in_preds = dash_pred_input[dash_pred_input['location']==av_locs].index[0]
        #additional info for user input
        add_n_schools = dash_pred_input[dash_pred_input['location']==av_locs]['n_schools'].get_values()[0]
        add_n_kindergts = dash_pred_input[dash_pred_input['location']==av_locs]['n_kindergartens'].get_values()[0]
        add_distance_from_cc = dash_pred_input[dash_pred_input['location']==av_locs]['distance_from_cc'].get_values()[0]
        add_crime_score = dash_pred_input[dash_pred_input['location']==av_locs]['crime_score'].get_values()[0]
        avg_crimes_per_h1000s = dash_pred_input['crime_score'].mean()
        
        # now we need to concat numerical features with categorical(one hot encoded) features 
        deployment_input = pd.concat([dash_numerical_features, dash_dummies_cat_features], axis=1)

        upper_int = my_model.predict(deployment_input, quantile=75)
        upper_int_main_pred = upper_int[user_input_loc_in_preds]
        
        mid_int = my_model.predict(deployment_input, quantile=50)
        mid_int_main_pred = mid_int[user_input_loc_in_preds]
        
        lower_int = my_model.predict(deployment_input, quantile=25)
        lower_int_main_pred = lower_int[user_input_loc_in_preds]
        
        ### Output htmls
        prediction_interval_html = html.Div([
                    html.Div([
                        html.P('Max price'),
                        html.Div(children='₮ {}'.format(comma_me(str(round(upper_int_main_pred)))),
                            style={
                              'background':'#ff4e4e',
                              'padding': '5px 10px',
                              'border-radius': '10px',
                              'color': 'whitesmoke',
                              'margin': '10px auto'
                            }
                        )
                    ], style={'display':'grid','grid-template-columns': '1fr 1fr'}),
                    html.Div([
                        html.P('Mid price'),
                        html.Div(children='₮ {}'.format(comma_me(str(round(mid_int_main_pred)))),
                            style={
                              'background':'#9ee3e9',
                              'padding': '5px 10px',
                              'border-radius': '10px',
                              'color': 'whitesmoke',
                              'margin': '10px auto'
                            }
                        )
                    ], style={'display':'grid','grid-template-columns': '1fr 1fr'}),
                    html.Div([
                        html.P('Min price'),
                        html.Div(children='₮ {}'.format(comma_me(str(round(lower_int_main_pred)))),
                            style={
                              'background':'#3dff6f',
                              'padding': '5px 10px',
                              'border-radius': '10px',
                              'color': 'whitesmoke',
                              'margin': '10px auto'
                              }
                        )
                    ], style={'display':'grid','grid-template-columns': '1fr 1fr'}),
        ], style={"float":'left', 'padding':'0 10%', 'grid-area':'first_con', 'display':'block', 'border':'1px solid red', 'border-radius':'10px'})
        
        additional_info_html = html.Div([
            html.Div([
                html.Div(children=add_n_kindergts,style={'text-align':'center', 'font-size':'25px', 'margin':'auto 0'}),
                html.Div([
                    html.I(className='fas fa-baby', title='Kindergartens', style={'font-size':'30px'}),
                    html.Plaintext('kindergartens')
                ], style={'text-align':'right'})
            ], style={
                'background':'rgb(145, 245, 164)',
                'font-size':'18px',
                'padding':'10px',
                'border-radius':'14px',
                'display':'grid',
                'grid-template-columns':'1fr 1fr',
                'height':'65px'
                }),
            html.Div([
                html.Div(children=add_n_schools,style={'text-align':'center', 'font-size':'25px', 'margin':'auto 0'}),
                html.Div([
                    html.I(className='fas fa-school', title='schools', style={'font-size':'30px'}),
                    html.Plaintext('schools')
                ], style={'text-align':'right'})
            ], style={
                'background':'rgb(218, 251, 0)',
                'font-size':'18px',
                'padding':'10px',
                'border-radius':'14px',
                'display':'grid',
                'grid-template-columns':'1fr 1fr',
                'height':'65px'
                }),
            html.Div([
                html.Div(children='{} vs avg {}'.format(round(add_crime_score,0), round(avg_crimes_per_h1000s)),style={}),
                html.I(className='fas fa-fist-raised', title='crime', style={'font-size':'30px'}),
                html.Div([
                    html.Plaintext('crimes per 100,000 ')
                ], style={'text-align':'right'})
            ], style={
                'background':'rgb(245, 55, 95)',
                'font-size':'18px',
                'padding':'10px',
                'border-radius':'14px',
                'display':'grid',
                'grid-template-columns':'1fr 1fr',
                'height':'70%'
                }),
            html.Div([
                html.Div(children='{} km'.format(add_distance_from_cc), style={'text-align':'center', 'font-size':'25px'}),
                html.I(className='fas fa-map-marked-alt', title='dfcc', style={'font-size':'30px'}),
                html.Div([
                    html.Plaintext('from city center')
                ], style={'text-align':'right'})
            ], style={
                'background':'rgb(145, 245, 234)',
                'font-size':'18px',
                'padding':'10px',
                'border-radius':'14px',
                'display':'grid',
                'grid-template-columns':'1fr 1fr',
                'height':'70%'
                })
        ], style={'grid-area':'second_con', 'display':'grid','grid-template-columns':'1fr 1fr'})
        my_colors = []
        for i in range(len(dash_pred_input)):
            if dash_pred_input['location'][i] == dash_pred_input['location'][user_input_loc_in_preds]:
                my_colors.append('rgb(245, 55, 95)')
            else:
                my_colors.append('rgb(7, 119, 106)')
        
        all_locs_graph = html.Div([
            dcc.Graph(
               figure = {
                    'data':[go.Bar(
                        y=mid_int, 
                        x=dash_pred_input['location'],
                        marker={'color':my_colors}
#                         orientation='h'
                    )],
                   'layout': go.Layout(
                       autosize=True,
                       title='Prediction on all locations',
                       yaxis={'ticklen':10}
                   )
                } 
            )
        ], style={'grid-area':'third_con'})
        
        
        return prediction_interval_html, additional_info_html, all_locs_graph

def comma_me(amount):
    orig = amount
    new = re.sub("^(-?\d+)(\d{3})", '\g<1>,\g<2>', amount)
    if orig == new:
        return new
    else:
        return comma_me(new)

