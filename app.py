"""The module contains the function build_plot, which builds a 
Plotly Dash application with an interactive DT Plot. The file
can be run both from the bash command as well as be called by an
instance of the InteractiveDiscriminationThreshold class from 
the module interactive_discrimination_threshold.
"""

from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import flask
from jupyter_dash import JupyterDash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Union


def build_plot(data: pd.DataFrame, test_size=0.2,
               app_mode='external') -> Union[Dash, JupyterDash]:
    """Generates an interactive version of the discrimination threshold
    plot.

    Parameters
    ----------
        data: pd.DataFrame
            Data to be used an input for the plot
        test_size: float, optional
            Share of the test set (default is 0.2)
        app_mode: str, optional
                If 'inline', the app is being created as instance of
                JupyterDash and the plot is drawn inside the notebook; if
                'external', then the app is created as JupyterDash
                instance and the plot is drawn in a separate window; if
                'server', then the app is created as a Dash instance to be
                deployed as a server application
                (default is 'inline')
    Returns
    -------
        An instance of Dash or JupyterDash to be used for plotting.
    """

    n_obs = int(data.shape[0])

    if app_mode in ('inline', 'external'):
        app = JupyterDash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    else:
        server = flask.Flask(__name__)
        app = Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

    controls = [
        dbc.Form(
            [
                dbc.Label("Interval"),
                dcc.Dropdown(
                            id='interval_dropdown',
                            clearable=False,
                            value=0.95,
                            options=[
                                {'label': f'{ii}%', 'value': ii / 100}
                                for ii in [0, 5, 10, 25, 50, 75, 90, 95]
                            ],
                            style={'margin-bottom': '8px', 'width': '100px'}
                        )]),
        dbc.Form(
            [
                dbc.Label("Metrics"),
                dbc.Checklist(
                    options=[
                        {'label': 'Precision', 'value': 'precision'},
                        {'label': 'Recall', 'value': 'recall'},
                        {'label': 'F1-score', 'value': 'f1'},
                        {'label': 'Queue Rate', 'value': 'queue_rate'},
                    ],
                    value=['precision', 'recall', 'queue_rate'],
                    id="metrics",
                    inline=True,
                    style={'margin-bottom': '8px'},
                ),
            ]
        ),
        dbc.Form(
            [
                dbc.Label('weight(Recall) / weight(Precision)'),
                dbc.Input(
                    id='recall_over_precision',
                    type='number',
                    min=0,
                    step=0.1,
                    value=1,
                    style={'width': '100px', 'margin-bottom': '15px'},
                ),
            ]
        ),
        dbc.Form(
            [
            dbc.Checklist(
                id='constrained',
                options=[{'label': 'Enable Constrained Mode',
                          'value': 'constrained'}],
                value=[],
            ),
            
            dbc.Form([
                dbc.Label('Reviewing Capacity (Ratio)'),
                dcc.Slider(min=0, max=n_obs, value=n_obs / 2, id='capacity'),
                ]),
            
            ]
        ),
        dbc.Form(
            [
                dbc.Label('Cost of Reviewing per Unit'),
                dbc.Input(
                    id='cost_per_unit',
                    type='number',
                    min=0,
                    step=0.5,
                    value=0,
                    placeholder='Enter',
                    style={'width': '100px', 'margin-top': '10px', 'margin-bottom': '5px'},
                ),
            ],
        ),
        dbc.Form(
            [
                dbc.Label('Payout per Success (True Positive)'),
                dbc.Input(
                    id='payout_per_tp',
                    type='number',
                    min=0,
                    step=0.5,
                    value=0,
                    placeholder='Enter',
                    style={'width': '100px'},
                ),
            ],
        ),
        
    ]

    app.layout = dbc.Container(
        fluid=True,
        style={"height": "100vh"},
        children=[
            html.H1('Discrimination Threshold'),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(
                        width=3,
                        children=dbc.Card(
                            [dbc.CardHeader("Controls"), dbc.CardBody(controls),]
                        ),
                    ),
                    dbc.Col(
                        width=9,
                        children=dbc.Card(
                            [
                                dbc.CardHeader("The Plot"),
                                dbc.CardBody(dcc.Graph(id='discrimination_threshold'),
                                             style={'height': '100%'})
                            ],
                        ),
                    ),
                ],
            ),
        ],
    )

    # The interface for interaction between widgets and the plot
    @app.callback(
        Output('cost_per_unit', 'disabled'),
        Output('payout_per_tp', 'disabled'),
        Output('capacity', 'disabled'),
        Input('constrained', 'value'),
    )
    def update_constrained(constrained: List[str]):
        """Controls the possibility for inserting values for constrained mode
        when the constrained checkbox is marked."""
        if 'constrained' not in constrained:
            return (True, True, True)
        else:
            return (False, False, False)
    
    @app.callback(
        Output('discrimination_threshold', 'figure'),
        Input('recall_over_precision', 'value'),
        Input('interval_dropdown', 'value'),
        Input('metrics', 'value'),
        Input('constrained', 'value'),
        Input('capacity', 'value'),
        Input('cost_per_unit', 'value'),
        Input('payout_per_tp', 'value')
    )
    def update_graph(recall_over_precision: float,
                     interval_dropdown: int,
                     metrics: List,
                     constrained: List,
                     capacity: int,
                     cost_per_unit: float,
                     payout_per_tp: float):
        """Draw the plot based on the widget values"""
        if recall_over_precision != 1:
            data['f1'] = 0
            beta = recall_over_precision ** 2
            data.loc[data['precision'] * data['recall'] != 0, 'f1'] = (
                (1 + beta) * data['precision'] * data['recall'] 
                / (beta * data['precision'] + data['recall'])
            )
        df_grp = data.groupby('thresholds')
        thresholds = list(df_grp.groups.keys())
        metric_title = {'precision': 'Precision', 'recall': 'Recall',
                        'queue_rate': 'Queue Rate', 'f1': 'F1-Score'}
        px_colors = px.colors.qualitative.G10
        all_metrics = ['precision', 'recall', 'queue_rate', 'f1', 'payout']
        metric_color = {metric: px_colors[metric_id] for metric_id,
                        metric in enumerate(all_metrics)}
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for metric in metrics:
            fig.add_trace(
                go.Scatter(x=thresholds,
                           y=df_grp[metric].quantile(0.5),
                           mode='lines',
                           name=metric_title[metric],
                           marker={'color': metric_color[metric]},
                           hovertemplate=f'Median {metric_title[metric]}:'
                           + '%{y:.2f}' + '<br>Threshold: %{x:.2f}'
                           '<extra></extra>'),
                secondary_y=False)
            if interval_dropdown > 0:
                interval_min, interval_max = (0.5 - interval_dropdown / 2,
                                              0.5 + interval_dropdown / 2)
                fig.add_trace(
                    go.Scatter(x=thresholds,
                               y=df_grp[metric].quantile(interval_min),
                               mode='lines',
                               name=f'{metric_title[metric]}: min',
                               showlegend=False,
                               marker={'color': metric_color[metric]},
                               line={'dash': 'dot'},
                               hovertemplate='Lower '
                               f'{metric_title[metric]}:'
                               + '%{y:.2f}' + '<br>Threshold: %{x:.2f}'
                               '<extra></extra>'),
                    secondary_y=False)
                fig.add_trace(
                    go.Scatter(x=thresholds,
                               y=df_grp[metric].quantile(interval_max),
                               mode='lines',
                               name=f'{metric_title[metric]}: max',
                               showlegend=False,
                               marker={'color': metric_color[metric]},
                               line={'dash': 'dot'},
                               hovertemplate='Upper '
                               f'{metric_title[metric]}:'
                               + '%{y:.2f}' + '<br>Threshold: %{x:.2f}'
                               '<extra></extra>',
                               fill='tonexty'),
                    secondary_y=False)
        if 'constrained' in constrained:
            payout_median = (
                (df_grp['precision'].quantile(0.5)
                 * payout_per_tp- cost_per_unit)
                * df_grp['queue_rate'].quantile(0.5) * n_obs)
            fig.add_trace(
                go.Scatter(x=thresholds,
                           y=payout_median,
                           mode='lines',
                           hovertemplate='Median Payout:'
                           + '%{y:.2f}' + '<br>Threshold: %{x:.2f}'
                           '<extra></extra>',
                           name='payout',
                           marker={'color': metric_color['payout']}),
                secondary_y=True)
            if interval_dropdown > 0:
                payout_min = (
                    (df_grp['precision'].quantile(interval_min)
                     * payout_per_tp - cost_per_unit)
                    * df_grp['queue_rate'].quantile(interval_min) * n_obs)
                payout_max = (
                    (df_grp['precision'].quantile(interval_max) 
                     * payout_per_tp - cost_per_unit)
                    * df_grp['queue_rate'].quantile(interval_max) * n_obs)
                fig.add_trace(
                    go.Scatter(x=thresholds,
                               y=payout_min,
                               mode='lines',
                               line={'dash': 'dot'},
                               hovertemplate='Lower Payout:'
                               ' %{y:.2f}' + '<br>Threshold: %{x:.2f}'
                               '<extra></extra>',
                               name='payout',
                               marker={'color': metric_color['payout']},
                               showlegend=False),
                    secondary_y=True)
                fig.add_trace(
                    go.Scatter(x=thresholds,
                               y=payout_max,
                               mode='lines',
                               line={'dash': 'dot'},
                               hovertemplate='Upper Payout: %{y:.2f}'
                               '<br>Threshold: %{x:.2f}<extra></extra>',
                               name='payout',
                               marker={'color': metric_color['payout']},
                               showlegend=False,
                               fill='tonexty'),
                    secondary_y=True)
            fig.add_trace(
                go.Scatter(x=[capacity / n_obs, capacity / n_obs],
                           y=[0, 1],
                           mode='lines',
                           line=dict(color='black', dash='dot'),
                           name='Screening Capacity',
                           hovertemplate='Maximum Screening Capacity:'
                           ' %{x:.2f}<extra></extra>'),
                secondary_y=False)
            opt_threshold = payout_median[:(capacity + 0.02 / n_obs)].idxmax()
        else:
            opt_threshold = df_grp['f1'].quantile(0.5).idxmax()
        fig.add_trace(
            go.Scatter(x=[opt_threshold, opt_threshold],
                       y=[0, 1],
                       mode='lines',
                       line=dict(color='grey', width=2, dash='dash'),
                       name='Optimal Threshold',
                       hovertemplate='Optimal Threshold: %{x:.2f}'
                       '<extra></extra>'),
            secondary_y=False),
        fig.update_layout(title_x=0.08,
                          title_font_size=30,
                          xaxis_title='threshold',
                          xaxis=dict(ticks='outside'),
                          height=800,
                          autosize=True,
                          template='plotly_white',
                          legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1))
        fig.update_yaxes(title_text='score',
                         secondary_y=False)
        fig.update_yaxes(title_text='payout value',
                         secondary_y=True)
        return fig

    return app

if __name__ == "__main__":
    data = pd.read_csv('data.csv')
    app = build_plot(data=data, app_mode='server')
    app.run_server(debug=True, port=8051)