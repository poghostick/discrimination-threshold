from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import flask
from jupyter_dash import JupyterDash
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Union


def build_plot(data: pd.DataFrame, width: int = 1200,
               height: int = 550, test_size=0.2,
               app_mode='jupyter_inline') -> Union[Dash, JupyterDash]:
    """Generates an interactive version of the discrimination threshold
    plot.

    Parameters
    ----------
        data: pd.DataFrame
            Data to be used an input for the plot
        width: int, optional
            Width of the plot (default is 1200)
        height: int, optional
            Height of the plot (default is 550)
        test_size: float, optional
            Share of the test set (default is 0.2)
        app_mode: str, optional
                If 'inline', the app is being created as instance of
                JupyterDash and the plot is drawn inside the notebook; if
                'external', then the app is created as JupyterDash
                instance and the plot is drawn in separate window; when
                'server' the app is created as Dash instance to be
                deployed as a server application
                (default is 'jupyter_inline')
    Returns
    -------
        An instance of Dash or JupyterDash to be used for plotting.
    """
    # Build all the widgets generating additional parameters for the plot
    n_obs = int(data.shape[0] * test_size)
    if app_mode in ('inline', 'external'):
        app = JupyterDash(__name__)
    else:
        server = flask.Flask(__name__)
        app = Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = html.Div([
        html.Div([
            html.Div(
                [
                    'Interval',
                    dcc.Dropdown(
                        id='interval_dropdown',
                        clearable=False,
                        value=0.95,
                        options=[
                            {'label': f'{ii}%', 'value': ii / 100}
                            for ii in [0, 5, 10, 25, 50, 75, 90, 95]
                        ],
                        style={'margin-top': '10px'}
                    ),
                ],
                style={'width': '100px', 'display': 'inline-block',
                       'margin-left': '5%', 'margin-top': '25px'}
            ),
            html.Div(
                dcc.Checklist(
                    id='checklist',
                    options=[
                        {'label': ' Precision', 'value': 'precision'},
                        {'label': ' Recall', 'value': 'recall'},
                        {'label': ' F1 Score', 'value': 'f1'},
                        {'label': ' Queue Rate', 'value': 'queue_rate'}],
                    value=['precision', 'recall', 'queue_rate'],
                    labelStyle={'display': 'block'},
                ),
                style={'width': '10%', 'display': 'inline-block',
                       'margin-left': '7%', 'margin-top': '25px'}
            ),
            html.Div([
                dcc.Checklist(
                    id='payout',
                    options=[{'label': ' Optimal Threshold by Screening'
                              ' Capacity, Cost and Revenue', 'value':
                                  'payout'}],
                    value=[],
                    labelStyle={'display': 'block',
                                'margin-bottom': '10px'},
                ),
                html.Div([dcc.Slider(id='slider_id', min=0,
                                     max=n_obs, value=n_obs / 2)],
                         style={'width': '500px'}),
                html.Div([
                    html.Label('Cost of screening per unit'),
                    dcc.Input(
                        id='cost_per_unit',
                        type='number',
                        step=0.5,
                        value=0,
                        placeholder='Enter cost (+) per unit',
                        style={'width': '190px', 'margin-bottom': '1%',
                               'margin-left': '65px'})]),
                html.Div([
                    html.Label('Revenue / Loss per True Positive'),
                    dcc.Input(
                        id='revenue_loss',
                        type='number',
                        step=0.5,
                        value=0,
                        placeholder='Enter revenue (+) or loss (-)',
                        style={'width': '190px', 'margin-bottom': '1%',
                               'margin-left': '10px'})])],
                style={'width': '50%', 'display': 'inline-block',
                       'margin-top': '20px', 'margin-left': '5%'})],
            style={'display': 'flex'},
        ),
        dcc.Graph(id='discrimination_threshold'),
    ])

    # The interface for interaction between widgets and the plot
    @app.callback(
        Output('cost_per_unit', 'disabled'),
        Output('revenue_loss', 'disabled'),
        Output('slider_id', 'disabled'),
        Input('payout', 'value'),
    )
    def update_payout(payout: List[str]):
        """Controls the possibility for inserting payout values when
        when the payout checkbox is marked."""
        if 'payout' not in payout:
            return (True, True, True)
        else:
            return (False, False, False)

    @app.callback(
        Output('discrimination_threshold', 'figure'),
        Input('interval_dropdown', 'value'),
        Input('checklist', 'value'),
        Input('payout', 'value'),
        Input('slider_id', 'value'),
        Input('cost_per_unit', 'value'),
        Input('revenue_loss', 'value'))
    def update_graph(interval_dropdown: int,
                     checklist: List,
                     payout: List,
                     slider: int,
                     cost_per_unit: float,
                     revenue_loss: float):
        """Draws the plot based on the widget values"""
        df_grp = data.groupby('thresholds')
        thresholds = list(df_grp.groups.keys())
        # widget_values = get_widget_values(widget)
        metric_title = {'precision': 'Precision', 'recall': 'Recall',
                        'queue_rate': 'Queue Rate', 'f1': 'F1-Score'}
        px_colors = px.colors.qualitative.G10
        metrics = ['precision', 'recall', 'queue_rate', 'f1', 'payout']
        metric_color = {metric: px_colors[metric_id] for metric_id,
                        metric in enumerate(metrics)}
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for metric in checklist:
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
        if 'payout' in payout:
            payout_median = (
                (df_grp['precision'].quantile(0.5) * 100
                 * revenue_loss - cost_per_unit)
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
                    (df_grp['precision'].quantile(interval_min) * 100 *
                     revenue_loss - cost_per_unit)
                    * df_grp['queue_rate'].quantile(interval_min) * n_obs)
                payout_max = (
                    (df_grp['precision'].quantile(interval_max) * 100
                     * revenue_loss - cost_per_unit)
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
                go.Scatter(x=[slider / n_obs, slider / n_obs],
                           y=[0, 1],
                           mode='lines',
                           line=dict(color='black', dash='dot'),
                           name='Screening Capacity',
                           hovertemplate='Maximum Screening Capacity:'
                           ' %{x:.2f}<extra></extra>'),
                secondary_y=False)
            opt_threshold = payout_median[:(slider / n_obs)].idxmax()
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
        fig.update_layout(title='Discrimination Threshold',
                          title_x=0.08,
                          title_font_size=30,
                          xaxis_title='threshold',
                          xaxis=dict(ticks='outside'),
                          autosize=True,
                          width=width,
                          height=height,
                          template='plotly_white')
        fig.update_yaxes(title_text='score',
                         secondary_y=False)
        fig.update_yaxes(title_text='payout value',
                         secondary_y=True)
        return fig
    
    return app
    

if __name__ == '__main__':
    # The function is called from here only when the app is deployed
    # and ran from server
    data = pd.read_csv('data.csv')
    app = build_plot(data=data, width=1400, height=900)
    app.run_server(port=8051)
