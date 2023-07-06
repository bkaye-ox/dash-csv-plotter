from dash import Dash, html, dcc, ctx
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import State, Output, DashProxy, Input, MultiplexerTransform
import dash_daq as daq
import dash_bootstrap_components as dbc

import pandas as pd


import math


import plibs.backend as lb


def bool_radio(n,  labels=None, default=None,):
    if default is None:
        default = [True] + [False for k in range(n-1)]
    if labels is None:
        labels = [(f'boolradio_{chr(97+k)}',
                   f'boolradio_{chr(97+k)}') for k in range(n)]
    return html.Div([
        dcc.Store(id='bool_radio', data='lines_bool')]+list(daq.BooleanSwitch(id=id, on=on, disabled=False,
                                                                              label=label) for on, (id, label) in zip(default, labels)),
        className='flx rw'
    )


app = DashProxy(__name__,
                prevent_initial_callbacks=True,
                transforms=[MultiplexerTransform()],
                external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])
application = app.server

store = html.Div([dcc.Store(id='file_memory', data={}),
                  dcc.Store(id='series_cache', data={}),
                  dcc.Store(id='index_range', data=[0, None]),
                  dcc.Store(id='update_graph', data=0), ])


upload_bar = html.Div([

], className='flx rw sp-ev')

header = html.Div([
    html.Div('Analysis Suite', className='header-h1'),
    dcc.Upload('Upload or drag files here',
               id='upload-data', className='upload', multiple=True),
    html.Div(
        dcc.Dropdown(id='file_selector', className='drop', multi=True, options=[]), style={'width': '20em'}),
    dbc.Button(html.Div([html.I(className="bi bi-list me-2"),
               ' Controls'], className='header-button'), color="secondary", id='sidebar-open'),
    html.Div(id='status_text')
], className='hdr flx rw hfill sp-ev wrap')


figure_source = html.Div([
    html.Div([
        html.Div([
            html.Div('x-axis source'),
            dcc.Dropdown(id='header_drop_x', placeholder='x',
                         options=[], className='drop')
        ], className='input-div'),
        html.Div([
            html.Div('y-axis source'),
            dcc.Dropdown(id='header_drop_y', placeholder='y(s)',
                         options=[], multi=True, className='drop'),

        ], className='input-div'),
        html.Div([
            html.Div('secondary-y-axis source'), dcc.Dropdown(id='header_drop_alty', placeholder='alt y(s)',
                                                              options=[], multi=True, className='drop')
        ], className='input-div'),
    ], style={'width': '80%'}, className='left-align flx cl'),
    html.Div([
        # daq.BooleanSwitch(id='spectrum_bool', disabled=True,
        #                   label='Spectrum'),
        daq.BooleanSwitch(id='filter_button', disabled=False,
                          label='Filter'),
        html.Div([
            html.Div('window size'),
            dcc.Input(id='filt_win', value=61, type='number',
                         className='num-input')
        ], className='small-input-div'),
    ], className='flex cl'),
], className='flx rw left-align sp-ev hfill')

figure_layout = html.Div([
    html.Div([
        html.Div('x-axis label'),
        dcc.Input(id='x_label', placeholder='x-axis label..',
                  debounce=True, className='input'),
    ], className='input-div'),
    html.Div([
        html.Div('y-axis label'),
        dcc.Input(id='y_label', placeholder='y-axis label..',
                  debounce=True, className='input'),
    ], className='input-div'),
    html.Div([
        html.Div('secondary y-axis label'),
        dcc.Input(id='alt_y_label', placeholder='y-axis label..',
                  debounce=True, className='input'),
    ], className='input-div'),
    html.Div([
        html.Div('plot title'),
        dcc.Input(id='title_in', placeholder='title..',
                  debounce=True, className='input'),
    ], className='input-div'),


], className='flx cl left-align sp-ev hfill')

figure_time = html.Div([
    html.Div([
        html.Div([
            html.Div('time source'),
            dcc.Dropdown(id='drop_time', placeholder='TimeS',
                         options=[], className='drop')

        ], className='input-div'),
        daq.BooleanSwitch(id='time_switch', on=False, disabled=True,
                             label='Enable'),

    ], className='flx rw sp-ev hfill'),
    dcc.RangeSlider(0, 0, value=[0, 0], id='slider', className='slider'),
], className='flx cl hfill left-align str sp-ev')

figure_wildcard = html.Div(
    bool_radio(3, [('lines_bool', 'Normal'),
               ('spectrum_bool', 'Spectrum'), ('box_bool', 'Box')]),
    className='hfill')

figure_filter_settings = html.Div(

)

sidebar = dbc.Offcanvas(
    # html.Div([
    #     figure_source,
    #     figure_time,
    #     figure_layout,
    # ]),
    dbc.ListGroup([
        dbc.ListGroupItem(figure_source, className='sidebar-cont'),
        dbc.ListGroupItem(figure_time, className='sidebar-cont'),
        dbc.ListGroupItem(figure_layout, className='sidebar-cont'),
        dbc.ListGroupItem(figure_wildcard, className='sidebar-cont'),
    ]),
    id="sidebar",
    title="Graph controls",
    is_open=False,)

# graph_controls = html.Div([
#     figure_source,
#     figure_layout,
#     figure_time
# ], className='flx rw hfill sp-ev', style={'minHeight': '6em'})


app.layout = html.Div([
    store,
    sidebar,
    html.Div([
        header,
        html.Div([
            html.Div(dcc.Graph(id='graph', className='hfill graph-container'),

                     className='graph-sizer')  # graph_controls,

        ], className='flx cl str app-body'),
    ], className='app-arranger')
], className='app-cont')


@app.callback(Output('series_cache', 'data'), Output('sidebar', 'is_open'), Output('file_selector', 'options'), Output('file_selector', 'value'), Input('upload-data', 'contents'),
              State('upload-data', 'filename'), State('file_memory', 'data'), State('series_cache', 'data'), State('file_selector', 'value'))
def file_ready(list_of_contents, list_of_names, file_mem, cache, fn_select):
    if list_of_contents is None:
        raise PreventUpdate

    if cache is None:
        cache = {}

    for fn, contents in zip(list_of_names, list_of_contents):
        df = lb.extract_data(fn, contents)
        if df is not None:
            cache[fn] = df

    fn_select = list_of_names if fn_select is None else fn_select + list_of_names

    return cache, True, list(cache.keys()), fn_select


@app.callback(Output('sidebar', 'is_open'), Input('sidebar-open', 'n_clicks'), State('sidebar', 'is_open'))
def open_sb(n_clicks, open):
    return not open


@app.callback(Output('graph', 'figure'),
              Input('update_graph', 'data'),
              State('file_selector', 'value'), State(
                  'series_cache', 'data'), State('file_memory', 'data'),
              State('header_drop_x', 'value'), State('header_drop_y',
                                                     'value'), State('header_drop_alty', 'value'),
              State('index_range', 'data'), State(
                  'filter_button', 'on'), State('filt_win', 'value'),
              State('x_label', 'value'), State(
                  'y_label', 'value'), State('alt_y_label', 'value'), State('title_in', 'value'),
              State('bool_radio', 'data'))
def graph_update(_, fns, dataframe, csv, col_x, cols_y, cols_y2, range_, filter_on, filt_window, xlabel, ylabel, altyabel, title, plot_type):

    if fns is None or len(fns) == 0:
        raise PreventUpdate
    
    if (not cols_y and not cols_y2):
        raise PreventUpdate

    col_x, cols_y, cols_y2 = lb.format_cols(col_x, cols_y, cols_y2)

    second_axis = bool(len(cols_y2) > 0)

    

    if plot_type == 'lines_bool':
        filt = (filter_on, filt_window)

        fig = lb.make_fig(col_x, cols_y, cols_y2, dataframe, range_, filt, fns)

    if plot_type == 'spectrum_bool':

        h_st = None
        if len(cols_y) == 1:

            h_st = cols_y[0]

        elif len(cols_y2) == 1:

            h_st = cols_y2[0]

        fig = lb.make_spectrum(col_x, h_st, dataframe, range_, fns)

    if plot_type == 'box_bool':

        fig = lb.make_box(col_x, cols_y, fns, dataframe)

    fig['layout']['xaxis']['title'] = xlabel
    fig['layout']['yaxis']['title'] = ylabel
    if second_axis:
        fig['layout']['yaxis2']['title'] = altyabel
    fig['layout']['title'] = title

    # , f'{col_x} vs. {",".join(cols_y)}', f'{col_x}', f'{",".join(hy)}'
    return fig


@ app.callback([Output('header_drop_x', 'options'), Output('header_drop_y', 'options'), Output('header_drop_alty', 'options'), Output('drop_time', 'options'), Output('status_text', 'children')],
               Input('series_cache', 'data'),
               State('file_selector', 'value'))
def update_output(csv, fns):

    disp_headers = list(set.intersection(
        *(set(h for h in csv[fn].keys()) for fn in fns)))
    # disp_headers = list(headers[0].intersect(*headers))
    return *(disp_headers for k in range(4)), f'file uploaded'


@ app.callback(Output('header_drop_x', 'value'), Output('header_drop_y', 'value'), Output('drop_time', 'value'), Input('header_drop_x', 'options'))
def update_default_drop(opts):
    h_x = 'TimeS' if 'TimeS' in opts else None
    h_y = 'ArterialTemperature' if 'ArterialTemperature' in opts else None
    h_t = 'TimeS' if 'TimeS' in opts else None
    return h_x, h_y, h_t


@ app.callback(Output('update_graph', 'data'),  # Output('title_in', 'value'), Output('x_label', 'value'), Output('y_label', 'value'),
               Input('filter_button', 'on'), Input('filt_win', 'value'), Input('header_drop_x', 'value'), Input('header_drop_y', 'value'), Input(
                   'header_drop_alty', 'value'), Input('index_range', 'data'), Input('x_label', 'value'), Input('y_label', 'value'), Input('alt_y_label', 'value'), Input('title_in', 'value'), Input('file_selector', 'value'), Input('bool_radio', 'data'),
               State('update_graph', 'data'))
def update_fig_cb(*args):
    # header_x, header_y, header_alty,spectrum, range, update_count
    update_count = args[-1]
    return update_count + 1
    #new_cache, f'{header_x} vs. {",".join(header_y)}', f'{header_x}', f'{",".join(header_y)}'


@app.callback(Output('time_switch', 'disabled'), Output('slider', 'disabled'), Input('drop_time', 'value'))
def disable_ts(val):
    if val is None:
        return tuple(True for k in range(2))
    else:
        return tuple(False for k in range(2))


@ app.callback(Output('index_range', 'data'),
               Input('slider', 'value'), Input('time_switch', 'on'),
               State('series_cache', 'data'), State('drop_time', 'value'))
def slidercb(value, time_src_enable, cache, time_col):
    t_min, t_max = value
    k_min, k_max = 0, None

    if not time_src_enable:
        return [0, None]

    ranges = {}
    for fn, v in cache.items():
        t = v[time_col]
        for k, kt in enumerate(t):
            if k_min == 0 and kt > t_min:
                k_min = k
            if k_max is None and kt is not None and kt > t_max:
                k_max = k
        ranges[fn] = [k_min if k_min is not None else 0, k_max]

    return ranges


@app.callback(Output('spectrum_bool', 'disabled'), Output('spectrum_bool', 'on'), Input('header_drop_x', 'value'), Input('header_drop_y', 'value'), Input('header_drop_alty', 'value'))
def update_sb(hx, hy, hay):
    if hx is not None and hy is not None and (hay is None or not hay) and (isinstance(hy, str) or len(hy) == 1):
        return False, False
    else:
        return True, False


@ app.callback(Output('slider', 'min'), Output('slider', 'max'), Output('slider', 'value'), Input('drop_time', 'value'), State('series_cache', 'data'))
def droptimecb(time_col, cache):

    maxt = -math.inf
    mint = math.inf
    for _, v in cache.items():

        time = v.get(time_col)
        if time is None:
            continue
        t = tuple(x for x in time if not math.isnan(x))
        maxt = max(maxt, max(t))
        mint = min(mint, min(t))

    return mint, maxt, [mint, maxt]


@app.callback(Output('bool_radio', 'data'), Output('lines_bool', 'on'), Output('spectrum_bool', 'on'), Output('box_bool', 'on'), Input('lines_bool', 'on'), Input('spectrum_bool', 'on'), Input('box_bool', 'on'))
def cb_radio_list(*on):

    if not any(on):
        return 'lines_bool', True, False, False

    triggered_id = ctx.triggered_id
    val = ctx.inputs
    if val[f'{triggered_id}.on'] == False:
        raise PreventUpdate
    outputs = {
        'lines_bool': False,
        'spectrum_bool': False,
        'box_bool': False,
    }
    outputs[triggered_id] = True
    return triggered_id, *(outputs.values())


if __name__ == '__main__':

    app.run_server(debug=False)
