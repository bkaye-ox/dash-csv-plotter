from dash import Dash, html, dcc
import pandas as pd
from dash.exceptions import PreventUpdate

import dash_daq as daq

import base64

import plotly.graph_objects as go
import plotly.subplots as psp

import math
# app = Dash(__name__)

from dash_extensions.enrich import State, Output, DashProxy, Input, MultiplexerTransform

app = DashProxy(__name__, prevent_initial_callbacks=True,
                transforms=[MultiplexerTransform()])

store = html.Div([dcc.Store(id='file_memory', data=[None, None]),
                  dcc.Store(id='series_cache', data={}),
                  dcc.Store(id='index_range', data=[0, None]),
                  dcc.Store(id='file_ready', data=False), ])


upload_bar = html.Div([
    dcc.Upload('Upload or drag files here',
               id='upload-data', className='upload'),
    html.Div('Current file: .', id='status_text', className='status')
], className='flx rw hfill sp-ev')

header = html.Div([html.H1('Analysis Suite', className='header-h1'),
                  upload_bar], className='flx rw hfill sp-ev hdr')

figure_source = html.Div([
    html.Div([
        html.Div('x-axis source'),
        dcc.Dropdown(id='header_drop_x', placeholder='x',
                     options=[], className='drop')
    ], className='input-div'),
    html.Div([
        html.Div('y-axis source'), dcc.Dropdown(id='header_drop_y', placeholder='y(s)',
                                                options=[], multi=True, className='drop')
    ], className='input-div'),
], className='flx cl hfill sp-ev')

figure_layout = html.Div([
    dcc.Input(id='x_label', placeholder='x-axis label..',
              debounce=True, className='input'),
    dcc.Input(id='y_label', placeholder='y-axis label..',
              debounce=True, className='input'),
    dcc.Input(id='title_in', placeholder='title..',
              debounce=True, className='input'), ], className='flx cl hfill sp-ev str')

figure_time = html.Div([
    daq.BooleanSwitch(id='time_switch', on=False,
                      label='Enable time source', labelPosition='left'),
    html.Div([

        html.Div([
            html.Div('time source'),
            dcc.Dropdown(id='drop_time', placeholder='TimeS',
                         options=[], className='drop')
        ]),
    ], className='flx rw hfill sp-bt no-wrap'),
    dcc.RangeSlider(0, 0, value=[0, 0], id='slider', className='slider'),
], className='flx cl hfill left-align str sp-ev')

graph_controls = html.Div([
    figure_source,
    figure_layout,
    figure_time
], className='flx rw hfill sp-ev', style={'minHeight': '6em'})


app.layout = html.Div([
    store,
    html.Div([
    header,
    html.Div([
        graph_controls,
        dcc.Graph(id='graph', style={'height': '70vh'}, className='hfill'),
    ], className='app-body flx cl str sp-ev'),
    ], className='app-arranger')
], className='app-cont')


@app.callback([Output('series_cache', 'data'), Output('file_memory', 'data')], Input('upload-data', 'contents'),
              State('upload-data', 'filename'))
def file_ready(list_of_contents, list_of_names):
    if list_of_contents is None:
        raise PreventUpdate

    fn = list_of_names
    contents = list_of_contents

    content_type, content_b64 = contents.split(',')

    content_str = base64.b64decode(content_b64).decode()
    end_idx = content_str.find('\r\n')

    headers = content_str[:end_idx].split(',')

    csv = (headers, content_str[end_idx+1:])

    time = get_time(csv)
    cache = {}
    if time is not None:
        csv[0].append('TimeS')
        cache = {'TimeS': time}

    return cache, csv


@ app.callback([Output('header_drop_x', 'options'), Output('header_drop_y', 'options'), Output('drop_time', 'options'), Output('status_text', 'children')],
               Input('file_memory', 'data'),
               State('upload-data', 'filename'))
def update_output(csv, fn):
    disp_headers = [h for h in csv[0] if h != "''"]
    return *(disp_headers for k in range(3)), f'file uploaded: {fn}'


@ app.callback(Output('header_drop_x', 'value'), Output('header_drop_y', 'value'), Output('drop_time', 'value'), Input('header_drop_x', 'options'))
def update_default_drop(opts):
    h_x = 'TimeS' if 'TimeS' in opts else None
    h_y = 'ArterialTemperature' if 'ArterialTemperature' in opts else None
    h_t = 'TimeS' if 'TimeS' in opts else None
    return h_x, h_y, h_t


def get_time(csv):

    headers = csv[0]
    time_header = 'PerfusionTime' if 'PerfusionTime' in headers else 'CurrentTicks' if 'CurrentTicks' in headers else None

    if time_header == 'PerfusionTime':
        return process_cols(time_header, csv)
    if time_header == 'CurrentTicks':
        ticks, tps = process_cols((time_header, 'TicksPerSecond'), csv)
        time = [None for k in range(len(ticks))]
        time[0] = 0
        for k in range(1, len(ticks)):
            time[k] = time[k-1] + (ticks[k] - ticks[k-1])*tps[k]*1e-6
        return time
    return None


@ app.callback([Output('graph', 'figure'), Output('series_cache', 'data'), Output('title_in', 'value'), Output('x_label', 'value'), Output('y_label', 'value')],
               [Input('header_drop_x', 'value'), Input('header_drop_y', 'value'), Input('file_memory', 'data')], State('series_cache', 'data'), State('index_range', 'data'))
def dropcb(header_x, header_y, csv, cache, range):
    if not header_x or not header_y:
        raise PreventUpdate

    header_y = (header_y,) if isinstance(header_y, str) else header_y

    new_cache = cache | cache_data((header_x,)+tuple(header_y), cache, csv)

    return make_fig(header_x, header_y, new_cache, range), new_cache, f'{header_x} vs. {",".join(header_y)}', f'{header_x}', f'{",".join(header_y)}'


def make_fig(h_x, h_ys, cache, range):
    x_c = cache.get(h_x)
    y_c = [cache.get(hy) for hy in h_ys]
    if not (x_c and all(y_c)):
        raise PreventUpdate

    x = x_c[range[0]:range[1]]
    ys = [y[range[0]:range[1]] for y in y_c]
    traces = [go.Scatter(x=x, y=yk, name=hyk) for yk, hyk in zip(ys, h_ys)]
    fig = psp.make_subplots()
    fig.add_traces(traces)
    return fig


@ app.callback([Output('index_range', 'data')], Input('slider', 'value'), State('series_cache', 'data'), State('drop_time', 'value'), State('time_switch','on'))
def slidercb(value, cache, time_col, time_src_enable):
    t_min, t_max = value
    k_min, k_max = 0, None

    if not time_src_enable:
        return [0,None]

    t = cache[time_col]
    for k, kt in enumerate(t):
        if k_min == 0 and kt > t_min:
            k_min = k
        if k_max is None and kt is not None and kt > t_max:
            k_max = k
    return [k_min if k_min is not None else 0, k_max]


@ app.callback(Output('graph', 'figure'), Input('index_range', 'data'), State('series_cache', 'data'), State('header_drop_x', 'value'), State('header_drop_y', 'value'))
def update_fig_on_range_change(index_range, cch, hx, hy):
    return make_fig(hx, (hy,) if isinstance(hy, str) else hy, cch, index_range)


@ app.callback(Output('slider', 'min'), Output('slider', 'max'), Output('slider', 'value'), Input('drop_time', 'value'), State('series_cache', 'data'))
def droptimecb(inval, cache):

    if cache.get(inval):
        time = cache[inval]
    else:
        raise PreventUpdate

    min_t, max_t = time[0] if time[0] is not None else None, None
    for k, kt in enumerate(time):
        if min_t is None and kt:
            min_t = kt
        if max_t is None and kt:
            max_t = kt
        if kt and kt > max_t:
            max_t = kt

    max_t = round(max_t,2)

    return min_t, max_t, [min_t, max_t]


@ app.callback(Output('graph', 'figure'), [Input('x_label', 'value'), State('graph', 'figure')])
def x_axis_cb(label_in, fig):
    if fig is not None:
        fig['layout']['xaxis']['title'] = label_in
    return fig


@ app.callback(Output('graph', 'figure'), [Input('y_label', 'value'), State('graph', 'figure')])
def y_axis_cb(label_in, fig):
    if fig is not None:
        fig['layout']['yaxis']['title'] = label_in
    return fig


@ app.callback(Output('graph', 'figure'), [Input('title_in', 'value'), State('graph', 'figure')])
def y_axis_cb(label_in, fig):
    if fig is not None:
        fig['layout']['title'] = label_in
    return fig


def cache_data(cols, cache, csv):
    not_cached = [k for k in cols if not cache.get(k)]
    return {col: series for col, series in zip(not_cached, process_cols(not_cached, csv))}


def process_cols(cols, csv):
    headers, contents = csv

    rows = contents.split('\r\n')

    k_cols = tuple(headers.index(y) for y in cols)

    data = [None for k in range(len(rows))]

    nan_count = [0 for k in range(len(cols))]

    # try cast columns to float
    for k, row in enumerate(rows):
        r = row.split(',')

        data[k] = tuple(v for v in get_col_val(k_cols, r))

        for h, v in enumerate(data[k]):
            if v is math.nan:
                nan_count[h] += 1

    # if dtype is not float, revert to string
    for h, nan_c in enumerate(nan_count):
        if nan_c > 0.5*len(rows):
            data[h] = [None for k in range(len(rows))]
            for k, r in enumerate(rows):
                try:
                    data[h][k] = r.split(',')[k_cols[h]]
                except:
                    data[h][k] = ''

    return zip(*data)


@app.callback(Output('index_range', 'data'), Input('time_switch','on'))
def enable_time_src(switch_on):
    if switch_on:
        return PreventUpdate
    else:
        return [0,None]
    

def get_col_val(col_idxs, r):
    '''Generator()'''
    for col in col_idxs:
        try:
            val = float(r[col])
        except:
            val = math.nan
        yield val


if __name__ == '__main__':
    app.run_server(debug=True)
