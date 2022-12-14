from dash import Dash, html, dcc, ctx
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import State, Output, DashProxy, Input, MultiplexerTransform
import dash_daq as daq
import dash_bootstrap_components as dbc

import plotly.graph_objects as go
import plotly.subplots as psp

import pandas as pd
import scipy.signal as sps
import scipy as sp
import numpy as np

import math
import base64
import io
import csv

app = DashProxy(__name__,
                prevent_initial_callbacks=True,
                transforms=[MultiplexerTransform()],
                external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])

store = html.Div([dcc.Store(id='file_memory', data={}),
                  dcc.Store(id='series_cache', data={}),
                  dcc.Store(id='index_range', data=[0, None]),
                  dcc.Store(id='update_graph', data=0), ])


upload_bar = html.Div([

], className='flx rw sp-ev')

header = html.Div([
    html.Div('Analysis Suite', className='header-h1'),
    dcc.Upload('Upload or drag files here',
               id='upload-data', className='upload'),
    html.Div(
    dcc.Dropdown(id='file_selector', className='drop', multi=True, options=[]), style={'width':'20em'}),
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
        daq.BooleanSwitch(id='spectrum_button', disabled=True,
                          label='Spectrum'),
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


@app.callback(Output('series_cache', 'data'), Output('file_memory', 'data'), Output('sidebar', 'is_open'), Output('file_selector', 'options'), Output('file_selector', 'value'), Input('upload-data', 'contents'),
              State('upload-data', 'filename'), State('file_memory', 'data'), State('series_cache', 'data'), State('file_selector', 'value'))
def file_ready(list_of_contents, list_of_names, file_mem, cache, fn_select):
    if list_of_contents is None:
        raise PreventUpdate

    fn = list_of_names
    contents = list_of_contents

    content_type, content_b64 = contents.split(',')
    content_str = base64.b64decode(content_b64).decode()

    rows = parse_csv(content_str)
    csv = (rows[0], rows[2:-1])

    new_file_mem = file_mem | {fn: csv}

    time = get_time(csv)
    new_cache = {fn: {}}
    if time is not None:
        csv[0].append('TimeS')
        new_cache = {fn: {'TimeS': time}}

    if fn_select is not None:
        fn_select.append(fn)
    else:
        fn_select = [fn]
    return cache | new_cache, new_file_mem, True, list(new_file_mem.keys()), fn_select


def parse_csv(string, lb='\r\n', quote='"', delim=','):
    # el_start = 0
    # hold = False

    # rows = [[]]
    # for k, s in enumerate(string):
    #     hold = not hold if s == quote else hold

    #     if not hold and s == delim:
    #         rows[-1].append(string[el_start:k])
    #         el_start = k+1
    #     if s == '\r' and string[k:k+2] == lb:
    #         rows[-1].append(string[el_start:k])
    #         row_start = k+2
    #         el_start = k+2
    #         hold = False
    #         rows.append([])
    # return rows
    return [row for row in csv.reader(io.StringIO(string))]


@app.callback(Output('sidebar', 'is_open'), Input('sidebar-open', 'n_clicks'), State('sidebar', 'is_open'))
def open_sb(n_clicks, open):
    return not open


@app.callback(Output('graph', 'figure'), Output('series_cache', 'data'),
              Input('update_graph', 'data'),
              State('file_selector', 'value'), State('series_cache', 'data'), State('file_memory', 'data'), State('header_drop_x', 'value'), State('header_drop_y', 'value'), State('header_drop_alty', 'value'), State('index_range', 'data'), State('spectrum_button', 'on'), State('filter_button', 'on'), State('filt_win', 'value'), State('x_label', 'value'), State('y_label', 'value'), State('title_in', 'value'))
def graph_update(_, fns, cache, csv, hx, hy, hay, range, spectrum, filter_on, filt_window, xlabel, ylabel, title):

    if fns is None or len(fns) == 0:
        raise PreventUpdate

    if not hx or (not hy and not hay):
        raise PreventUpdate

    hy = (hy,) if isinstance(hy, str) else tuple(
        hy) if isinstance(hy, list) else hy
    hay = (hay,) if isinstance(hay, str) else tuple(
        hay) if isinstance(hay, list) else hay

    cols = (hx,) + hy + \
        hay if hay is not None else (hx,) + hy

    new_cache = cache_data(cols, cache, csv, fns)

    if spectrum:
        hy_singleton = hy if isinstance(hy, str) else hy[0] if isinstance(
            hy, list) or isinstance(hy, tuple) else None
        fig = make_spectrum(hx, hy_singleton, new_cache, range, fns)
    else:
        filt = (filter_on, filt_window)

        fig = make_fig(hx, hy, hay, new_cache, range, filt, fns)

    fig['layout']['xaxis']['title'] = xlabel
    fig['layout']['yaxis']['title'] = ylabel
    fig['layout']['title'] = title

    # , f'{hx} vs. {",".join(hy)}', f'{hx}', f'{",".join(hy)}'
    return fig, new_cache


@ app.callback([Output('header_drop_x', 'options'), Output('header_drop_y', 'options'), Output('header_drop_alty', 'options'), Output('drop_time', 'options'), Output('status_text', 'children')],
               Input('file_memory', 'data'),
               State('file_selector', 'value'))
def update_output(csv, fns):


    disp_headers = list(set.intersection(*(set(h for h in csv[fn][0]) for fn in fns)))
    # disp_headers = list(headers[0].intersect(*headers))
    return *(disp_headers for k in range(4)), f'file uploaded'


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


@ app.callback(Output('update_graph', 'data'),  # Output('title_in', 'value'), Output('x_label', 'value'), Output('y_label', 'value'),
               Input('filter_button', 'on'), Input('filt_win', 'value'), Input('header_drop_x', 'value'), Input('header_drop_y', 'value'), Input(
                   'header_drop_alty', 'value'), Input('spectrum_button', 'on'), Input('index_range', 'data'), Input('x_label', 'value'), Input('y_label', 'value'), Input('title_in', 'value'), Input('file_selector','value'),
               State('update_graph', 'data'))
def update_fig_cb(*args):
    # header_x, header_y, header_alty,spectrum, range, update_count
    update_count = args[-1]
    return update_count + 1
    #new_cache, f'{header_x} vs. {",".join(header_y)}', f'{header_x}', f'{",".join(header_y)}'


def make_fig(h_x, h_ys, h_ays, super_cache, id_range, filt, fns):
    trs = []
    scnds = []
    for fn in fns:
        cache = super_cache.get(fn)

        trs_fn, seconds_fn = make_fig_traces(
            h_x, h_ys, h_ays, cache, id_range, filt)

        trs.extend(trs_fn)
        if seconds_fn is not None:
            scnds.extend(seconds_fn)

    if h_ays is not None:
        fig = psp.make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_traces(trs, scnds)
    else:
        fig = psp.make_subplots()
        fig.add_traces(trs)
    return fig

def make_fig_traces(h_x, h_ys, h_ays, cache, id_range, filt):
    h_ys = (h_ys,) if isinstance(h_ys, str) else tuple(
        h_ys) if isinstance(h_ys, list) else h_ys
    h_ays = (h_ays,) if isinstance(h_ays, str) else tuple(
        h_ays) if isinstance(h_ays, list) else h_ays

    x_c = cache.get(h_x)
    y_c = [cache.get(hy) for hy in h_ys]

    if not (x_c and all(y_c)):
        raise PreventUpdate

    if filt[0]:
        y_c = [filter(y, filt[1]) for y in y_c]
    x = x_c[id_range[0]:id_range[1]]
    ys = [y[id_range[0]:id_range[1]] for y in y_c]

    if h_ays is not None:
        alt_y_c = [cache.get(hy) for hy in h_ays]

        if filt[0]:
            alt_y_c = [filter(y, filt[1]) for y in alt_y_c]
        alt_ys = [y[id_range[0]:id_range[1]] for y in alt_y_c]

    second = None
    if h_ays is not None:
        traces = [go.Scatter(x=x, y=yk, name=hyk)
                  for yk, hyk in zip(ys+alt_ys, h_ys+h_ays)]
        second = tuple(False for k in range(len(ys))) + \
            tuple(True for k in range(len(alt_ys)))
    else:
        traces = [go.Scatter(x=x, y=yk, name=hyk) for yk, hyk in zip(ys, h_ys)]

    return traces, second


def filter(series, window):

    nan_as_zeros = tuple(0 if math.isnan(yk) else yk for yk in series)

    return sp.ndimage.uniform_filter1d(
        nan_as_zeros, window, mode='constant', cval=0)


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

    t = cache[time_col]
    for k, kt in enumerate(t):
        if k_min == 0 and kt > t_min:
            k_min = k
        if k_max is None and kt is not None and kt > t_max:
            k_max = k
    return [k_min if k_min is not None else 0, k_max]


@app.callback(Output('spectrum_button', 'disabled'), Output('spectrum_button', 'on'), Input('header_drop_x', 'value'), Input('header_drop_y', 'value'), Input('header_drop_alty', 'value'))
def update_sb(hx, hy, hay):
    if hx is not None and hy is not None and (hay is None or not hay) and (isinstance(hy, str) or len(hy) == 1):
        return False, False
    else:
        return True, False


def make_spectrum(hx, hy, ch, rng, fns):
    trs = []
    for fn in fns:
        cache = ch.get(fn)
        trs.append(make_spectrum_tr(hx, hy, cache, rng))
    fig = psp.make_subplots()
    fig.add_traces(trs)


def make_spectrum_tr(hx, hy, ch, rng):
    t = ch.get(hx)[rng[0]:rng[1]]
    fs = 1/np.mean(np.diff(t))

    y = ch.get(hy)[rng[0]:rng[1]]
    f, Pxx = sps.periodogram(y, fs)

    # fig = psp.make_subplots()
    tr = go.Scatter(x=f, y=Pxx, mode='lines')
    # fig.add_trace(tr)

    return tr



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

    max_t = round(max_t, 2)

    return min_t, max_t, [min_t, max_t]


def cache_data(cols, cache, csv, fns):

    for fn in fns:
        fn_csv = csv.get(fn)
        not_cached = [k for k in cols if not cache.get(k)]
        cache[fn] = cache[fn] | {col: series for col, series in zip(
            not_cached, process_cols(not_cached, fn_csv))}
    return cache


def process_cols(cols, csv):
    headers, rows = csv

    # rows = contents.split('\r\n')

    k_cols = tuple(headers.index(y) for y in cols)

    data = [None for k in range(len(rows))]

    nan_count = [0 for k in range(len(cols))]

    # try cast columns to float
    for k, row in enumerate(rows):
        r = row

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
                    data[h][k] = r[k_cols[h]]
                except:
                    data[h][k] = ''

    return zip(*data)


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
