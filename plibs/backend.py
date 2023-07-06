import plotly.graph_objects as go
import plotly.subplots as psp
import plotly.express as px

import pandas as pd
import scipy.signal as sps
import scipy as sp
import numpy as np

import math
import base64
import io


def format_cols(col_x, cols_y, cols_y2):
    if cols_y is None:
        cols_y = ()
    if cols_y2 is None:
        cols_y2 = ()

    def format(cols):
        cols = (cols,) if isinstance(cols, str) else tuple(
            cols) if isinstance(cols, list) else cols

    return col_x, format(cols_y), format(cols_y2)


def extract_data(fn, upload_contents):
    content_type, content_b64 = upload_contents.split(',')

    df = None

    if '.csv' in fn:
        content_str = base64.b64decode(content_b64).decode()
        df = pd.read_csv(io.StringIO(content_str))
    elif '.feather' in fn:
        content = base64.b64decode(content_b64)
        df = pd.read_feather(io.BytesIO(content))

    return df.to_dict(orient='list') if df is not None else df


def make_spectrum(col_x, col_y, df, ranges, fns):
    trs = []
    for fn in fns:
        cache = df.get(fn)
        rng = ranges[fn] if type(
            ranges) is dict and ranges.get(fn) else [0, None]
        trs.append(make_spectrum_tr(col_x, col_y, cache, rng))

    fig = psp.make_subplots()
    fig.add_traces(trs)

    return fig


def make_spectrum_tr(col_x, col_y, df, rng):
    t = df.get(col_x)[rng[0]:rng[1]]
    fs = 1/np.mean(np.diff(t))

    y = df.get(col_y)[rng[0]:rng[1]]
    f, Pxx = sps.periodogram(x=y, fs=fs)

    tr = go.Scatter(x=f, y=Pxx, mode='lines')

    return tr


def get_col_val(col_idxs, r):
    '''Generator()'''
    for col in col_idxs:
        try:
            val = float(r[col])
        except:
            val = math.nan
        yield val


def multiplot(*, fig,  x, y, y2=None):
    '''
    Adds data to an existing figure

    fig: plotly.Figure
    x: x data as iterable
    y: list of y data
    y2: <optional> list of y data for 2nd axis
    '''
    if y2 is None:
        left_arg = dict()
    else:
        left_arg = dict(secondary_y=False)
        right_arg = dict(secondary_y=True)

    def trace(x, y):
        return go.Scatter(x=x, y=y)

    def add_traces(x, y_list, args):
        if len(y_list) > 0 and hasattr(y_list[0], '__len__'):
            for y_k in y:
                fig.add_trace(trace(x, y_k), **args)
        else:
            fig.add_trace(trace(x, y), **args)

    add_traces(x, y, left_arg)
    if y2 is not None:
        add_traces(x, y2, right_arg)


def make_fig(col_x, cols_y, cols_y2, data, id_range, filt, fns):

    secondary = bool(len(cols_y2) > 0)
    fig = go.Figure() if not secondary else psp.make_subplots(
        specs=[[{"secondary_y": True}]])

    for fn in fns:
        dct = data[fn]
        x = dct[col_x]
        y = [dct[col] for col in cols_y]
        y2 = [dct[col] for col in cols_y2] if secondary else None

        multiplot(fig=fig, x=x, y=y, y2=y2)

    return fig


def filter(series, window):

    nan_as_zeros = tuple(0 if math.isnan(yk) else yk for yk in series)

    return sp.ndimage.uniform_filter1d(
        nan_as_zeros, window, mode='constant', cval=0)


def make_box(col_x, cols_y, fns, data_dict):

    dfs = []
    for fn in fns:
        df = pd.DataFrame()
        data = data_dict.get(fn)

        df[col_x] = data.get(col_x)
        for col_y in cols_y:
            df[col_y] = data.get(col_y)

        df['fn'] = fn
        dfs.append(df)
    plot_df = pd.concat(dfs)

    fig = px.box(plot_df, x=col_x, y=list(cols_y), color='fn')
    return fig
