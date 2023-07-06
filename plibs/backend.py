import plotly.graph_objects as go
import plotly.subplots as psp
import plotly.express as px

import pandas as pd
import scipy.signal as sps
import scipy as sp
import numpy as np


from dash.exceptions import PreventUpdate

import math
import io
import csv


def make_spectrum(hx, hy, ch, ranges, fns):
    trs = []
    for fn in fns:
        cache = ch.get(fn)
        rng = ranges[fn] if type(
            ranges) is dict and ranges.get(fn) else [0, None]
        trs.append(make_spectrum_tr(hx, hy, cache, rng))

    fig = psp.make_subplots()
    fig.add_traces(trs)

    return fig


def make_spectrum_tr(hx, hy, ch, rng):
    t = ch.get(hx)[rng[0]:rng[1]]
    fs = 1/np.mean(np.diff(t))

    y = ch.get(hy)[rng[0]:rng[1]]
    f, Pxx = sps.lombscargle(x=t, y=y)

    # fig = psp.make_subplots()
    tr = go.Scatter(x=f, y=Pxx, mode='lines')
    # fig.add_trace(tr)

    return tr


def get_col_val(col_idxs, r):
    '''Generator()'''
    for col in col_idxs:
        try:
            val = float(r[col])
        except:
            val = math.nan
        yield val


def trace(x, y):
    return go.Scatter(x=x, y=y)


def multiplot(*, fig,  x, y, y2=None):
    if y2 is None:
        # fig = go.Figure()
        secondary = dict()
    else:
        # fig = psp.make_subplots(specs=[[{"secondary_y": True}]])
        secondary = dict(secondary_y=True)

    if len(y) > 0 and hasattr(y[0], '__len__'):
        for y_k in y:
            fig.add_trace(trace(x, y_k), **secondary)
    else:
        fig.add_trace(trace(x, y), **secondary)

    if y2 is not None:
        if len(y2) > 0 and hasattr(y2[0], '__len__'):
            for y2_k in y2:
                fig.add_trace(trace(x, y2_k), **secondary)
        else:
            fig.add_trace(trace(x, y2), **secondary)


def make_fig(h_x, h_ys, h_ays, data, id_range, filt, fns):

    secondary = bool(len(h_ays) > 0)
    fig = go.Figure() if secondary else psp.make_subplots(
        specs=[[{"secondary_y": True}]])

    for fn in fns:
        dct = data[fn]
        x = dct[h_x]
        y = [dct[h_y] for h_y in h_ys]
        y2 = [dct[h_y2] for h_y2 in h_ays] if secondary else None

        multiplot(fig=fig, x=x, y=y, y2=y2)

    return fig


def make_fig_traces(h_x, h_ys, h_ays, cache, id_range, filt):
    h_ys = (h_ys,) if isinstance(h_ys, str) else tuple(
        h_ys) if isinstance(h_ys, list) else h_ys
    h_ays = (h_ays,) if isinstance(h_ays, str) else tuple(
        h_ays) if isinstance(h_ays, list) else h_ays

    y_c = [cache.get(hy) for hy in h_ys]

    if not (all(y_c)):
        raise PreventUpdate

    if filt[0]:
        y_c = [filter(y, filt[1]) for y in y_c]
    # x = x_c[id_range[0]:id_range[1]]
    ys = [y[id_range[0]:id_range[1]] for y in y_c]

    if h_ays is not None and len(h_ays) > 0:
        alt_y_c = [cache.get(hy) for hy in h_ays]

        if filt[0]:
            alt_y_c = [filter(y, filt[1]) for y in alt_y_c]
        alt_ys = [y[id_range[0]:id_range[1]] for y in alt_y_c]

    if len(y_c) > 0:
        x_c = cache.get(h_x) if h_x is not None else [
            k for k in range(len(y_c[0]))]
        x = x_c[id_range[0]:id_range[1]]
    elif len(alt_y_c) > 0:
        x_c = cache.get(h_x) if h_x is not None else [
            k for k in range(len(alt_y_c[0]))]
        x = x_c[id_range[0]:id_range[1]]

    second = None
    if h_ays is not None and len(h_ays) > 0:
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


def make_box(hx, hys, fns, cache):

    dfs = []
    for fn in fns:
        df = pd.DataFrame()
        data = cache.get(fn)

        df[hx] = data.get(hx)
        for hy in hys:
            df[hy] = data.get(hy)

        df['fn'] = fn
        dfs.append(df)
    plot_df = pd.concat(dfs)

    fig = px.box(plot_df, x=hx, y=list(hys), color='fn')
    return fig
