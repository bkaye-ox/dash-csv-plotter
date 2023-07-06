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
            if v is math.nan or v == 0.:
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


def make_fig(h_x, h_ys, h_ays, super_cache, id_range, filt, fns):
    trs = []
    scnds = []
    for fn in fns:
        cache = super_cache.get(fn)

        rng = id_range[fn] if type(
            id_range) is dict and id_range.get(fn) else [0, None]

        trs_fn, seconds_fn = make_fig_traces(
            h_x, h_ys, h_ays, cache, rng, filt)

        trs.extend(trs_fn)
        if seconds_fn is not None:
            scnds.extend(seconds_fn)

    if h_ays is not None and len(h_ays) > 0:
        fig = psp.make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_traces(trs, secondary_ys=scnds)
    else:
        fig = psp.make_subplots()
        fig.add_traces(trs)
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

    if h_ays is not None:
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
