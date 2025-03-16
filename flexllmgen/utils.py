import os
import argparse
import functools
import dataclasses
import numpy as np




def piecewise_linear_func(xs, ys):
    """Return a function created by linear inerpolation."""
    indices = np.argsort(xs)
    xs = [xs[i] for i in indices]
    ys = [ys[i] for i in indices]

    # pad left and right
    k = 1e5
    delta_x_left = xs[0] - xs[1]
    delta_y_left = ys[0] - ys[1]
    delta_x_right = xs[-1] - xs[-2]
    delta_y_right = ys[-1] - ys[-2]

    xs = [xs[0] + delta_x_left * k] + xs + [xs[-1] + delta_x_right * k]
    ys = [ys[0] + delta_y_left * k] + ys + [ys[-1] + delta_y_right * k]

    return functools.partial(piecewise_linear_func_ret_func, xs, ys)


def piecewise_linear_func_ret_func(xs, ys, x):
    assert x >= xs[0] and x <= xs[-1]
    return np.interp(x, xs, ys)


def sample_from_range(n, k):
    assert n >= 1

    if k == -1:
        ret = [1]
        while ret[-1] * 2 < n:
            ret.append(ret[-1] * 2)
        return ret
    else:
        if k == 1: return [1]
        step = (n - 1) // (k - 1)
        return list(range(1, n + 1, step))

def run_cmd(cmd):
    print(cmd)
    os.system(cmd)


