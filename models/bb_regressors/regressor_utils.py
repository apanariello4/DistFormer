from itertools import chain

import numpy as np


def flatten_bboxes(bboxes: list) -> list:
    # return [b.squeeze(0) if b.ndim > 2 else b for b in sum(bboxes, [])]
    return [b.squeeze(0) if b.ndim > 2 else b for b in chain.from_iterable(bboxes)]


def split_by_frame(x, omini_per_frame):
    c = np.cumsum([0] + omini_per_frame)
    return [x[c[i] : c[i + 1]] for i in range(len(c) - 1)]


def get_last_frame(omini, B, T):
    omini_batches = [omini[b * T : (b + 1) * T][-1] for b in range(B)]
    lens = [len(o) for o in omini_batches]

    return omini_batches, lens
