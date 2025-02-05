from typing import Union
import numpy as np
import numexpr


def place(
    foreground: np.ndarray,
    background: np.ndarray,
    x: int,
    y: int,
    mode: str = "default",
) -> np.ndarray:
    """
    Place the foreground image on the background image so that the center of the
    foreground image is at the (x, y) coordinates on the background image.
    NOTE: inplace operation --> `background` will be modified!
    :param foreground: foreground image; shape: (..., H, W)
    :param background: background image; shape: (..., H, W)
    :param x: column index on the background image
    :param y: row index on the background image
    :param mode: blanding mode; values in {'default', 'add', 'max', 'mean'}
    :return: background with foreground on it
    """

    if foreground.shape[0] not in {1, 3}:
        foreground = np.expand_dims(foreground, 0)

    squeeze = False
    if background.shape[0] not in {1, 3}:
        squeeze = True
        background = np.expand_dims(background, 0)

    assert (
        foreground.shape[1] % 2 == 1
    ), f"[Temporary Limitation] foreground H and W must be ODD; you have foreground.shape = {foreground.shape}"

    x -= 1
    y -= 1

    # height/2 of the foreground image
    ry = foreground.shape[1] // 2

    # width/2 of the foreground image
    rx = foreground.shape[2] // 2

    # limits on y-axis --> [ya, yb[
    ya = max(0, y - ry)
    yb = min(y + ry + 1, background.shape[1])

    # limits on x-axis --> [xa, xb[
    xa = max(0, x - rx)
    xb = min(x + rx + 1, background.shape[2])

    if mode == "add":
        background[..., ya:yb, xa:xb] += foreground[
            ..., ya - y + ry : yb - y + ry, xa - x + rx : xb - x + rx
        ]

    elif mode == "max":
        try:
            _patch = np.nanmax(
                [
                    background[..., ya:yb, xa:xb],
                    foreground[
                        ..., ya - y + ry : yb - y + ry, xa - x + rx : xb - x + rx
                    ],
                ],
                axis=0,
            )
            background[..., ya:yb, xa:xb] = _patch
        except Exception as e:
            print(
                f"EXCEPTION {e}:\n{x, y} for background with shape {background.shape}"
            )
            raise Exception(
                f"EXCEPTION {e}:\n{x, y} for background with shape {background.shape}"
            )

    elif mode == "mean":
        _patch = np.nanmean(
            [
                background[..., ya:yb, xa:xb],
                foreground[..., ya - y + ry : yb - y + ry, xa - x + rx : xb - x + rx],
            ],
            axis=0,
        )
        background[..., ya:yb, xa:xb] = _patch

    else:
        background[..., ya:yb, xa:xb] = foreground[
            ..., ya - y + ry : yb - y + ry, xa - x + rx : xb - x + rx
        ]

    if squeeze:
        background = np.squeeze(background, 0)
    return background


def get_hmap_patch(
    hmap_h: int, hmap_w: int, center_xy: Union[list[int], str], sigma: float = 4
) -> np.ndarray:
    """
    :param hmap_h: patch height
    :param hmap_w: patch width
    :param center_xy: gaussian center in the form (x, y) or the string 'auto'
        >> NOTE: if center=='auto', then its coordinates are set to (hmap_w//2, hmap_h//2)
    :param sigma: gaussian sigma
    :return: heatmap patch with a gaussian centered in `center`; shape (hmap_h, hmap_w)
    """
    if center_xy == "auto":
        center_xy = [hmap_w // 2, hmap_h // 2]

    if center_xy is not None:
        expression = "exp(-1 * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)"

        local_dict = {
            "x": np.arange(0, hmap_w, 1, float),
            "y": np.expand_dims(np.arange(0, hmap_h, 1, float), 1),
            "x0": center_xy[0],
            "y0": center_xy[1],
            "sigma": sigma,
        }

        return numexpr.evaluate(expression, local_dict=local_dict)

    return np.zeros(hmap_h, hmap_w)
