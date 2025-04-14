from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns


def get_datetime():
    formatted_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return formatted_datetime


def find_smallest_uint_type(n):
    if n < 0:
        return "No unsigned type can store a negative integer."

    if n <= 15:  # 4 bits (2^4 - 1 = 15)
        return "uint4"
    elif n <= 255:  # 8 bits (2^8 - 1 = 255)
        return "uint8"
    elif n <= 65535:  # 16 bits (2^16 - 1 = 65535)
        return "uint16"
    elif n <= 4294967295:  # 32 bits (2^32 - 1 = 4294967295)
        return "uint32"
    elif n <= 18446744073709551615:  # 64 bits (2^64 - 1 = 18446744073709551615)
        return "uint64"
    else:
        return "No available type can store this integer."


def plot_error_band(
    data=None, *, x=None, y=None, delta_y=None, ax=None, alpha=0.2, color=None, **kwargs
):
    target = plt if ax is None else ax
    if data is None:
        line, _ = target.plot(x, y, **kwargs)
        target.fill_between(
            x, y - delta_y, y + delta_y, color=line.get_color(), alpha=0.2
        )
    else:
        ax = sns.lineplot(data, x=x, y=y, ax=ax, **kwargs)
        target.fill_between(
            data[x],
            data[y] - data[delta_y],
            data[y] + data[delta_y],
            color=ax.get_lines()[-1].get_color(),
            alpha=alpha,
        )
