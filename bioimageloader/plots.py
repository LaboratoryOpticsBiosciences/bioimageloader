"""Plot utilities to visualize statistics

"""
from typing import Sequence, TypeVar

T = TypeVar('T')


def cycle_colors(
    cm: Sequence[T],
    length: int
) -> Sequence[T]:
    """Cycle colors

    cm : sequence of colors
    length : int
        Length you want

    """
    len_cm = len(cm)
    if length <= len_cm:
        return cm[:length]
    colors = []
    cycle = -1
    for i in range(length):
        if i % len_cm == 0:
            cycle += 1
        if i >= len_cm:
            i -= len_cm * cycle
        colors.append(cm[i])
    return colors


def to_hex_color(x):
    """Ignore alpha"""
    if isinstance(x[0], float):
        x = [int(255*_x) for _x in x]
    return '#{:02X}{:02X}{:02X}'.format(x[0], x[1], x[2])
