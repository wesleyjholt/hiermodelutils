__all__ = ["violinplot_half"]

import numpy as np
import seaborn as sns

def violinplot_half(*args, ax, side="left", color=None, zorder=None, lw=0, alpha=1.0, **kwargs):
    """Plots a violinplot on one side of the axis.
    
    See matplotlib's violinplot documentation for more information.

    :param ax: The axis to plot on.
    :type ax: matplotlib.axes.Axes
    :param side: The side to plot on. Can be 'both', 'left', or 'right'.
    :type side: str
    :param return_axes: Whether to return the axis.
    :type return_axes: bool
    """
    if side not in ("both", "left", "right"):
        raise ValueError(f"'side' can only be 'both', 'left', or 'right', got: '{side}'")
    v = ax.violinplot(*args, **kwargs)
    if side == "both":
        return v
    for b in v["bodies"]:
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        if side == "left":
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
        else:
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
        b.set_color(color)
        b.set_zorder(zorder)
        b.set_lw(lw)
        b.set_alpha(alpha)
    return ax

# def plot_time_series_with_kde(ax, t, y):
#     n_paths = y.shape[0]
#     n_paths_plot = np.maximum(t.shape[0], 100)
#     n_steps = t.shape[0]
#     n_paths_plot = int(n_paths/100)
#     for i in np.arange(n_paths_plot):
#         ax.plot(t, y[i], lw=0.1, alpha=0.5, color='C0', zorder=2)
#     ylim = ax.get_ylim()
#     time_points_plot = [int(.05*n_steps) - 1, int(.25*n_steps) - 1, int(.5*n_steps) - 1, int(.75*n_steps) - 1, n_steps - 1]
#     v = ax.violinplot(y[:,time_points_plot], positions=t[time_points_plot], widths=n_steps/10, showmeans=False, showextrema=False, showmedians=False)
#     for b in v['bodies']:
#         # get the center
#         m = np.mean(b.get_paths()[0].vertices[:, 0])
#         # modify the paths to not go further right than the center
#         b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
#         b.set_color('r')
#         b.set_zorder(3)
#         b.set_lw(0.5)
#     ax.set_ylim(ylim)
#     return ax

# import matplotlib.collections

# # The following code is from a post by @conchoecia https://stackoverflow.com/questions/43357274/separate-halves-of-split-violinplot-to-compare-tail-data/45902085#45902085

# def add_

# def _offset_violinplot_halves(ax, delta, width, inner, direction):
#     """
#     This function offsets the halves of a violinplot to compare tails
#     or to plot something else in between them. This is specifically designed
#     for violinplots by Seaborn that use the option `split=True`.

#     For lines, this works on the assumption that Seaborn plots everything with
#      integers as the center.

#     Args:
#      <ax>    The axis that contains the violinplots.
#      <delta> The amount of space to put between the two halves of the violinplot
#      <width> The total width of the violinplot, as passed to sns.violinplot()
#      <inner> The type of inner in the seaborn
#      <direction> Orientation of violinplot. 'hotizontal' or 'vertical'.

#     Returns:
#      - NA, modifies the <ax> directly
#     """
#     # offset stuff
#     if inner == 'sticks':
#         lines = ax.get_lines()
#         for line in lines:
#             if direction == 'horizontal':
#                 data = line.get_ydata()
#                 if int(data[0] + 1)/int(data[1] + 1) < 1:
#                     # type is top, move neg, direction backwards for horizontal
#                     data -= delta
#                 else:
#                     # type is bottom, move pos, direction backward for hori
#                     data += delta
#                 line.set_ydata(data)
#             elif direction == 'vertical':
#                 data = line.get_xdata()
#                 if int(data[0] + 1)/int(data[1] + 1) < 1:
#                     # type is left, move neg
#                     data -= delta
#                 else:
#                     # type is left, move pos
#                     data += delta
#                 line.set_xdata(data)


#     for ii, item in enumerate(ax.collections):
#         # axis contains PolyCollections and PathCollections
#         if isinstance(item, matplotlib.collections.PolyCollection):
#             # get path
#             path, = item.get_paths()
#             vertices = path.vertices
#             half_type = _wedge_dir(vertices, direction)
#             # shift x-coordinates of path
#             if half_type in ['top','bottom']:
#                if inner in ["sticks", None]:
#                     if half_type == 'top': # -> up
#                         vertices[:,1] -= delta
#                     elif half_type == 'bottom': # -> down
#                         vertices[:,1] += delta
#             elif half_type in ['left', 'right']:
#                 if inner in ["sticks", None]:
#                     if half_type == 'left': # -> left
#                         vertices[:,0] -= delta
#                     elif half_type == 'right': # -> down
#                         vertices[:,0] += delta

# def _wedge_dir(vertices, direction):
#     """
#     Args:
#       <vertices>  The vertices from matplotlib.collections.PolyCollection
#       <direction> Direction must be 'horizontal' or 'vertical' according to how
#                    your plot is laid out.
#     Returns:
#       - a string in ['top', 'bottom', 'left', 'right'] that determines where the
#          half of the violinplot is relative to the center.
#     """
#     if direction == 'horizontal':
#         result = (direction, len(set(vertices[1:5,1])) == 1)
#     elif direction == 'vertical':
#         result = (direction, len(set(vertices[-3:-1,0])) == 1)
#     outcome_key = {('horizontal', True): 'bottom',
#                    ('horizontal', False): 'top',
#                    ('vertical', True): 'left',
#                    ('vertical', False): 'right'}
#     # if the first couple x/y values after the start are the same, it
#     #  is the input direction. If not, it is the opposite
#     return outcome_key[result]