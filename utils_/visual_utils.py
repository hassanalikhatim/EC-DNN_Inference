import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output



def export_legend(legend, filename):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename+'.pdf', dpi="figure", bbox_inches=bbox)
    return


def generate_broken_vertical_axis_plot(
    x_values, 
    first_set_of_y_values: dict, 
    second_set_of_y_values: dict
):
    
    fig, (legend_axis, additional_axis) = plt.subplots(2, 1, figsize=(3.5, 2.5))
    
    useful_metrics = {
        'first_set_min': 1000, 'first_set_max': -1000,
        'second_set_min': 1000, 'second_set_max': -1000
    }
    
    for key in first_set_of_y_values.keys():
        legend_axis.plot(
            x_values, first_set_of_y_values[key]['values'], 
            color=first_set_of_y_values[key]['color'],
            marker=first_set_of_y_values[key]['marker'],
            label=key
        )
        
        useful_metrics['first_set_min'] = min(
            useful_metrics['first_set_min'], np.min(first_set_of_y_values[key]['values'])
        )
        useful_metrics['first_set_max'] = max(
            useful_metrics['first_set_max'], np.max(first_set_of_y_values[key]['values'])
        )
    
    for key in second_set_of_y_values.keys():
        legend_axis.plot(
            x_values, second_set_of_y_values[key]['values'], 
            color=second_set_of_y_values[key]['color'],
            marker=second_set_of_y_values[key]['marker'],
            label=key
        )
        additional_axis.plot(
            x_values, second_set_of_y_values[key]['values'], 
            color=second_set_of_y_values[key]['color'],
            marker=second_set_of_y_values[key]['marker'],
            label=key
        )
        
        useful_metrics['second_set_min'] = min(
            useful_metrics['second_set_min'], np.min(second_set_of_y_values[key]['values'])
        )
        useful_metrics['second_set_max'] = max(
            useful_metrics['second_set_max'], np.max(second_set_of_y_values[key]['values'])
        )
        
    legend_axis.set_ylim( useful_metrics['first_set_min'], useful_metrics['first_set_max'] )
    additional_axis.set_ylim( useful_metrics['second_set_min'], useful_metrics['second_set_max'] )
    
    legend_axis.spines['bottom'].set_visible(False)
    additional_axis.spines['top'].set_visible(False)
    legend_axis.xaxis.tick_top()
    legend_axis.tick_params(labeltop=False)  # don't put tick labels at the top
    additional_axis.xaxis.tick_bottom()
    
    
    """
    The following code makes the diagonal coordinates
    """
    d = .02  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=legend_axis.transAxes, color='k', clip_on=False)
    legend_axis.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    legend_axis.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=additional_axis.transAxes)  # switch to the bottom axes
    additional_axis.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    additional_axis.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    
    return fig, legend_axis, additional_axis