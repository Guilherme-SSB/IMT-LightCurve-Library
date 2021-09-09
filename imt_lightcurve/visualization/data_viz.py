from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook
from bokeh.models import Legend, LegendItem

output_notebook()
# output_file('graph.html')


def line_plot(
        x_data=None,
        y_data=None,
        title='Line Plot',
        x_axis='x-axis',
        y_axis='y-axis',
        label='Data',
        y_axis_type='auto', 
        reverse=False):

    delta_x = x_data[1]-x_data[0]
    delta_y = y_data[1]-y_data[0]


    p = figure(title=title,
               y_axis_type=y_axis_type,
               plot_width=650, plot_height=400,
               background_fill_color='#fefefe',
               #x_range=(min(x_data)-2*delta_x, max(x_data)+2*delta_x),
               #y_range=(min(y_data), max(y_data))
               )
    if reverse:
            p = figure(title=title,
               y_axis_type=y_axis_type,
               plot_width=650, plot_height=400,
               background_fill_color='#fefefe',
               x_range=(min(x_data), max(x_data)),
               y_range=(max(y_data), min(y_data))
               )

    p.xaxis[0].axis_label = x_axis
    p.yaxis[0].axis_label = y_axis

    p.line(x_data, y_data, color='blue', line_width=2, legend_label=label)

    show(p)


def multi_line_plot(
        x_data=None,
        y1_data=None,
        y2_data=None,
        label_y1='y1 Data',
        label_y2='y2 Data',
        title='Multi-Line Plot',
        x_axis='x-axis',
        y_axis='y-axis',
        y_axis_type='auto'):

    p = figure(title=title,
               y_axis_type=y_axis_type,
               plot_width=650, plot_height=400,
               background_fill_color='#fefefe')

    p.xaxis[0].axis_label = x_axis
    p.yaxis[0].axis_label = y_axis

    xs = [x_data, x_data]
    ys = [y1_data, y2_data]

    r = p.multi_line(xs, ys, color=['blue', 'red'], line_width=2)

    legend = Legend(items=[
        LegendItem(label=label_y1, renderers=[r], index=0),
        LegendItem(label=label_y2, renderers=[r], index=1)
    ])

    p.add_layout(legend)

    show(p)
