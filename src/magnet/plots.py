import plotly.express as px
import plotly.graph_objects as go
from magnet.core import plot_label, plot_title
from plotly.subplots import make_subplots


import pandas as pd
from plotly.graph_objs.layout import YAxis, XAxis


def scatter_plot(df, x='Frequency_kHz', y='Power_Loss_kW/m3', c='Flux_Density_mT'):
    return px.scatter(
        df,
        x=df[x],
        y=df[y],
        color=df[c],
        log_x=True,
        log_y=True,
        color_continuous_scale=px.colors.sequential.Turbo,
        labels={
            x: f' {plot_label(x)}',
            y: f' {plot_label(y)}',
            c: f' {plot_label(c)}'
        },
        title=f' {plot_title(c)} vs {plot_title(x)} and {plot_title(y)}',
    )


# From https://stackoverflow.com/questions/67589451/how-to-add-secondary-xaxis-in-plotly-using-plotly-express
def waveform_visualization_2axes(st, x1, x2, y1, y2, x1_aux, y1_aux, title='Waveform Visualization',
                                 x1_title='Time [us]', x2_title='Fraction of the cycle',
                                 y1_title='Flux Density [mT]', y2_title='Normalized Voltage', width=4):
    fig = go.Figure()
    fig.layout = go.Layout(dict(
        xaxis1=XAxis(dict(
            overlaying='x',
            side='bottom',
            title_text=x1_title
        )),
        yaxis1=YAxis(dict(
            overlaying='y',
            side='left',
            title_text=y1_title,
            color='mediumslateblue'
        )),
        xaxis2=XAxis(dict(
            overlaying='x',
            side='top',
            title_text=x2_title
        )),
        yaxis2=YAxis(dict(
            overlaying='y',
            side='right',
            title_text=y2_title,
            color='firebrick',
            range=[-1.1, 1.1]
        )),
        title=title,
        legend=dict(
            yanchor="bottom",
            y=1.08,
            xanchor="right",
            x=1,
        )
    ))
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 0],
            xaxis='x2',
            yaxis='y2',
            line=dict(color='black', dash='longdash', width=2),
            marker=dict(opacity=0),
            showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x2,
            y=y2,
            xaxis='x2',
            yaxis='y2',
            line=dict(color='firebrick', width=width),
            name=y2_title
        )
    )
    # Adding this plot won't work
    fig.add_trace(
        go.Scatter(
            x=x1,
            y=y1,
            line=dict(color='mediumslateblue', width=width),
            name=y1_title,
            showlegend=True
        )
    )
    # The only workaround I could find is to refer the first plot to the other axes
    # So we need to translate x1 and y1 variables to the x2 and y2 axes,
    # this is what x1_aux and y1_aux are for.
    # TODO: find a nice way to implement this code instead (valid with B bias too)
    fig.add_trace(
        go.Scatter(
            x=x1_aux,
            y=y1_aux,
            xaxis='x2',
            yaxis='y2',
            line=dict(color='mediumslateblue', width=width),
            showlegend=False
        )
    )
    st.plotly_chart(fig, use_container_width=True)


def waveform_visualization(st, x, y, x_title='Duty in a Cycle', y_title=None, color='firebrick', width=4):
  
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            line=dict(color=color, width=width)
        )
    )
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title
    )

    st.plotly_chart(fig, use_container_width=True)


def core_loss_multiple(st, x, y1, y2, x0, y01, y02, title, x_title, y_title='Power Loss [kW/m^3]', x_log=True,
                       y_log=True):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="iGSE",
            x=x,
            y=y1,
            line=dict(color='firebrick', width=4)
        )
    )
    fig.add_trace(
        go.Scatter(
            marker_symbol="diamond",
            marker_size=13,
            name="iGSE",
            x=x0,
            y=y01,
            line=dict(color='firebrick', width=4)
        )
    )
    fig.add_trace(
        go.Scatter(
            name="ML",
            x=x,
            y=y2,
            line=dict(color='darkslategrey', width=4)
        )
    )
    fig.add_trace(
        go.Scatter(
            marker_symbol="diamond",
            marker_size=13,
            name="ML",
            x=x0,
            y=y02,
            line=dict(color='darkslategrey', width=4)
        )
    )
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title,
        title=title,
        title_x=0.5
    )

    if x_log:
        fig.update_xaxes(type='log')
    if y_log:
        fig.update_yaxes(type='log')

    st.plotly_chart(fig, use_container_width=True)
