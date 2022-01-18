import plotly.express as px
import plotly.graph_objects as go
from magnet.core import plot_label, plot_title
import numpy as np

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
        title=f'<b>{plot_title(c)} vs {plot_title(x)} and {plot_title(y)}</b>',
    )


# From https://stackoverflow.com/questions/67589451/how-to-add-secondary-xaxis-in-plotly-using-plotly-express
def waveform_visualization_2axes(
        st, x1, x2, y1, y2, x1_aux, y1_aux,
        title='Waveform Visualization', x1_title='Time [us]', x2_title='Fraction of the cycle',
        y1_title='Flux Density [mT]', y2_title='Normalized Voltage', width=4):
    fig = go.Figure()
    fig.layout = go.Layout(dict(
        xaxis1=XAxis(dict(
            overlaying='x',
            side='bottom',
            title_text=x1_title,
            range=[0, np.amax(x1)]
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
            title_text=x2_title,
            range=[0, 1]
        )),
        yaxis2=YAxis(dict(
            overlaying='y',
            side='right',
            title_text=y2_title,
            color='firebrick',
            range=[-1.1, 1.1]
        )),
        title=title,
        legend=dict(yanchor="bottom", y=0, xanchor="right", x=1)

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


def waveform_visualization(
        st, x, y, x_title='Fraction of the cycle', y_title='Flux Density [mT]', color='mediumslateblue', width=4):
  
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


def core_loss_multiple(
        st, x, y1, y2, x0, y01, y02, title, x_title, y_title='Power Loss [kW/m^3]', x_log=True, y_log=True, y3=None, y03=None, y4=None, y04=None):
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
        go.Scatter(dict(
            name="iGSE",
            marker_symbol="diamond",
            marker_size=13,
            showlegend=False,
            x=x0,
            y=y01,
            line=dict(color='firebrick', width=4)
        ))
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
        go.Scatter(dict(
            name="ML",
            marker_symbol="diamond",
            marker_size=13,
            showlegend=False,
            x=x0,
            y=y02,
            line=dict(color='darkslategrey', width=4)
        ))
    )
    if (y3 is not None) and max(y3) > 0.0:
        fig.add_trace(
            go.Scatter(
                name="Datasheet",
                x=x,
                y=y3,
                line=dict(color='mediumslateblue', width=0)
            )
        )
    if y03 is not None:
        fig.add_trace(
            go.Scatter(dict(
                name="Datasheet",
                marker_symbol="diamond",
                marker_size=13,
                showlegend=False,
                x=x0,
                y=y03,
                line=dict(color='mediumslateblue', width=4)
            ))
        )
    if (y4 is not None) and max(y4) > 0.0:
        fig.add_trace(
            go.Scatter(
                name="Measurement",
                x=x,
                y=y4,
                line=dict(color='limegreen', width=0)
            )
        )
    if y04 is not None:
        fig.add_trace(
            go.Scatter(dict(
                name="Measurement",
                marker_symbol="diamond",
                marker_size=13,
                showlegend=False,
                x=x0,
                y=y04,
                line=dict(color='limegreen', width=4)
            ))
        )

    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title,
        title=title,
        title_x=0.5,
        legend=dict(yanchor="bottom", y=0, xanchor="right", x=1)
    )
    if x_log:
        fig.update_xaxes(type='log')
    if y_log:
        fig.update_yaxes(type='log')

    st.plotly_chart(fig, use_container_width=True)


# Points for the representation of the plots
def cycle_points_sinusoidal(point):
    cycle_list = np.linspace(0, 1, point)
    flux_list = np.sin(np.multiply(cycle_list, np.pi * 2))
    volt_list = np.cos(np.multiply(cycle_list, np.pi * 2))
    return [cycle_list, flux_list, volt_list]


def cycle_points_trapezoidal(duty_p, duty_n, duty_0):
    if duty_p > duty_n:
        volt_p = (1 - (duty_p - duty_n)) / -(-1 - (duty_p - duty_n))
        volt_0 = - (duty_p - duty_n) / -(-1 - (duty_p - duty_n))
        volt_n = -1  # The negative voltage is maximum
        b_p = 1  # Bpk is proportional to the voltage, which is is proportional to (1-dp+dN) times the dp
        b_n = -(-1 - duty_p + duty_n) * duty_n / ((1 - duty_p + duty_n) * duty_p)  # Prop to (-1-dp+dN)*dn
    else:
        volt_p = 1  # The positive voltage is maximum
        volt_0 = - (duty_p - duty_n) / (1 - (duty_p - duty_n))
        volt_n = (-1 - (duty_p - duty_n)) / (1 - (duty_p - duty_n))
        b_n = 1  # Proportional to (-1-dP+dN)*dN
        b_p = -(1 - duty_p + duty_n) * duty_p / ((-1 - duty_p + duty_n) * duty_n)  # Prop to (1-dP+dN)*dP
    cycle_list = [0, 0, duty_p, duty_p, duty_p + duty_0, duty_p + duty_0, 1 - duty_0, 1 - duty_0, 1]
    flux_list = [-b_p, -b_p, b_p, b_p, b_n, b_n, -b_n, -b_n, -b_p]
    volt_list = [volt_0, volt_p, volt_p, volt_0, volt_0, volt_n, volt_n, volt_0, volt_0]
    return [cycle_list, flux_list, volt_list]