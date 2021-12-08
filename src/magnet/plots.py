import plotly.express as px
import plotly.graph_objects as go
from magnet.core import plot_label,plot_title


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


def waveform_visualization(st, x, y, title='Waveform Visualization', x_title='Duty in a Cycle',
                           y_title='Flux Density [mT]', color='firebrick', width=4):
    st.subheader(title)
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


def core_loss_multiple(st, x, y1, y2, title, x_title, y_title='Power Loss [kW/m^3]', x_log=True, y_log=True):
    st.subheader(title)
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
            name="ML",
            x=x,
            y=y2,
            line=dict(color='darkslategrey', width=4)
        )
    )

    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title
    )

    if x_log:
        fig.update_xaxes(type='log')
    if y_log:
        fig.update_yaxes(type='log')

    st.plotly_chart(fig, use_container_width=True)
