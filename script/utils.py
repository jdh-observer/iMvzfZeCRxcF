import plotly.graph_objects as go
from datetime import timedelta

def sec_to_tc(sec : float = 0) :
    try :
        return str(timedelta(seconds = sec))
    except Exception as e:
        print(e)
        print(sec)

def get_fig_safe(fig) :
    # Extract data from the original figure
    new_fig = go.Figure()

    for trace in fig.data:
        if trace.type == 'scattergl':
            marker_dict = trace.marker.to_plotly_json()
            
            # Use hovertext if available, else fallback to text
            hover_texts = list(getattr(trace, 'hovertext', trace.text))
            # extract the URL part
            cleaned_texts = []
            thumbnail_urls = []

            for ht in hover_texts:
                if ht and '|' in ht:
                    parts = ht.split('|')
                    # Ensure at least 3 parts
                    if len(parts) >= 3:
                        text = parts[0] + parts[2]
                        url = parts[1]
                    else:
                        text = parts[0]
                        url = ''
                else:
                    text = ht or ''
                    url = ''
                cleaned_texts.append(text)
                thumbnail_urls.append(url)


            new_trace = go.Scatter(
                x=trace.x,
                y=trace.y,
                mode=trace.mode,
                hovertext=cleaned_texts,  # assign hover text here
                marker=marker_dict,
                name=trace.name,

                hovertemplate='%{hovertext}<extra></extra>'


            )
            new_fig.add_trace(new_trace)
        else:
            new_fig.add_trace(trace)



    # Copy layout
    new_fig.update_layout(fig.layout)
    return new_fig


import plotly.graph_objects as go

def merge_bertopic_visualizations(
    fig1,
    fig2,
    label1="Model 48",
    label2="Model 78",
    legend_char_limit=None  # NEW
):
    """
    Merge two BERTopic visualization figures with a toggle button.

    Optional:
    ---------
    legend_char_limit : int or None
        Maximum number of characters for legend labels.
        If None, legend labels are left untouched.
    """

    def trim_legend(trace, max_len):
        if max_len is None:
            return trace

        if hasattr(trace, "name") and trace.name is not None:
            if len(trace.name) > max_len:
                trace.name = trace.name[: max_len - 1] + "…"

        if hasattr(trace, "legendgroup") and trace.legendgroup is not None:
            if len(trace.legendgroup) > max_len:
                trace.legendgroup = trace.legendgroup[: max_len - 1] + "…"

        return trace

    # Create a new figure
    merged_fig = go.Figure()

    # Add all traces from fig1 (initially visible)
    n_traces_fig1 = len(fig1.data)
    for trace in fig1.data:
        trace = trim_legend(trace, legend_char_limit)
        merged_fig.add_trace(trace)

    # Add all traces from fig2 (initially hidden)
    n_traces_fig2 = len(fig2.data)
    for trace in fig2.data:
        trace = trim_legend(trace, legend_char_limit)
        trace.visible = False
        merged_fig.add_trace(trace)

    # Create visibility arrays for toggle
    visible_fig1 = [True] * n_traces_fig1 + [False] * n_traces_fig2
    visible_fig2 = [False] * n_traces_fig1 + [True] * n_traces_fig2

    # Add toggle buttons
    merged_fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[{"visible": visible_fig1}],
                        label=label1,
                        method="update"
                    ),
                    dict(
                        args=[{"visible": visible_fig2}],
                        label=label2,
                        method="update"
                    )
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.11,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )
        ]
    )

    # Copy layout from first figure
    merged_fig.update_layout(
        title=fig1.layout.title,
        template=fig1.layout.template,
        width=fig1.layout.width,
        height=fig1.layout.height,
        xaxis=fig1.layout.xaxis,
        yaxis=fig1.layout.yaxis,
        shapes=fig1.layout.shapes,
        annotations=fig1.layout.annotations
    )

    return merged_fig
