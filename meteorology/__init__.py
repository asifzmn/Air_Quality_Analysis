import math
import numpy as np
import xarray as xr

import plotly.graph_objects as go


def all_factor_correlation_matrix(corr_matrix):
    # Create correlation matrix with masked values
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix))  # For upper triangle
    # Or use this for lower triangle:
    # mask = np.tril(np.ones_like(corr_matrix))

    colors = [
        [0.0, '#708090'],  # Slate Gray for lowest values
        [0.5, '#FFFFFF'],  # White for middle values
        [1.0, '#4169E1']  # Royal Blue for highest values
    ]

    # Create annotation text matrix (only for upper triangle)
    annotations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            if mask[j, i] == 1:  # Only add annotations for upper triangle
                annotations.append(
                    dict(
                        x=corr_matrix.columns[i],
                        y=corr_matrix.columns[j],
                        text=str(round(corr_matrix.corr().iloc[j, i], 2)),
                        showarrow=False,
                        font=dict(size=18)
                    )
                )

    corr_matrix = corr_matrix * mask  # Mask the lower triangle
    # Set lower triangle to None to make it blank
    corr_matrix[mask == 0] = None

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale=colors,
        hovertemplate='%{x}<br>%{y}<br>Correlation: %{z:.2f}<extra></extra>',
        zmin=-1,
        zmax=1
    ))

    fig.update_layout(
        title=dict(
            text='Correlation Heatmap (Upper Triangle)',  # Title of the plot
            x=0.5,
            font=dict(size=24)
        ),
        annotations=annotations,
        xaxis=dict(
            tickangle=45,  # Tilt the tick labels on the x-axis
            title=dict(
                text="Variables",
                font=dict(size=18),
            ),
            automargin=True  # Automatically adjusts x-axis margins
        ),
        yaxis=dict(
            title=dict(
                text="Variables",
                font=dict(size=18),
                standoff=10  # Adds fixed padding between title and tick labels
            ),
            automargin=True  # Automatically adjusts y-axis margins
        ),
        width=900,
        height=900,
        font=dict(size=18),
        margin=dict(t=100, l=100, r=50, b=100),
        template=None
    )

    fig.update_traces(
        colorbar=dict(
            title=dict(
                text="Correlation",
                side="right",
                font=dict(size=18)
            ),
            tickfont=dict(size=18),
            len=1,
        )
    )

    fig.show()
    # fig.write_image("your_graph.pdf", format="pdf")


class MeteorologicalVariableType:
    def __init__(self, name, unit, factor_list, color_list):
        assert len(factor_list) == len(color_list)
        self.name = name
        self.unit = unit
        self.factor_list = factor_list
        self.color_list = color_list


def get_factor_data(meteo_data, factor):
    return meteo_data.sel(factor=factor).to_dataframe().drop('factor', axis=1).unstack().T.droplevel(level=0)


def get_district_data(meteo_data, district):
    return meteo_data.sel(district=district).to_dataframe().drop('district', axis=1).unstack().T.droplevel(level=0).T


def vector_calculation(x):
    if (x['Wind Speed'] == 0).all() or x.empty:
        return 0, 0
    EW_Vector = (np.sin(np.radians(x['Wind Direction'])) * x['Wind Speed']).sum()
    NS_Vector = (np.cos(np.radians(x['Wind Direction'])) * x['Wind Speed']).sum()

    EW_Average = (EW_Vector / x.shape[0]) * 1
    NS_Average = (NS_Vector / x.shape[0]) * 1

    averageWindSpeed = np.sqrt(EW_Average * EW_Average + NS_Average * NS_Average)

    Atan2Direction = math.atan2(EW_Average, NS_Average)

    averageWindDirection = np.degrees(Atan2Direction)

    # //Correction As specified in webmet.com webpage http://www.webmet.com/met_monitoring/622.html
    # if(AvgDirectionInDeg > 180):
    #     AvgDirectionInDeg = AvgDirectionInDeg - 180
    # elif (AvgDirectionInDeg < 180):
    #     AvgDirectionInDeg = AvgDirectionInDeg + 180

    if averageWindDirection < 0:
        averageWindDirection += 360

    return averageWindSpeed, averageWindDirection
