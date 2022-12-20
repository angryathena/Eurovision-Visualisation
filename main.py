import pandas
import pandas as pd
import panel as pn
import matplotlib.pyplot as plt
import seaborn
import plotly.graph_objects as go
from plotly.offline import iplot
import plotly.express as px


def getColours(df):
    region = ('A', 'E', 'CE', 'SE', 'C', 'S', 'W', 'N')
    rgbalist = ('209, 17, 0', '232, 97, 0', '252, 198, 0', '140, 227, 0', '52, 194, 163', '74, 80, 255', '146, 12, 242',
                '255, 89, 167')
    opacity = 0.5
    for i in range(df.__len__()):
        for j in ['GiverReg', 'Reg']:
            df.loc[i, j] = 'rgba(' + rgbalist[region.index(df.loc[i, j])] + ',' + str(opacity) + ')'
    colsG = df[['GiverReg', 'Giver']]
    colsG.drop_duplicates(subset='Giver', keep='first', inplace=True, ignore_index=True)
    colsR = df[['Reg', 'Receiver']]
    colsR.drop_duplicates(subset='Receiver', keep='first', inplace=True, ignore_index=True)
    colsR.rename(columns={'Reg': 'GiverReg', 'Receiver': "Giver"}, inplace=True)
    cols = pd.concat([colsG, colsR], ignore_index=True)
    colours = list(cols[['GiverReg']].values.ravel('K'))
    print(colours)
    return colours


def getCoordinates(df, links):
    points = df.groupby(['Receiver'], as_index=False)['Score'].sum()
    total = points['Score'].sum()
    nodeX = []
    nodeY = []
    y = 0
    for i in [0, 1]:
        n = int(len(pd.unique(links.iloc[:, i])))
        for j in range(n):
            nodeX.append(i * 0.9 + 0.09)
            pos = (j + 0.09) / n if i == 0 else y
            country = pd.unique(links.iloc[:, i])[j]
            if i == 1:
                end = int(points.loc[points.Receiver == country, 'Score'].values[0]) * 0.999 / total
                y = y + end / 2
                pos = y
                y = y + end / 2
            nodeY.append(pos)
    return [nodeX, nodeY]


def getSankey(df):
    dfCopy = df.copy(deep=True)
    colours = getColours(dfCopy)
    links = dfCopy[['Giver', 'Receiver', 'Score', 'GiverReg']]
    unique_source_target = list(pd.unique(links[['Giver', 'Receiver']].values.ravel('K')))
    nodeX, nodeY = getCoordinates(dfCopy, links)
    mapping_dict = {k: v for v, k in enumerate(unique_source_target)}
    links['Giver'] = links['Giver'].map(mapping_dict)
    links['Receiver'] = links['Receiver'].map(mapping_dict)
    links_dict = links.to_dict(orient='list')
    fig = go.Figure(data=[go.Sankey(
        arrangement="perpendicular",
        node=dict(
            label=unique_source_target, color=colours, x=nodeX, y=nodeY
        ),
        link=dict(
            source=links_dict["Giver"],
            target=links_dict["Receiver"],
            value=links_dict["Score"], color=links_dict['GiverReg']
        ))])
    fig.update_yaxes(ticklabelposition="outside")
    year = dfCopy.Year.values[0]
    title = str(year) + ' Eurovision Song Contest Final Voting Results'
    fig.update_layout(title_text=title, font_size=10, height=2000, width=1000)
    # fig.show()
    return fig


def sankeyDiagram(df):
    buttondicts = []
    for year in range(1998, 2013):
        tempDf = df.loc[df.Year == year]
        tempDf.reset_index(inplace=True)
        diagram = getSankey(tempDf)
        buttondicts.append(dict(args=[diagram], label=str(year), method="animate"))

    fig = go.Figure(diagram)
    fig.update_layout(updatemenus=[
        dict(
            type="buttons",
            direction="down",
            buttons=list(buttondicts),
            showactive=True,
            xanchor="left",
            yanchor="top"
        ),
    ])
    fig.write_html("sankey.html")
    # iplot(fig)


def mapDatasets(df):
    data = []
    wins = df.loc[df['Is.Final'] == 1, ['Year', 'Country', 'Code', 'Place']]
    wins = wins.loc[df.Place == 1, ['Year', 'Country', 'Code', 'Place']]
    wins = wins.groupby(['Country', 'Code'], as_index=False)['Place'].sum()
    wins.rename(columns={'Place': 'Value'}, inplace=True)
    data.append(wins)

    final = df.loc[df['Is.Final'] == 1, ['Year', 'Country', 'Code']]
    final = final.groupby(['Country', 'Code'], as_index=False)['Year'].count()
    final.rename(columns={'Year': 'Value'}, inplace=True)
    data.append(final)

    semifinal = df[['Year', 'Country', 'Code']]
    semifinal = semifinal.groupby(['Country', 'Code'], as_index=False)['Year'].count()
    semifinal.rename(columns={'Year': 'Value'}, inplace=True)
    data.append(semifinal)

    for d in data:
        d.reset_index(inplace=True)
    return [['Wins', 'Finals', 'Semifinals'], data]


def getChoropleth(data, title):
    print(data)
    fig = go.Figure(data=go.Choropleth(
        locations=data['Code'],
        z=data['Value'],
        text=data['Country'],
        colorscale='YlOrRd',
        autocolorscale=False,
        marker_line_color='white',
        marker_line_width=1,
    ))
    fig.update_geos(
        lataxis_range=[28.6, 71.4], lonaxis_range=[-24.53,50.34]
    )
    fig.update_layout(
        title_text='Total wins and (semi)final participations by country',height=600, width=800,
        geo=dict(
            landcolor = 'rgb(227, 250, 215)',
            showcoastlines=False,
            showcountries=True,
            projection_type='times'
        )
    )
    return fig


def choropleths(data, title):
    buttondicts = []
    for i, df in enumerate(data):
        map = getChoropleth(df, title[i])
        buttondicts.append(dict(args=[map], label=title[i], method="animate"))

    fig = go.Figure(map)
    fig.update_layout(updatemenus=[
        dict(
            type="buttons",
            direction="down",
            buttons=list(buttondicts),
            showactive=True,
            xanchor='left',
            yanchor='top',
            y=1,
            x=-0.3
        ),
    ])
    fig.write_html("map.html")
    iplot(fig)

def scatterPlots(df):
    buttons1 = []
    cols = ['Energy', 'Duration', 'Acousticness', 'Danceability', 'Tempo', 'Speechiness', 'Key', 'Liveness',
            'Time signature', 'Mode', 'Loudness', 'Valence', 'Happiness']
    for col in cols:
        buttons1.append(dict(label=col, method="update",
                             args=[{"x": [df[col].tolist()]}, {"title": ("Normalised points by " + col)}]))

    buttons2 = []
    cols = ['Language', 'Gender', 'Group or Solo']
    for col in cols:
        scatter = px.scatter(df, x=df['Energy'].tolist(),
                             y=df['Normalized Points'].tolist(), color=col,
                             color_discrete_sequence=['rgb(74, 80, 255)', 'rgb(255, 89, 167)'],
                             title="Normalised points by Energy", height=600, width=800)
        buttons2.append(dict(label=col,
                             method="animate",
                             args=[scatter]))
    fig = go.Figure(scatter)

    button_layer_1_height = 1.05
    button_layer_2_height = 0
    fig.update_layout(updatemenus=[dict(
        type="dropdown",
        direction="down",
        buttons=list(buttons1),
        showactive=True,
        xanchor='left',
        yanchor='top',
        y=0.9,
        x=-0.3
        #xanchor='left',
        #yanchor='top',
        #y=1.1,
        #x=0.25
    ),
        dict(
            type="dropdown",
            direction="down",
            buttons=list(buttons2),
            showactive=True,
            xanchor='left',
            yanchor='top',
            y=1,
            x=-0.3
            #xanchor='left',
            #yanchor='top',
           # x=0,
            #y=1.1
        ),
    ])
    fig.update_layout(plot_bgcolor='rgba(72, 24, 133, 0.07)',xaxis_title="Song Feature",
    yaxis_title="Normalised Points",legend=dict(
            xanchor='left',
            yanchor='top',
            y=0.8,
            x=-0.3
))
    iplot(fig)
    fig.write_html("scatter.html")

def scatterPlotsWinners(dfOriginal):
    df = dfOriginal.copy(deep=True)
    df = df.loc[df['Is.Final'] == 1]
    df = df.loc[df.Place == 1]
    df.reset_index(inplace=True)
    buttons1 = []
    cols = ['Energy', 'Duration', 'Acousticness', 'Danceability', 'Tempo', 'Speechiness', 'Key', 'Liveness',
            'Time signature', 'Mode', 'Loudness', 'Valence', 'Happiness']
    for col in cols:
        buttons1.append(dict(label=col, method="update",
                             args=[{"y": [df[col].tolist()]}]))

    scatter = px.scatter(df, x=df['Year'].tolist(),
                             y=df['Energy'].tolist(), color_discrete_sequence=['rgb(52, 194, 163)'],
                             title="Winning songs timeline", hover_data=['Year', 'Country', 'Artist', 'Song'],
                             height=600, width=800,labels = ['Year','Song Feature'])
    fig = go.Figure(scatter)

    button_layer_1_height = 1.05
    button_layer_2_height = 0
    fig.update_layout(updatemenus=[dict(
        type="dropdown",
        direction="down",
        buttons=list(buttons1),
        showactive=True,
        xanchor='left',
        yanchor='top',
        y=1,
        x=-0.3
    )
    ])
    fig.update_layout(plot_bgcolor='rgba(74, 80, 255, 0.07)',xaxis_title="Year",
    yaxis_title="Song Feature",)
    #fig.update_xaxes(label='Year')
    #fig.update_yaxes(visible=False)
    iplot(fig)
    fig.write_html("scatterWins.html")

def main():
    df = pandas.read_csv('allvotes.csv', header=0)

    # sankeyDiagram(df)
    df = pandas.read_csv('songs.csv', header=0)
    titles, data = mapDatasets(df)
    choropleths(data,titles)
    scatterPlots(df)
    scatterPlotsWinners(df)



main()
