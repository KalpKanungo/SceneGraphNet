import plotly.graph_objects as go
import networkx as nx


def visualize_graph(G):
    pos = nx.spring_layout(G, seed=42)

    edge_x = []
    edge_y = []
    edge_text = []

    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_text.append(data["label"])

    edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    line=dict(width=1),
    hoverinfo='none',
    mode='lines'
)

    node_x = []
    node_y = []
    node_text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(size=20)
    )

    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40)
    )
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        fig.add_annotation(
    x=x1,
    y=y1,
    ax=x0,
    ay=y0,
    xref='x',
    yref='y',
    axref='x',
    ayref='y',
    showarrow=True,
    
    arrowhead=4,        
    arrowsize=2.5,      
    arrowwidth=2.5,     
    arrowcolor="black", 

    opacity=0.9
)

    return fig
