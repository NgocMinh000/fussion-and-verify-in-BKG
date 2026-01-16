"""
Interactive Dashboard for FuseLinker Visualization

Run this file to launch the interactive web dashboard:
    python -m visualization.app

Or from fuselinker directory:
    python visualization/app.py
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import json
import numpy as np
from pathlib import Path
import argparse

from .config import VIZ_CONFIG, VIZ_OUTPUT_DIR, RELATION_TYPES, SEMANTIC_TYPES
from .graph_visualizer import GraphVisualizer
from .embedding_visualizer import EmbeddingVisualizer


# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True
)

app.title = "FuseLinker Visualization"

# Global variables for data
GRAPH_VIZ = None
EMBED_VIZ = None
DATA_DIR = None


def load_data(data_dir: str):
    """Load visualization data from directory"""
    global GRAPH_VIZ, EMBED_VIZ, DATA_DIR
    DATA_DIR = Path(data_dir)

    print(f"Loading data from {DATA_DIR}...")

    # Load graph
    graph_path = DATA_DIR / "train_graph.json"
    if graph_path.exists():
        GRAPH_VIZ = GraphVisualizer()
        GRAPH_VIZ.load_from_json(graph_path)
        print(f"✓ Loaded graph: {len(GRAPH_VIZ.nodes_data)} nodes, {len(GRAPH_VIZ.edges_data)} edges")
    else:
        print(f"Warning: Graph file not found at {graph_path}")

    # Load embeddings
    embed_path = DATA_DIR / "node_embeddings.npy"
    if embed_path.exists():
        EMBED_VIZ = EmbeddingVisualizer()

        # Load entity mapping
        meta_path = DATA_DIR / "node_embeddings.meta.json"
        entity_mapping = None
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                entity_mapping = {int(k): v for k, v in meta.get('entity_mapping', {}).items()}

        EMBED_VIZ.load_embeddings(embed_path, entity_mapping)
        print(f"✓ Loaded embeddings: {EMBED_VIZ.embeddings.shape}")
    else:
        print(f"Warning: Embeddings file not found at {embed_path}")


# Layout components
def create_header():
    """Create dashboard header"""
    return dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H2("🧬 FuseLinker Visualization Dashboard",
                           className="text-light mb-0")
                ], width=8),
                dbc.Col([
                    html.P(f"Data: {DATA_DIR.name if DATA_DIR else 'Not loaded'}",
                          className="text-muted mb-0 text-end")
                ], width=4)
            ], align="center", className="w-100")
        ], fluid=True),
        color="dark",
        dark=True,
        className="mb-4"
    )


def create_filter_panel():
    """Create filter controls panel"""
    return dbc.Card([
        dbc.CardHeader(html.H5("🔍 Filters", className="mb-0")),
        dbc.CardBody([
            # Semantic type filter
            html.Label("Semantic Types:", className="fw-bold mt-2"),
            dcc.Dropdown(
                id='semantic-type-filter',
                options=[{'label': f"{k} - {v}", 'value': k}
                        for k, v in SEMANTIC_TYPES.items()],
                multi=True,
                placeholder="Select semantic types...",
                className="mb-3"
            ),

            # Relation type filter
            html.Label("Relations:", className="fw-bold mt-2"),
            dcc.Dropdown(
                id='relation-filter',
                options=[{'label': rel, 'value': rel} for rel in RELATION_TYPES],
                multi=True,
                placeholder="Select relations...",
                className="mb-3"
            ),

            # Max nodes slider
            html.Label("Max Nodes to Display:", className="fw-bold mt-2"),
            dcc.Slider(
                id='max-nodes-slider',
                min=100,
                max=2000,
                step=100,
                value=500,
                marks={i: str(i) for i in range(100, 2001, 500)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),

            html.Hr(),

            # Apply filters button
            dbc.Button(
                "Apply Filters",
                id='apply-filters-btn',
                color="primary",
                className="w-100 mt-3"
            ),

            # Reset button
            dbc.Button(
                "Reset Filters",
                id='reset-filters-btn',
                color="secondary",
                className="w-100 mt-2"
            ),
        ])
    ], className="h-100")


def create_stats_panel():
    """Create statistics panel"""
    return dbc.Card([
        dbc.CardHeader(html.H5("📊 Statistics", className="mb-0")),
        dbc.CardBody([
            html.Div(id='stats-content', children=[
                html.P("Apply filters to see statistics", className="text-muted")
            ])
        ])
    ], className="h-100")


# Main layout
app.layout = dbc.Container([
    create_header(),

    dbc.Row([
        # Left sidebar - Filters
        dbc.Col([
            create_filter_panel(),
            html.Br(),
            create_stats_panel()
        ], width=3),

        # Main content area
        dbc.Col([
            dbc.Tabs([
                # Graph visualization tab
                dbc.Tab(
                    label="📊 Graph Structure",
                    tab_id="tab-graph",
                    children=[
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Loading(
                                    id="loading-graph",
                                    type="default",
                                    children=html.Div(id='graph-container')
                                )
                            ])
                        ], className="mt-3")
                    ]
                ),

                # Embedding visualization tab
                dbc.Tab(
                    label="🎯 Embedding Space",
                    tab_id="tab-embedding",
                    children=[
                        dbc.Card([
                            dbc.CardBody([
                                # Embedding controls
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Reduction Method:"),
                                        dcc.Dropdown(
                                            id='reduction-method',
                                            options=[
                                                {'label': 'UMAP', 'value': 'umap'},
                                                {'label': 't-SNE', 'value': 'tsne'},
                                                {'label': 'PCA', 'value': 'pca'}
                                            ],
                                            value='umap'
                                        )
                                    ], width=3),
                                    dbc.Col([
                                        html.Label("Dimensions:"),
                                        dcc.Dropdown(
                                            id='n-components',
                                            options=[
                                                {'label': '2D', 'value': 2},
                                                {'label': '3D', 'value': 3}
                                            ],
                                            value=2
                                        )
                                    ], width=3),
                                    dbc.Col([
                                        html.Label("Color By:"),
                                        dcc.Dropdown(
                                            id='color-by',
                                            options=[
                                                {'label': 'Semantic Type', 'value': 'semantic_type'},
                                                {'label': 'Cluster', 'value': 'cluster'}
                                            ],
                                            value='semantic_type'
                                        )
                                    ], width=3),
                                    dbc.Col([
                                        dbc.Button(
                                            "Compute",
                                            id='compute-embedding-btn',
                                            color="success",
                                            className="mt-4"
                                        )
                                    ], width=3)
                                ], className="mb-3"),

                                dcc.Loading(
                                    id="loading-embedding",
                                    type="default",
                                    children=html.Div(id='embedding-container')
                                )
                            ])
                        ], className="mt-3")
                    ]
                ),

                # About tab
                dbc.Tab(
                    label="ℹ️ About",
                    tab_id="tab-about",
                    children=[
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("About FuseLinker Visualization"),
                                html.Hr(),
                                html.P([
                                    "This dashboard visualizes the knowledge graph and embeddings ",
                                    "generated by the FuseLinker model."
                                ]),
                                html.H5("Features:"),
                                html.Ul([
                                    html.Li("Interactive graph visualization with PyVis"),
                                    html.Li("Embedding space visualization with UMAP/t-SNE"),
                                    html.Li("Filter by semantic types and relations"),
                                    html.Li("Statistics and analytics"),
                                ]),
                                html.H5("Usage:"),
                                html.Ol([
                                    html.Li("Use filters in the left sidebar to focus on specific entities/relations"),
                                    html.Li("Click 'Apply Filters' to update visualizations"),
                                    html.Li("Switch between tabs to explore different views"),
                                    html.Li("Hover over nodes/points to see details"),
                                ]),
                                html.Hr(),
                                html.P([
                                    html.Strong("Data Directory: "),
                                    html.Code(str(DATA_DIR) if DATA_DIR else "Not loaded")
                                ])
                            ])
                        ], className="mt-3")
                    ]
                ),
            ], id="tabs", active_tab="tab-graph")
        ], width=9)
    ], className="g-3")
], fluid=True, className="p-4")


# Callbacks
@app.callback(
    [Output('graph-container', 'children'),
     Output('stats-content', 'children')],
    [Input('apply-filters-btn', 'n_clicks')],
    [State('semantic-type-filter', 'value'),
     State('relation-filter', 'value'),
     State('max-nodes-slider', 'value')]
)
def update_graph(n_clicks, semantic_types, relations, max_nodes):
    """Update graph visualization based on filters"""
    if GRAPH_VIZ is None:
        return html.P("No graph data loaded", className="text-danger"), html.P("N/A")

    # Create a copy for filtering
    viz = GraphVisualizer()
    viz.nodes_data = GRAPH_VIZ.nodes_data.copy()
    viz.edges_data = GRAPH_VIZ.edges_data.copy()

    # Apply filters
    if semantic_types:
        viz.filter_by_semantic_types(semantic_types)

    if relations:
        viz.filter_by_relations(relations)

    # Sample if too many nodes
    if len(viz.nodes_data) > max_nodes:
        viz.sample_nodes(max_nodes, strategy='degree')

    # Create visualization
    output_path = VIZ_OUTPUT_DIR / "temp_graph.html"
    viz.create_pyvis_network(physics_enabled=True, layout='force_directed')
    viz.render(output_path)

    # Get statistics
    stats = viz.get_statistics()

    # Create stats display
    stats_content = [
        html.H6("Current View:"),
        html.P([
            html.Strong("Nodes: "), f"{stats['num_nodes']}", html.Br(),
            html.Strong("Edges: "), f"{stats['num_edges']}", html.Br(),
            html.Strong("Density: "), f"{stats['density']:.4f}", html.Br(),
            html.Strong("Avg Degree: "), f"{stats['avg_degree']:.2f}", html.Br(),
        ]),
        html.Hr(),
        html.H6("Semantic Types:"),
        html.Ul([
            html.Li(f"{stype}: {count}")
            for stype, count in list(stats['semantic_type_distribution'].items())[:5]
        ]),
    ]

    # Return iframe with graph
    graph_html = html.Iframe(
        src=f"/assets/../../../{output_path.relative_to(VIZ_OUTPUT_DIR.parent)}",
        style={'width': '100%', 'height': '700px', 'border': 'none'}
    )

    return graph_html, stats_content


@app.callback(
    Output('embedding-container', 'children'),
    [Input('compute-embedding-btn', 'n_clicks')],
    [State('reduction-method', 'value'),
     State('n-components', 'value'),
     State('color-by', 'value')]
)
def update_embedding(n_clicks, method, n_components, color_by):
    """Update embedding visualization"""
    if EMBED_VIZ is None or n_clicks is None:
        return html.P("No embedding data loaded. Click 'Compute' to generate visualization.",
                     className="text-muted")

    # Reduce dimensions
    EMBED_VIZ.reduce_dimensions(method=method, n_components=n_components)

    # Cluster if needed
    if color_by == 'cluster' and EMBED_VIZ.clusters is None:
        EMBED_VIZ.cluster_embeddings(n_clusters=8)

    # Create plot
    fig = EMBED_VIZ.create_scatter_plot(color_by=color_by,
                                        title=f"Embedding Space ({method.upper()})")

    return dcc.Graph(figure=fig, style={'height': '700px'})


@app.callback(
    [Output('semantic-type-filter', 'value'),
     Output('relation-filter', 'value'),
     Output('max-nodes-slider', 'value')],
    [Input('reset-filters-btn', 'n_clicks')]
)
def reset_filters(n_clicks):
    """Reset all filters to default"""
    return None, None, 500


def main(data_dir: str, host: str = '0.0.0.0', port: int = 8050, debug: bool = True):
    """
    Run the dashboard application

    Args:
        data_dir: Directory containing visualization data
        host: Host address
        port: Port number
        debug: Debug mode
    """
    # Load data
    load_data(data_dir)

    # Run app
    print("\n" + "=" * 60)
    print("FuseLinker Visualization Dashboard")
    print("=" * 60)
    print(f"Starting server at http://{host}:{port}")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FuseLinker Visualization Dashboard')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing visualization data')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host address (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port number (default: 5000)')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')

    args = parser.parse_args()

    main(args.data_dir, args.host, args.port, args.debug)
