"""
Interactive Prediction Explorer Dashboard.
Explore link predictions, explanations, and model reasoning.

Requirements: streamlit, plotly
Install: pip install streamlit plotly

Run: streamlit run visualization/prediction_app.py
"""

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    print("ERROR: streamlit and plotly required")
    print("Install: pip install streamlit plotly")
    exit(1)

import json
import numpy as np
from pathlib import Path


@st.cache_data
def load_predictions(prediction_file):
    """Load predictions."""
    with open(prediction_file, 'r') as f:
        return json.load(f)


@st.cache_data
def load_explanations(explanation_file):
    """Load explanations if available."""
    if Path(explanation_file).exists():
        with open(explanation_file, 'r') as f:
            return json.load(f)
    return None


def show_prediction_statistics(pred_data):
    """Display overall prediction statistics."""
    st.header("📊 Prediction Statistics")

    stats = pred_data['statistics']

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Queries", stats['total_queries'])
        st.metric("Hits@1", f"{stats['hits_at_1']:.4f}")

    with col2:
        st.metric("Hits@3", f"{stats['hits_at_3']:.4f}")
        st.metric("Hits@10", f"{stats['hits_at_10']:.4f}")

    with col3:
        st.metric("MRR", f"{stats['mrr']:.4f}")
        st.metric("Avg True Rank", f"{stats['average_true_rank']:.2f}")

    # Rank distribution
    st.subheader("Rank Distribution")

    ranks = []
    for result in pred_data['predictions']:
        rank = result['query']['true_object_rank']
        if rank is not None:
            ranks.append(rank)

    # Create histogram
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=ranks,
        nbinsx=20,
        marker_color='#3498db',
        opacity=0.7
    ))

    fig.update_layout(
        title='Distribution of True Answer Ranks',
        xaxis_title='Rank',
        yaxis_title='Count',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def show_query_explorer(pred_data, explanations):
    """Interactive query prediction explorer."""
    st.header("🔍 Query Prediction Explorer")

    predictions = pred_data['predictions']

    # Query selector
    query_idx = st.selectbox(
        "Select Query",
        range(len(predictions)),
        format_func=lambda i: f"Query {i+1}: ({predictions[i]['query']['subject']['name']}, "
                              f"{predictions[i]['query']['relation']['name']}, ?)"
    )

    result = predictions[query_idx]
    query = result['query']
    preds = result['predictions']

    st.markdown("---")

    # Query info
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Query Triple")
        st.markdown(f"**Subject:** {query['subject']['name']}")
        st.markdown(f"**Relation:** {query['relation']['name']}")
        st.markdown(f"**Object:** ?")

    with col2:
        st.markdown("### True Answer")
        st.markdown(f"**Entity:** {query['true_object']['name']}")
        st.markdown(f"**Rank:** {query['true_object_rank']}")
        st.markdown(f"**Score:** {query['true_object_score']:.4f}")

        if query['true_object_rank'] == 1:
            st.success("✓ Perfect prediction!")
        elif query['true_object_rank'] <= 3:
            st.info("✓ Top-3 prediction")
        elif query['true_object_rank'] <= 10:
            st.warning("✓ Top-10 prediction")
        else:
            st.error("✗ Outside Top-10")

    st.markdown("---")

    # Predictions table
    st.subheader("Top Predictions")

    # Show as bar chart
    top_k = st.slider("Show Top-K", 5, 20, 10)

    entities = [p['entity_name'][:50] for p in preds[:top_k]]
    scores = [p['score'] for p in preds[:top_k]]
    is_correct = [p['is_correct'] for p in preds[:top_k]]

    colors = ['#2ecc71' if correct else '#3498db' for correct in is_correct]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[f"#{i+1} {entities[i]}" for i in range(len(entities))],
        x=scores,
        orientation='h',
        marker_color=colors,
        text=[f'{s:.3f}' for s in scores],
        textposition='outside'
    ))

    fig.update_layout(
        title='Prediction Scores',
        xaxis_title='Score',
        yaxis_title='Entity',
        height=max(400, len(entities) * 30),
        yaxis={'autorange': 'reversed'}
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detailed predictions table
    with st.expander("📋 Detailed Predictions Table"):
        pred_table = []
        for p in preds[:top_k]:
            pred_table.append({
                'Rank': p['rank'],
                'Entity': p['entity_name'],
                'Score': f"{p['score']:.4f}",
                'Correct': '✓' if p['is_correct'] else ''
            })

        st.table(pred_table)

    # Show explanation if available
    if explanations and query_idx < len(explanations):
        st.markdown("---")
        show_explanation(explanations[query_idx])


def show_explanation(explanation):
    """Display explanation for a prediction."""
    st.header("💡 Explanation: Why This Prediction?")

    query = explanation['query']

    st.markdown(f"**Query:** ({query['subject']['name']}, {query['relation']['name']}, ?)")
    st.markdown(f"**True Answer:** {query['true_object']['name']}")

    tabs = st.tabs([
        "🧮 Embedding Similarity",
        "📚 Training Patterns",
        "👥 Nearest Neighbors",
        "🔄 Similar Queries"
    ])

    # Tab 1: Embedding Similarity
    with tabs[0]:
        st.subheader("Embedding Similarity Analysis")
        st.markdown("Cosine similarities between embeddings explain why predictions score high.")

        emb_analysis = explanation['embedding_analysis']

        if emb_analysis:
            # Similarity heatmap
            entities = [e['entity_name'][:30] for e in emb_analysis]
            similarity_types = ['subject_object', 'subject_relation', 'relation_object', 'combined_alignment']
            similarity_labels = ['Subject↔Object', 'Subject↔Relation', 'Relation↔Object', 'Combined']

            similarity_matrix = []
            for e in emb_analysis:
                row = [e['similarities'][st] for st in similarity_types]
                similarity_matrix.append(row)

            similarity_matrix = np.array(similarity_matrix).T

            fig = go.Figure(data=go.Heatmap(
                z=similarity_matrix,
                x=[f"#{i+1} {entities[i]}" for i in range(len(entities))],
                y=similarity_labels,
                colorscale='RdYlGn',
                zmid=0,
                text=similarity_matrix,
                texttemplate='%{text:.2f}',
                textfont={"size": 10}
            ))

            fig.update_layout(
                title='Embedding Similarities',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Reasoning
            st.markdown("**Reasoning:**")
            for e in emb_analysis[:5]:
                with st.expander(f"Rank {e['rank']}: {e['entity_name']} (score: {e['score']:.3f})"):
                    for reason in e['reasoning']:
                        st.markdown(f"• {reason}")

    # Tab 2: Training Patterns
    with tabs[1]:
        st.subheader("Training Data Patterns")
        st.markdown("What patterns did the model learn from training data?")

        training_stats = explanation['training_stats']

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Subject with this Relation",
                f"{training_stats['subject_relation_count']} times"
            )

        with col2:
            st.metric(
                "This Relation (all subjects)",
                f"{training_stats['relation_total_count']} times"
            )

        st.markdown("**Most Common Objects for This Relation:**")
        for obj in training_stats['most_common_objects']:
            st.markdown(f"• {obj['entity_name']}: {obj['count']} times")

        st.markdown("---")
        st.markdown("**Prediction Patterns:**")

        pattern_explanations = explanation['training_patterns']
        for expl in pattern_explanations:
            with st.expander(f"Rank {expl['rank']}: {expl['entity_name']}"):
                for pattern in expl['patterns']:
                    st.markdown(f"• {pattern}")

    # Tab 3: Nearest Neighbors
    with tabs[2]:
        st.subheader("Nearest Neighbors in Embedding Space")
        st.markdown(f"Entities most similar to **{query['subject']['name']}**:")

        neighbors = explanation['nearest_neighbors']

        neighbor_data = []
        for i, neighbor in enumerate(neighbors, 1):
            neighbor_data.append({
                'Rank': i,
                'Entity': neighbor['entity_name'],
                'Similarity': f"{neighbor['similarity']:.3f}"
            })

        st.table(neighbor_data)

    # Tab 4: Similar Queries
    with tabs[3]:
        st.subheader("Similar Queries in Training Data")
        st.markdown("The model may generalize from these similar patterns:")

        similar = explanation['similar_queries']

        for i, sim_query in enumerate(similar, 1):
            st.markdown(
                f"{i}. **({sim_query['subject']}, {sim_query['relation']}, {sim_query['object']})**  "
                f"_Similarity: {sim_query['similarity']:.3f}_"
            )


def show_comparison(pred_data):
    """Compare predictions across multiple queries."""
    st.header("📈 Prediction Comparison")

    predictions = pred_data['predictions']

    # Score distribution
    st.subheader("Score Distribution")

    all_scores = []
    all_ranks = []

    for result in predictions:
        all_scores.append(result['query']['true_object_score'])
        rank = result['query']['true_object_rank']
        if rank is not None:
            all_ranks.append(rank)

    col1, col2 = st.columns(2)

    with col1:
        # Score histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=all_scores,
            nbinsx=30,
            marker_color='#3498db',
            opacity=0.7
        ))

        fig.update_layout(
            title='Distribution of True Answer Scores',
            xaxis_title='Score',
            yaxis_title='Count',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Rank histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=all_ranks,
            nbinsx=20,
            marker_color='#e74c3c',
            opacity=0.7
        ))

        fig.update_layout(
            title='Distribution of True Answer Ranks',
            xaxis_title='Rank',
            yaxis_title='Count',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    # Score vs Rank scatter
    st.subheader("Score vs Rank")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=all_ranks,
        y=all_scores,
        mode='markers',
        marker=dict(
            size=8,
            color=all_ranks,
            colorscale='RdYlGn_r',
            opacity=0.6,
            colorbar=dict(title='Rank')
        )
    ))

    fig.update_layout(
        title='Prediction Score vs True Answer Rank',
        xaxis_title='True Answer Rank',
        yaxis_title='Prediction Score',
        height=500,
        xaxis_type='log'
    )

    st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(
        page_title="FuseLinker Prediction Explorer",
        page_icon="🔗",
        layout="wide"
    )

    st.title("🔗 FuseLinker Link Prediction Explorer")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        prediction_file = st.text_input(
            "Prediction File",
            "fuselinker-complex/suppkg/link_predictions.json"
        )

        if not Path(prediction_file).exists():
            st.error(f"File not found: {prediction_file}")
            st.info("Run: `python -m visualization.link_predictor` to generate predictions first")
            st.stop()

        explanation_file = prediction_file.replace('predictions.json', 'explanations.json')

        if st.button("🔄 Reload Data"):
            st.cache_data.clear()

    # Load data
    try:
        with st.spinner("Loading predictions..."):
            pred_data = load_predictions(prediction_file)
            explanations = load_explanations(explanation_file)

        st.sidebar.success(f"✓ Loaded {pred_data['statistics']['total_queries']} queries")

        if explanations:
            st.sidebar.success(f"✓ Loaded {len(explanations)} explanations")
        else:
            st.sidebar.warning("⚠ No explanations found")
            st.sidebar.info("Run: `python -m visualization.explainer` to generate explanations")

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Navigation
    page = st.sidebar.radio(
        "📖 Navigation",
        ["Statistics", "Query Explorer", "Comparison"]
    )

    # Display selected page
    if page == "Statistics":
        show_prediction_statistics(pred_data)
    elif page == "Query Explorer":
        show_query_explorer(pred_data, explanations)
    elif page == "Comparison":
        show_comparison(pred_data)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
        <small>FuseLinker Link Prediction Explorer |
        {queries} queries analyzed |
        Hits@1: {hits1:.2%} | MRR: {mrr:.4f}</small>
        </div>
        """.format(
            queries=pred_data['statistics']['total_queries'],
            hits1=pred_data['statistics']['hits_at_1'],
            mrr=pred_data['statistics']['mrr']
        ),
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
