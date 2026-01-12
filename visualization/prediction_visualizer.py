"""
Link Prediction Visualization.
Creates visual representations of predicted triples and explanation insights.
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import seaborn as sns


def load_predictions(prediction_file):
    """Load predictions from JSON."""
    with open(prediction_file, 'r') as f:
        return json.load(f)


def plot_prediction_scores(query_result, output_path, top_k=10):
    """
    Plot prediction scores for a single query as a horizontal bar chart.
    Highlights the correct answer.
    """
    query = query_result['query']
    predictions = query_result['predictions'][:top_k]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Extract data
    entities = [p['entity_name'][:50] for p in predictions]
    scores = [p['score'] for p in predictions]
    is_correct = [p['is_correct'] for p in predictions]

    # Color bars: green if correct, blue otherwise
    colors = ['#2ecc71' if correct else '#3498db' for correct in is_correct]

    # Create horizontal bar chart
    y_pos = np.arange(len(entities))
    bars = ax.barh(y_pos, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.2)

    # Add score labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f' {score:.3f}',
                ha='left', va='center', fontsize=9, fontweight='bold')

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"#{i+1} {entities[i]}" for i in range(len(entities))], fontsize=10)
    ax.set_xlabel('Prediction Score', fontsize=12, fontweight='bold')
    ax.set_title(
        f"Link Prediction: ({query['subject']['name']}, {query['relation']['name']}, ?)\n"
        f"True Answer: {query['true_object']['name']} (Rank: {query['true_object_rank']})",
        fontsize=13, fontweight='bold', pad=20
    )
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()  # Highest score at top

    # Legend
    correct_patch = mpatches.Patch(color='#2ecc71', label='Correct Answer')
    predicted_patch = mpatches.Patch(color='#3498db', label='Predictions')
    ax.legend(handles=[correct_patch, predicted_patch], loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved prediction plot: {output_path}")


def plot_hits_at_k_distribution(pred_data, output_path):
    """Plot distribution of true answer ranks (Hits@K analysis)."""
    predictions = pred_data['predictions']

    ranks = []
    for result in predictions:
        rank = result['query']['true_object_rank']
        if rank is not None:
            ranks.append(rank)

    # Count ranks
    rank_counts = {}
    for r in ranks:
        if r <= 10:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        else:
            rank_counts['>10'] = rank_counts.get('>10', 0) + 1

    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort keys
    sorted_ranks = sorted([r for r in rank_counts.keys() if r != '>10']) + ['>10']
    counts = [rank_counts.get(r, 0) for r in sorted_ranks]

    # Color: green for rank 1, yellow for 2-3, orange for 4-10, red for >10
    colors = []
    for r in sorted_ranks:
        if r == 1:
            colors.append('#2ecc71')  # green
        elif r in [2, 3]:
            colors.append('#f39c12')  # yellow
        elif r == '>10':
            colors.append('#e74c3c')  # red
        else:
            colors.append('#3498db')  # blue

    bars = ax.bar(range(len(sorted_ranks)), counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.2)

    # Add count labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(sorted_ranks)))
    ax.set_xticklabels([str(r) for r in sorted_ranks], fontsize=11)
    ax.set_xlabel('Rank of True Answer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Queries', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of True Answer Ranks (Hits@K Analysis)', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Legend
    rank1_patch = mpatches.Patch(color='#2ecc71', label='Rank 1 (Hits@1)')
    rank2_3_patch = mpatches.Patch(color='#f39c12', label='Rank 2-3 (Hits@3)')
    rank4_10_patch = mpatches.Patch(color='#3498db', label='Rank 4-10 (Hits@10)')
    rank_over_patch = mpatches.Patch(color='#e74c3c', label='Rank >10')
    ax.legend(handles=[rank1_patch, rank2_3_patch, rank4_10_patch, rank_over_patch],
              loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved rank distribution: {output_path}")


def plot_score_vs_rank(pred_data, output_path):
    """Plot prediction scores vs actual ranks."""
    all_scores = []
    all_ranks = []

    for result in pred_data['predictions']:
        true_rank = result['query']['true_object_rank']
        true_score = result['query']['true_object_score']

        if true_rank is not None:
            all_scores.append(true_score)
            all_ranks.append(true_rank)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot with log scale for ranks
    scatter = ax.scatter(all_ranks, all_scores, alpha=0.6, s=50, c=all_ranks,
                        cmap='RdYlGn_r', edgecolors='black', linewidth=0.5)

    ax.set_xlabel('True Answer Rank', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prediction Score', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Score vs True Answer Rank', fontsize=14, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Rank', fontsize=11)

    # Add trend line
    if len(all_ranks) > 0:
        z = np.polyfit(np.log(all_ranks), all_scores, 1)
        p = np.poly1d(z)
        sorted_ranks = sorted(all_ranks)
        ax.plot(sorted_ranks, p(np.log(sorted_ranks)), "r--", alpha=0.8, linewidth=2, label='Trend')
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved score vs rank plot: {output_path}")


def plot_similarity_heatmap(explanation, output_path):
    """Plot heatmap of embedding similarities for top predictions."""
    emb_analysis = explanation['embedding_analysis']

    if not emb_analysis:
        return

    # Extract similarities
    ranks = [e['rank'] for e in emb_analysis]
    entities = [e['entity_name'][:30] for e in emb_analysis]

    similarity_types = ['subject_object', 'subject_relation', 'relation_object', 'combined_alignment']
    similarity_labels = ['Subject↔Object', 'Subject↔Relation', 'Relation↔Object', 'Combined']

    similarity_matrix = []
    for e in emb_analysis:
        row = [e['similarities'][st] for st in similarity_types]
        similarity_matrix.append(row)

    similarity_matrix = np.array(similarity_matrix)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Heatmap
    im = ax.imshow(similarity_matrix.T, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)

    # Ticks and labels
    ax.set_xticks(np.arange(len(entities)))
    ax.set_yticks(np.arange(len(similarity_labels)))
    ax.set_xticklabels([f"#{ranks[i]} {entities[i]}" for i in range(len(entities))],
                       rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(similarity_labels, fontsize=10)

    # Add values
    for i in range(len(similarity_labels)):
        for j in range(len(entities)):
            text = ax.text(j, i, f'{similarity_matrix[j, i]:.2f}',
                          ha="center", va="center", color="black", fontsize=9, fontweight='bold')

    ax.set_title(f"Embedding Similarity Analysis\n{explanation['query']['subject']['name']} → {explanation['query']['relation']['name']} → ?",
                fontsize=12, fontweight='bold', pad=20)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved similarity heatmap: {output_path}")


def create_summary_report(pred_data, output_path):
    """Create a text summary report of predictions."""
    stats = pred_data['statistics']
    predictions = pred_data['predictions']

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("LINK PREDICTION SUMMARY REPORT\n")
        f.write("="*100 + "\n\n")

        # Statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*100 + "\n")
        f.write(f"Total queries analyzed: {stats['total_queries']}\n")
        f.write(f"Hits@1:  {stats['hits_at_1']:.4f} ({stats['hits_at_1']*100:.2f}%)\n")
        f.write(f"Hits@3:  {stats['hits_at_3']:.4f} ({stats['hits_at_3']*100:.2f}%)\n")
        f.write(f"Hits@10: {stats['hits_at_10']:.4f} ({stats['hits_at_10']*100:.2f}%)\n")
        f.write(f"MRR:     {stats['mrr']:.4f}\n")
        f.write(f"Average true rank: {stats['average_true_rank']:.2f}\n\n")

        # Sample predictions
        f.write("\n" + "="*100 + "\n")
        f.write("SAMPLE PREDICTIONS (Top 10)\n")
        f.write("="*100 + "\n\n")

        for i, result in enumerate(predictions[:10], 1):
            query = result['query']
            preds = result['predictions']

            f.write(f"\nQuery {i}:\n")
            f.write(f"  Subject:  {query['subject']['name']}\n")
            f.write(f"  Relation: {query['relation']['name']}\n")
            f.write(f"  True Object: {query['true_object']['name']}\n")
            f.write(f"  True Rank: {query['true_object_rank']}\n")
            f.write(f"  True Score: {query['true_object_score']:.4f}\n\n")
            f.write(f"  Top-10 Predictions:\n")

            for pred in preds[:10]:
                marker = "✓ CORRECT" if pred['is_correct'] else ""
                f.write(f"    {pred['rank']:2d}. {pred['entity_name'][:60]:60s}  Score: {pred['score']:8.4f}  {marker}\n")

            f.write("\n" + "-"*100 + "\n")

    print(f"  ✓ Saved summary report: {output_path}")


def visualize_all(prediction_file, output_dir, num_samples=5):
    """Generate all prediction visualizations."""
    print("="*100)
    print("LINK PREDICTION VISUALIZATION")
    print("="*100)

    # Load predictions
    pred_data = load_predictions(prediction_file)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating visualizations...")

    # 1. Overall statistics plots
    plot_hits_at_k_distribution(pred_data, f'{output_dir}/rank_distribution.png')
    plot_score_vs_rank(pred_data, f'{output_dir}/score_vs_rank.png')

    # 2. Individual query visualizations
    predictions = pred_data['predictions'][:num_samples]

    for i, query_result in enumerate(predictions):
        plot_prediction_scores(query_result, f'{output_dir}/query_{i+1}_predictions.png')

    # 3. Summary report
    create_summary_report(pred_data, f'{output_dir}/prediction_summary.txt')

    # 4. If explanations exist, visualize them
    explanation_file = prediction_file.replace('predictions.json', 'explanations.json')
    if Path(explanation_file).exists():
        print(f"\nFound explanations file, generating similarity heatmaps...")
        with open(explanation_file, 'r') as f:
            explanations = json.load(f)

        for i, explanation in enumerate(explanations[:num_samples]):
            plot_similarity_heatmap(explanation, f'{output_dir}/query_{i+1}_similarities.png')

    print(f"\n{'='*100}")
    print(f"✓ ALL VISUALIZATIONS SAVED TO: {output_dir}")
    print(f"{'='*100}")


def main():
    parser = argparse.ArgumentParser(description='Visualize link predictions')
    parser.add_argument(
        '--prediction_file',
        default='suppkg/link_predictions.json',
        help='Prediction file from link_predictor.py'
    )
    parser.add_argument(
        '--output_dir',
        default='suppkg/prediction_plots',
        help='Directory to save plots'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='Number of sample queries to visualize'
    )

    args = parser.parse_args()

    visualize_all(args.prediction_file, args.output_dir, args.num_samples)


if __name__ == '__main__':
    main()
