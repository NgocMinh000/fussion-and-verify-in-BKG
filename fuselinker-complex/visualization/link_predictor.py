"""
Link Prediction Module - Extract và Analyze Fused Links

Module này cho phép:
1. Load trained FuseLinker model
2. Predict new links (chưa có trong training data)
3. Rank predictions by confidence score
4. Export và visualize predicted links
5. So sánh với existing links
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from tqdm import tqdm
import pickle

from model import LinkPredict
from data_loader import Data
import myutils


class LinkPredictor:
    """
    Predict và analyze fused links từ trained FuseLinker model

    Usage:
        predictor = LinkPredictor()
        predictor.load_model('suppkg_model_state.pth', ...)
        predictions = predictor.predict_new_links(top_k=1000)
        predictor.export_predictions('predicted_links.csv')
    """

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize LinkPredictor

        Args:
            device: Device to run predictions on
        """
        self.device = torch.device(device)
        self.model = None
        self.embeddings = None
        self.relation_weights = None
        self.entity2index = None
        self.index2entity = None
        self.relation2index = None
        self.index2relation = None
        self.existing_triplets = None
        self.predictions = None

    def load_model(
        self,
        model_state_path: str,
        data_dir: str,
        text_embedding_path: str,
        knowledge_embedding_path: str,
        n_hidden: int = 200,
        num_bases: int = 20,
        num_hidden_layers: int = 2,
        dropout: float = 0.2,
        reg_param: float = 0.01,
        w: float = 0.75
    ):
        """
        Load trained model và data mappings

        Args:
            model_state_path: Path to saved model state (.pth file)
            data_dir: Directory containing data (train.tsv, entity2index.pkl, etc.)
            text_embedding_path: Path to text embeddings
            knowledge_embedding_path: Path to domain knowledge embeddings
            n_hidden: Hidden dimension size
            num_bases: Number of basis for R-GCN
            num_hidden_layers: Number of hidden layers
            dropout: Dropout rate
            reg_param: Regularization parameter
            w: Fusion weight
        """
        print("="*60)
        print("Loading FuseLinker Model for Link Prediction")
        print("="*60)

        data_dir = Path(data_dir)

        # Load data
        print("\n[1/5] Loading data...")
        train = pd.read_csv(data_dir / 'train.tsv', sep='\t', header=None)
        valid = pd.read_csv(data_dir / 'valid.tsv', sep='\t', header=None)
        test = pd.read_csv(data_dir / 'test.tsv', sep='\t', header=None)
        graph = pd.concat([train, valid, test])

        knowledge_graph = Data(graph, train, valid, test)

        self.entity2index = knowledge_graph.entity2index
        self.index2entity = knowledge_graph.index2entity
        self.relation2index = knowledge_graph.relation2index
        self.index2relation = knowledge_graph.index2relation

        # Store existing triplets
        self.existing_triplets = set(
            tuple(triplet) for triplet in knowledge_graph.total_data.tolist()
        )

        num_nodes, num_rels, num_edges = knowledge_graph.get_stats()
        print(f"  ✓ Entities: {num_nodes}")
        print(f"  ✓ Relations: {num_rels}")
        print(f"  ✓ Existing links: {num_edges}")

        # Load embeddings
        print("\n[2/5] Loading embeddings...")
        text_embeddings = np.load(text_embedding_path)
        ontology_embeddings = np.load(knowledge_embedding_path)
        print(f"  ✓ Text embeddings: {text_embeddings.shape}")
        print(f"  ✓ Domain embeddings: {ontology_embeddings.shape}")

        # Initialize model
        print("\n[3/5] Initializing model...")
        self.model = LinkPredict(
            num_nodes,
            n_hidden,
            num_rels,
            num_bases=num_bases,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
            use_cuda=(self.device.type == 'cuda'),
            regularization_param=reg_param,
            pretrained_text_embeddings=text_embeddings,
            pretrained_domain_embeddings=ontology_embeddings,
            freeze=False,
            w=w
        )
        self.model = self.model.to(self.device)

        # Load trained weights
        print("\n[4/5] Loading trained weights...")
        checkpoint = torch.load(model_state_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        print(f"  ✓ Loaded from iteration {checkpoint.get('iteration', 'unknown')}")

        # Generate embeddings
        print("\n[5/5] Generating embeddings...")
        test_graph, test_rel, test_norm = myutils.build_graph(
            num_nodes, num_rels, knowledge_graph.test_data
        )
        test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
        test_rel = torch.from_numpy(test_rel)
        test_norm = myutils.node_norm_2_edge_norm(
            test_graph, torch.from_numpy(test_norm).view(-1, 1)
        )

        with torch.no_grad():
            self.embeddings = self.model(
                test_graph.to(self.device),
                test_node_id.to(self.device),
                test_rel.to(self.device),
                test_norm.to(self.device)
            )
            self.relation_weights = self.model.relation_weights

        print(f"  ✓ Embeddings shape: {self.embeddings.shape}")
        print(f"  ✓ Relation weights shape: {self.relation_weights.shape}")
        print("\n" + "="*60)
        print("✓ Model loaded successfully!")
        print("="*60 + "\n")

    def predict_links_for_relation(
        self,
        relation_idx: int,
        top_k: int = 100,
        min_score: float = 0.5,
        exclude_existing: bool = True
    ) -> List[Tuple]:
        """
        Predict top-k links cho một relation cụ thể

        Args:
            relation_idx: Index của relation
            top_k: Number of top predictions to return
            min_score: Minimum confidence score
            exclude_existing: Exclude links already in train/valid/test

        Returns:
            List of (subject_idx, relation_idx, object_idx, score)
        """
        if self.embeddings is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        num_nodes = self.embeddings.shape[0]
        predictions = []

        # Get relation embedding
        rel_emb = self.relation_weights[relation_idx]

        # Calculate scores for all possible (subject, object) pairs
        # This is memory intensive, so we do it in batches
        batch_size = 1000

        print(f"Predicting links for relation {relation_idx}...")
        with torch.no_grad():
            for s_idx in tqdm(range(num_nodes)):
                # Subject embedding
                s_emb = self.embeddings[s_idx]

                # Calculate scores for all objects
                scores = torch.sigmoid(
                    torch.sum(s_emb * rel_emb * self.embeddings, dim=1)
                )

                # Get top scores
                for o_idx in range(num_nodes):
                    if s_idx == o_idx:  # Skip self-loops
                        continue

                    score = scores[o_idx].item()

                    if score < min_score:
                        continue

                    triplet = (s_idx, relation_idx, o_idx)

                    # Exclude existing triplets
                    if exclude_existing and triplet in self.existing_triplets:
                        continue

                    predictions.append((*triplet, score))

        # Sort by score and get top-k
        predictions.sort(key=lambda x: x[3], reverse=True)
        return predictions[:top_k]

    def predict_new_links(
        self,
        top_k_per_relation: int = 100,
        min_score: float = 0.5,
        relations: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Predict new links for all (or specified) relations

        Args:
            top_k_per_relation: Top K predictions per relation
            min_score: Minimum confidence score
            relations: List of relation indices to predict (None = all)

        Returns:
            DataFrame with columns: subject, relation, object, score, subject_type, object_type
        """
        if self.embeddings is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        num_rels = self.relation_weights.shape[0]

        if relations is None:
            relations = list(range(num_rels))

        all_predictions = []

        for rel_idx in relations:
            print(f"\n[Relation {rel_idx+1}/{len(relations)}]")
            preds = self.predict_links_for_relation(
                rel_idx,
                top_k=top_k_per_relation,
                min_score=min_score,
                exclude_existing=True
            )
            all_predictions.extend(preds)

        # Convert to DataFrame
        df_data = []
        for s_idx, r_idx, o_idx, score in all_predictions:
            subject = self.index2entity[s_idx]
            object_entity = self.index2entity[o_idx]

            # Get relation name
            relation_tuple = self.index2relation[r_idx]
            if isinstance(relation_tuple, tuple):
                relation = relation_tuple[1] if len(relation_tuple) > 1 else str(relation_tuple)
            else:
                relation = str(relation_tuple)

            # Get semantic types
            subject_type = subject.split('_')[-1] if '_' in subject else 'unknown'
            object_type = object_entity.split('_')[-1] if '_' in object_entity else 'unknown'

            df_data.append({
                'subject': subject,
                'relation': relation,
                'object': object_entity,
                'score': score,
                'subject_type': subject_type,
                'object_type': object_type,
                'subject_idx': s_idx,
                'relation_idx': r_idx,
                'object_idx': o_idx
            })

        self.predictions = pd.DataFrame(df_data)
        self.predictions = self.predictions.sort_values('score', ascending=False).reset_index(drop=True)

        print(f"\n{'='*60}")
        print(f"✓ Generated {len(self.predictions)} new link predictions")
        print(f"{'='*60}\n")

        return self.predictions

    def get_top_predictions(self, k: int = 100) -> pd.DataFrame:
        """Get top K predictions by score"""
        if self.predictions is None:
            raise ValueError("No predictions yet. Call predict_new_links() first.")
        return self.predictions.head(k)

    def filter_predictions(
        self,
        semantic_types: Optional[List[str]] = None,
        relations: Optional[List[str]] = None,
        min_score: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Filter predictions by criteria

        Args:
            semantic_types: Filter by subject/object semantic types
            relations: Filter by relation names
            min_score: Minimum score threshold

        Returns:
            Filtered DataFrame
        """
        if self.predictions is None:
            raise ValueError("No predictions yet. Call predict_new_links() first.")

        df = self.predictions.copy()

        if semantic_types:
            df = df[
                df['subject_type'].isin(semantic_types) |
                df['object_type'].isin(semantic_types)
            ]

        if relations:
            df = df[df['relation'].isin(relations)]

        if min_score:
            df = df[df['score'] >= min_score]

        return df

    def export_predictions(self, output_path: str):
        """Export predictions to CSV"""
        if self.predictions is None:
            raise ValueError("No predictions yet. Call predict_new_links() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.predictions.to_csv(output_path, index=False)
        print(f"✓ Exported {len(self.predictions)} predictions to {output_path}")

    def analyze_predictions(self) -> Dict:
        """
        Analyze prediction statistics

        Returns:
            Dictionary of statistics
        """
        if self.predictions is None:
            raise ValueError("No predictions yet. Call predict_new_links() first.")

        stats = {
            'total_predictions': len(self.predictions),
            'score_mean': self.predictions['score'].mean(),
            'score_median': self.predictions['score'].median(),
            'score_std': self.predictions['score'].std(),
            'score_min': self.predictions['score'].min(),
            'score_max': self.predictions['score'].max(),
            'relations': self.predictions['relation'].value_counts().to_dict(),
            'subject_types': self.predictions['subject_type'].value_counts().to_dict(),
            'object_types': self.predictions['object_type'].value_counts().to_dict(),
        }

        return stats

    def print_top_predictions(self, k: int = 10):
        """Print top K predictions"""
        if self.predictions is None:
            raise ValueError("No predictions yet. Call predict_new_links() first.")

        print(f"\n{'='*80}")
        print(f"TOP {k} PREDICTED LINKS (New Fused Links)")
        print(f"{'='*80}\n")

        top_k = self.predictions.head(k)

        for idx, row in top_k.iterrows():
            print(f"[{idx+1}] Score: {row['score']:.4f}")
            print(f"    {row['subject']} ({row['subject_type']})")
            print(f"      --[{row['relation']}]-->")
            print(f"    {row['object']} ({row['object_type']})")
            print()


def quick_predict(
    model_state_path: str,
    data_dir: str,
    text_embedding_path: str,
    knowledge_embedding_path: str,
    output_path: str = 'predicted_links.csv',
    top_k_per_relation: int = 100,
    min_score: float = 0.7,
    **model_params
) -> pd.DataFrame:
    """
    Quick function to predict links from trained model

    Args:
        model_state_path: Path to trained model
        data_dir: Data directory
        text_embedding_path: Path to text embeddings
        knowledge_embedding_path: Path to domain embeddings
        output_path: Where to save predictions
        top_k_per_relation: Top K per relation
        min_score: Minimum score
        **model_params: Additional model parameters

    Returns:
        DataFrame of predictions
    """
    predictor = LinkPredictor()

    # Load model
    predictor.load_model(
        model_state_path,
        data_dir,
        text_embedding_path,
        knowledge_embedding_path,
        **model_params
    )

    # Predict
    predictions = predictor.predict_new_links(
        top_k_per_relation=top_k_per_relation,
        min_score=min_score
    )

    # Export
    predictor.export_predictions(output_path)

    # Show top predictions
    predictor.print_top_predictions(10)

    # Print statistics
    stats = predictor.analyze_predictions()
    print(f"\n{'='*60}")
    print("PREDICTION STATISTICS")
    print(f"{'='*60}")
    print(f"Total predictions: {stats['total_predictions']}")
    print(f"Score range: [{stats['score_min']:.4f}, {stats['score_max']:.4f}]")
    print(f"Score mean: {stats['score_mean']:.4f} ± {stats['score_std']:.4f}")
    print(f"\nTop relations:")
    for rel, count in list(stats['relations'].items())[:5]:
        print(f"  {rel}: {count}")
    print(f"{'='*60}\n")

    return predictions
