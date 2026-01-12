import argparse
import numpy as np
import pandas as pd
import torch
import pickle

import myutils
from model import LinkPredict
from data_loader import Data


def add_reciprocal_relations(data_df, num_relations):
    """
    Add reciprocal (inverse) relations for each triple.
    For each (h, r, t), add (t, r_inv, h) where r_inv = r + num_relations.

    Args:
        data_df: DataFrame with columns [head, relation, tail]
        num_relations: number of unique relations

    Returns:
        augmented_df: DataFrame with reciprocal triples added
        new_num_relations: num_relations * 2
    """
    data_np = data_df.values

    # Create inverse triples
    inverse_data = data_np.copy()
    inverse_data[:, 0] = data_np[:, 2]  # head <- tail
    inverse_data[:, 2] = data_np[:, 0]  # tail <- head
    inverse_data[:, 1] = data_np[:, 1] + num_relations  # relation <- relation + offset

    # Concatenate original and inverse
    augmented_data = np.vstack([data_np, inverse_data])
    augmented_df = pd.DataFrame(augmented_data)

    return augmented_df, num_relations * 2


def main(args):
    import os

    train_path = f'{args.data}/train.tsv'
    valid_path = f'{args.data}/valid.tsv'
    test_path = f'{args.data}/test.tsv'

    # Handle embedding paths: use absolute path if provided, otherwise use relative to data dir
    if os.path.isabs(args.text_embedding_file):
        text_embedding_path = os.path.expanduser(args.text_embedding_file)
    else:
        text_embedding_path = f'{args.data}/{args.text_embedding_file}'

    if os.path.isabs(args.knowledge_embedding_file):
        knowledge_embedding_path = os.path.expanduser(args.knowledge_embedding_file)
    else:
        knowledge_embedding_path = f'{args.data}/{args.knowledge_embedding_file}'

    freeze = args.freeze

    train = pd.read_csv(train_path, sep='\t', header=None)
    valid = pd.read_csv(valid_path, sep='\t', header=None)
    test = pd.read_csv(test_path, sep='\t', header=None)

    # Add reciprocal relations (best practice for KGE)
    if args.use_reciprocal:
        num_relations_original = int(train[1].max()) + 1
        print(f"Adding reciprocal relations... Original relations: {num_relations_original}")

        train, num_relations = add_reciprocal_relations(train, num_relations_original)
        valid, _ = add_reciprocal_relations(valid, num_relations_original)
        test, _ = add_reciprocal_relations(test, num_relations_original)

        print(f"✓ Added reciprocal relations: {num_relations_original} -> {num_relations} relations")
        print(f"  Train triples: {len(train)} (doubled)")
    else:
        print("Reciprocal relations disabled (use --use_reciprocal to enable)")

    graph = pd.concat([train, valid, test])

    print("Loading Pretrained Embeddings files...")
    print(f"Text embedding path: {text_embedding_path}")
    try:
        text_embeddings = np.load(text_embedding_path)
        print(f"✓ Loaded Text Embeddings file successfully! Shape: {text_embeddings.shape}")
    except Exception as e:
        text_embeddings = None
        print(f"✗ Failed to load Text Embeddings file: {e}")
        print("  Random embeddings will be created.")

    print(f"Knowledge embedding path: {knowledge_embedding_path}")
    try:
        ontology_embeddings = np.load(knowledge_embedding_path)
        print(f"✓ Loaded Domain Knowledge Embeddings file successfully! Shape: {ontology_embeddings.shape}")
    except Exception as e:
        ontology_embeddings = None
        print(f"✗ Failed to load Domain Knowledge Embeddings file: {e}")
        print("  Random embeddings will be created.")

    print(f"w: {args.w}")
    
    print("Data Processing...")
    knowledge_graph = Data(graph, train, valid, test)
    num_nodes, num_rels, num_edges = knowledge_graph.get_stats()
    print('# entities:', num_nodes)
    print('# relations:', num_rels)
    print('# edges:', num_edges)
    with open(f'{args.data}/relation2index.pkl', 'wb') as file:
        pickle.dump(knowledge_graph.relation2index, file)
    with open(f'{args.data}/index2relation.pkl', 'wb') as file:
        pickle.dump(knowledge_graph.index2relation, file)
    with open(f'{args.data}/entity2index.pkl', 'wb') as file:
        pickle.dump(knowledge_graph.entity2index, file)
    with open(f'{args.data}/index2entity.pkl', 'wb') as file:
        pickle.dump(knowledge_graph.index2entity, file)

    train_data_np = knowledge_graph.train_data
    valid_data_np = knowledge_graph.valid_data
    test_data_np = knowledge_graph.test_data
    total_data_np = knowledge_graph.total_data

    train_data = torch.LongTensor(train_data_np)
    valid_data = torch.LongTensor(valid_data_np)
    test_data = torch.LongTensor(test_data_np)
    total_data = torch.LongTensor(total_data_np)

    model = LinkPredict(num_nodes,
                        args.n_hidden,
                        num_rels,
                        num_bases=args.num_bases,
                        num_hidden_layers=args.num_hidden_layers,
                        dropout=args.dropout,
                        use_cuda=args.use_cuda,
                        regularization_param=args.reg_param,
                        pretrained_text_embeddings=text_embeddings,
                        pretrained_domain_embeddings=ontology_embeddings,
                        freeze=freeze,
                        w=args.w,
                        use_n3_reg=args.use_n3_reg)


    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    model = model.to(device)
    print(device)

    # build train graph
    train_graph, train_rel, train_norm = myutils.build_graph(num_nodes, num_rels, train_data_np)
    train_deg = train_graph.in_degrees(range(train_graph.number_of_nodes())).float().view(-1, 1)

    # build test graph
    test_graph, test_rel, test_norm = myutils.build_graph(num_nodes, num_rels, test_data_np)
    test_deg = test_graph.in_degrees(range(test_graph.number_of_nodes())).float().view(-1, 1)
    test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    test_rel = torch.from_numpy(test_rel)
    test_norm = myutils.node_norm_2_edge_norm(test_graph, torch.from_numpy(test_norm).view(-1, 1))

    # build adj list and calculate degrees for sampling
    adj_list = myutils.get_adj(num_nodes, train_data_np)  # degrees

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # training loop
    print("Start training...")

    # epoch is step
    for iteration in range(1, 1 + args.iterations):
        model.train()

        # perform edge neighborhood sampling to generate training graph and data
        g, node_id, edge_type, node_norm, data, labels = \
            myutils.generate_sampled_graph_and_labels(
                train_data_np, args.graph_batch_size, args.graph_split_size,
                num_rels, adj_list, train_deg, args.negative_sample,
                args.edge_sampler)

        # set node/edge feature
        node_id = torch.from_numpy(node_id).view(-1, 1).long()
        edge_type = torch.from_numpy(edge_type)
        edge_norm = myutils.node_norm_2_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
        data, labels = torch.from_numpy(data), torch.from_numpy(labels)
        deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)

        # Load on device
        g = g.to(device)
        node_id = node_id.to(device)
        edge_type = edge_type.to(device)
        edge_norm = edge_norm.to(device)
        data = data.to(device)
        labels = labels.to(device)

        embed = model(g, node_id, edge_type, edge_norm)
        loss = model.get_loss(g, embed, data, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
        optimizer.step()

        if iteration % args.evaluate_every == 0:
            print("Epoch {} | Loss {:.5f}".format(iteration, loss.item()))

        optimizer.zero_grad()

    torch.save({'state_dict': model.state_dict(), 'iteration': iteration}, args.model_state_file)

    print("Evaluating...")
    model.eval()
    test_data = torch.LongTensor(test_data_np)
    test_graph = test_graph.to(device)
    test_node_id = test_node_id.to(device)
    test_rel = test_rel.to(device)
    test_norm = test_norm.to(device)
    test_data = test_data.to(device)
    total_data = torch.LongTensor(total_data)

    output = model(test_graph, test_node_id, test_rel, test_norm)

    import time
    old_time = time.time()

    hits = [1, 3, 10]
    mr, mrr, hits_dict = myutils.calc_mrr(output, model.relation_weights, test_data,
                                              torch.LongTensor(total_data).to(device),
                                              batch_size=args.eval_batch_size, neg_sample_size_eval=args.neg_sample_size_eval,
                                              hits=hits, score_function=model.calculate_score, eval_p=args.eval_protocol)

    new_time = time.time()
    print(new_time - old_time)

    print(f"MR: {mr:.6f}")
    print(f"MRR: {mrr:.6f}")
    for key, value in hits_dict.items():
        print(f"Hits @ {key} = {value:.6f}")

    # Export visualization data
    print("\n" + "="*60)
    print("Exporting data for visualization...")
    print("="*60)

    try:
        from visualization.export_utils import export_full_visualization_data

        viz_output_dir = f'{args.data}/visualization_outputs'

        export_full_visualization_data(
            model=model,
            graph=test_graph,
            node_ids=test_node_id,
            rel_ids=test_rel,
            norm=test_norm,
            train_data=train_data_np,
            test_data=test_data_np,
            entity2index=knowledge_graph.entity2index,
            index2entity=knowledge_graph.index2entity,
            relation2index=knowledge_graph.relation2index,
            index2relation=knowledge_graph.index2relation,
            output_dir=viz_output_dir,
            device=device
        )

        print(f"\n✓ Visualization data exported to: {viz_output_dir}")
        print(f"\nTo visualize:")
        print(f"  1. Graph structure: python -m visualization.graph_visualizer")
        print(f"  2. Embeddings: python -m visualization.embedding_visualizer")
        print(f"  3. Dashboard: python -m visualization.app")

    except ImportError as e:
        print(f"\nWarning: Could not import visualization module: {e}")
        print("Skipping visualization data export.")
    except Exception as e:
        print(f"\nError during visualization export: {e}")
        print("Training completed but visualization data not exported.")

    print("\nTraining done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parser For Arguments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data",
        dest="data",
        default="data",
        help="The folder of data, should include train.tsv, valid.tsv and test.tsv",
    )

    parser.add_argument(
        "--text_embedding_file",
        dest="text_embedding_file",
        default="pubmedbert_embeddings_768.npy",
        help="Path of text embedding for each node",
    )

    parser.add_argument(
        "--knowledge_embedding_file",
        dest="knowledge_embedding_file",
        default="poincare_embeddings.npy",
        help="Path of domain knowledge embedding for each node",
    )
    

    parser.add_argument(
        "--freeze", action="store_true",
        help="Freeze text embedding and domain knowledge or not"
    )

    parser.add_argument(
        "--use_reciprocal", action="store_true",
        help="Add reciprocal (inverse) relations for each triple (recommended for better performance)"
    )

    parser.add_argument(
        "--w",
        dest="w",
        type=float,
        default=0.5,
        help="The weight for fusing embedings",
    )

    parser.add_argument(
        "--use_n3_reg", action="store_true",
        help="Use N3 regularization instead of L2 (recommended for ComplEx)"
    )

    parser.add_argument(
        "--n_hidden",
        dest="n_hidden",
        type=int,
        default=200,
        help="Dimensions of the hidden layer",
    )

    parser.add_argument(
        "--num_bases",
        dest="num_bases",
        type=int,
        default=20,
        help="Number of basis relation vectors to use",
    )

    parser.add_argument(
        "--num_hidden_layers",
        dest="num_hidden_layers",
        type=int,
        default=1,
        help="Number of hidden layers",
    )

    parser.add_argument(
        "--dropout",
        dest="dropout",
        type=float,
        default=0.2,
        help="Number of hidden layers",
    )

    parser.add_argument(
        "--use_cuda",
        dest="use_cuda",
        type=bool,
        default=True,
        help="GPU",
    )

    parser.add_argument(
        "--reg_param",
        dest="reg_param",
        type=float,
        default=0.01,
        help="GPU",
    )


    parser.add_argument(
        "--iterations",
        dest="iterations",
        type=int,
        default=1,
        help="Number of iterations (iterations = epochs * (datasize / batchsize))",
    )

    parser.add_argument(
        "--evaluate_every",
        dest="evaluate_every",
        type=int,
        default=4000,
    )

    parser.add_argument(
        "--lr",
        dest="lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )

    parser.add_argument(
        "--graph_batch_size",
        dest="graph_batch_size",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--graph_split_size",
        dest="graph_split_size",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--negative_sample",
        dest="negative_sample",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--edge_sampler",
        dest="edge_sampler",
        default="uniform",
    )

    parser.add_argument(
        "--grad_norm",
        dest="grad_norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--eval_batch_size",
        dest="eval_batch_size",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--neg_sample_size_eval",
        dest="neg_sample_size_eval",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--eval_protocol",
        dest="eval_protocol",
        default="filtered",
    )

    parser.add_argument(
        "--model_state_file",
        dest="model_state_file",
        default="model_state.pth",
    )

    args = parser.parse_args()

    main(args)