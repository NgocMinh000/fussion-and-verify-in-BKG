#!/usr/bin/env python3
"""
Calculate required iterations for fair comparison with/without reciprocal relations
"""

# Your data
num_train_triples = 244788
graph_batch_size = 250
target_epochs = 40  # Target number of epochs to train

print("=" * 70)
print("ITERATIONS CALCULATOR FOR FAIR COMPARISON")
print("=" * 70)
print()

# Without reciprocal
iterations_per_epoch_no_recip = num_train_triples / graph_batch_size
iterations_needed_no_recip = int(target_epochs * iterations_per_epoch_no_recip)

print(f"WITHOUT --use_reciprocal:")
print(f"  Training triples: {num_train_triples:,}")
print(f"  Batch size: {graph_batch_size}")
print(f"  Iterations per epoch: {iterations_per_epoch_no_recip:.1f}")
print(f"  To train {target_epochs} epochs: {iterations_needed_no_recip:,} iterations")
print()

# With reciprocal (data doubled)
num_train_triples_recip = num_train_triples * 2
iterations_per_epoch_recip = num_train_triples_recip / graph_batch_size
iterations_needed_recip = int(target_epochs * iterations_per_epoch_recip)

print(f"WITH --use_reciprocal:")
print(f"  Training triples: {num_train_triples_recip:,} (doubled)")
print(f"  Batch size: {graph_batch_size}")
print(f"  Iterations per epoch: {iterations_per_epoch_recip:.1f}")
print(f"  To train {target_epochs} epochs: {iterations_needed_recip:,} iterations")
print()

print("=" * 70)
print("RECOMMENDATION FOR FAIR COMPARISON:")
print("=" * 70)
print()
print(f"NO reciprocal:   --iterations {iterations_needed_no_recip:,}")
print(f"WITH reciprocal: --iterations {iterations_needed_recip:,}")
print()
print(f"Ratio: {iterations_needed_recip / iterations_needed_no_recip:.2f}x")
print()
print("This ensures both models see the same number of epochs!")
print("=" * 70)
