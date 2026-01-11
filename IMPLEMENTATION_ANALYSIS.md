# Analysis Report: Why TransE, ComplEx, ConvE Underperform DistMult

## Executive Summary

After comparing current implementations with original papers and best practices, I found **critical issues** that explain why the alternative scoring functions underperform DistMult:

### Critical Issues Found:
1. **TransE**: Missing L2 normalization (CRITICAL), wrong loss function
2. **ComplEx**: Imaginary parts incorrectly derived instead of independent, missing N3 regularization
3. **ConvE**: Potential batch normalization mode issues
4. **All methods**: Missing reciprocal relations, suboptimal hyperparameters

---

## 1. TransE Issues

### Current Implementation Problems:

#### ❌ CRITICAL: Missing L2 Normalization
```python
# Current (WRONG):
def calculate_score(self, embeddings, triplets):
    h = embeddings[triplets[:, 0]]
    r = self.relation_weights[triplets[:, 1]]
    t = embeddings[triplets[:, 2]]
    score = -torch.norm(h + r - t, p=1, dim=1)
    return score
```

**Problem**: No normalization! This is THE MOST CRITICAL issue with TransE.

**From research**:
- TransE **REQUIRES** entity embeddings to be L2-normalized to unit length
- Without normalization, embeddings grow arbitrarily large during training
- This is the #1 reason TransE fails in implementations

#### ❌ Wrong Loss Function
**Current**: Using `binary_cross_entropy_with_logits` (BCE)
**Should be**: Margin-based ranking loss

TransE produces **negative scores** (negative distances), which don't work well with BCE. Original paper uses:
```python
L = Σ max(0, γ + score_positive - score_negative)
```

### Recommended Fixes:

```python
def calculate_score(self, embeddings, triplets):
    """TransE with proper normalization"""
    h = embeddings[triplets[:, 0]]
    r = self.relation_weights[triplets[:, 1]]
    t = embeddings[triplets[:, 2]]

    # CRITICAL: L2 normalize entity embeddings
    h = F.normalize(h, p=2, dim=1)
    t = F.normalize(t, p=2, dim=1)

    # TransE score: -||h + r - t||_p
    score = -torch.norm(h + r - t, p=1, dim=1)  # or p=2 for L2
    return score

def get_loss(self, graph, embeddings, triplets, labels):
    """Use margin-based loss for TransE"""
    # Separate positive and negative triplets
    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_scores = self.calculate_score(embeddings, triplets[pos_mask])
    neg_scores = self.calculate_score(embeddings, triplets[neg_mask])

    # Margin ranking loss
    margin = 1.0
    loss = torch.mean(F.relu(margin + pos_scores.mean() - neg_scores.mean()))

    # Add regularization
    reg_loss = self.regularization_loss(embeddings)
    return loss + self.regularization_param * reg_loss
```

---

## 2. ComplEx Issues

### Current Implementation Problems:

#### ❌ CRITICAL: Imaginary Parts Incorrectly Derived
```python
# Current (WRONG):
self.entity_to_imag = nn.Linear(hidden_dim, hidden_dim, bias=False)

# In calculate_score:
h_imag = self.entity_to_imag(h_real)  # Derived from real!
t_imag = self.entity_to_imag(t_real)
```

**Problem**: Imaginary parts are computed as a LINEAR TRANSFORMATION of real parts. This is WRONG!

**From research**:
- Real and imaginary parts should be **INDEPENDENT parameters**
- Each entity needs TWO separate embedding vectors: one for real, one for imaginary
- Current approach severely limits expressiveness

#### ❌ Missing N3 Regularization
**Current**: Only L2 regularization
**Should be**: N3 (nuclear 3-norm) regularization

Research shows N3 regularization is superior for ComplEx:
```python
N3_reg = λ * Σ (|embedding|³)
```

### Recommended Fixes:

ComplEx requires significant architectural changes. We need to store separate real and imaginary embeddings for entities:

```python
class ComplExLinkPredict(nn.Module):
    def __init__(self, ...):
        # CRITICAL: Separate embeddings for real and imaginary
        self.entity_embeddings_real = nn.Parameter(...)
        self.entity_embeddings_imag = nn.Parameter(...)

        # Relation embeddings (real and imaginary)
        self.relation_weights_real = nn.Parameter(...)
        self.relation_weights_imag = nn.Parameter(...)

    def calculate_score(self, triplets):
        """ComplEx with independent real/imaginary parts"""
        # Get INDEPENDENT real and imaginary embeddings
        h_real = self.entity_embeddings_real[triplets[:, 0]]
        h_imag = self.entity_embeddings_imag[triplets[:, 0]]  # Independent!

        r_real = self.relation_weights_real[triplets[:, 1]]
        r_imag = self.relation_weights_imag[triplets[:, 1]]

        t_real = self.entity_embeddings_real[triplets[:, 2]]
        t_imag = self.entity_embeddings_imag[triplets[:, 2]]

        # ComplEx score: Re(<h, r, conj(t)>)
        score = torch.sum(
            h_real * r_real * t_real +
            h_real * r_imag * t_imag +
            h_imag * r_real * t_imag -
            h_imag * r_imag * t_real,
            dim=1
        )
        return score

    def n3_regularization(self):
        """N3 regularization (better than L2 for ComplEx)"""
        reg = (
            torch.mean(torch.abs(self.entity_embeddings_real) ** 3) +
            torch.mean(torch.abs(self.entity_embeddings_imag) ** 3) +
            torch.mean(torch.abs(self.relation_weights_real) ** 3) +
            torch.mean(torch.abs(self.relation_weights_imag) ** 3)
        )
        return reg
```

**PROBLEM**: This conflicts with the current R-GCN architecture which only produces ONE set of embeddings.

---

## 3. ConvE Issues

### Current Implementation Problems:

#### ⚠️ Batch Normalization Mode
Research shows ConvE is EXTREMELY sensitive to batch norm mode:
- Must use `model.eval()` during inference
- Training vs eval behavior is very different

#### ⚠️ Dropout + BatchNorm Interaction
Current order may not be optimal. Research suggests:
1. Convolution
2. Batch Normalization
3. Activation (ReLU)
4. Dropout

Current code has dropout BEFORE batch norm in some places.

### Recommended Fixes:

```python
def calculate_score(self, embeddings, triplets):
    """ConvE with proper batch norm handling"""
    batch_size = triplets.size(0)

    h_emb = embeddings[triplets[:, 0]]
    r_emb = self.relation_weights[triplets[:, 1]]
    t_emb = embeddings[triplets[:, 2]]

    # Reshape
    h_2d = h_emb.view(batch_size, 1, self.embedding_height, self.embedding_width)
    r_2d = r_emb.view(batch_size, 1, self.embedding_height, self.embedding_width)

    # Stack
    stacked = torch.cat([h_2d, r_2d], dim=2)

    # Proper order: BN -> Dropout -> Conv -> BN -> ReLU -> Dropout
    x = self.bn0(stacked)
    x = self.input_dropout(x)
    x = self.conv1(x)
    x = self.bn1(x)
    x = F.relu(x)
    x = self.feature_map_dropout(x)

    # Flatten and project
    x = x.view(batch_size, -1)
    x = self.fc(x)
    x = self.bn2(x)
    x = F.relu(x)
    x = self.output_dropout(x)

    # Score
    score = torch.sum(x * t_emb, dim=1) + self.b[triplets[:, 2]]
    return score
```

Also need to ensure proper train/eval mode switching in main training loop.

---

## 4. Common Issues Across All Methods

### ❌ Missing Reciprocal Relations

**Critical finding**: Adding reciprocal relations significantly improves performance for ALL methods.

For each triple (h, r, t), should also add (t, r_inv, h).

This needs to be done at **data loading** time, not in the model.

### Recommended Fix:

Modify data loader to add reciprocal relations:

```python
def add_reciprocal_relations(data):
    """Add inverse relations for each triple"""
    num_relations = data[:, 1].max() + 1

    # Create inverse triples
    inverse_data = data.copy()
    inverse_data[:, 0] = data[:, 2]  # swap head and tail
    inverse_data[:, 2] = data[:, 0]
    inverse_data[:, 1] = data[:, 1] + num_relations  # inverse relation ID

    # Concatenate
    augmented_data = np.vstack([data, inverse_data])
    return augmented_data, num_relations * 2  # doubled number of relations
```

This should be applied in `data_loader.py` or `main.py` before creating the knowledge graph.

---

## 5. Fundamental Architecture Conflict

### The R-GCN Problem

**Current system**: All variants use R-GCN to produce entity embeddings, then apply different scoring functions.

**Problem**:
- R-GCN produces ONE set of embeddings
- ComplEx needs TWO sets (real + imaginary) as INDEPENDENT parameters
- TransE needs normalized embeddings at scoring time
- Each scoring function has different embedding requirements

### Possible Solutions:

#### Option A: Post-process R-GCN embeddings
```python
# For TransE: normalize after R-GCN
embeddings = self.rgcn(...)
embeddings_normalized = F.normalize(embeddings, p=2, dim=1)

# For ComplEx: project to real/imag
embeddings = self.rgcn(...)
embeddings_real = embeddings
embeddings_imag = self.project_to_imag(embeddings)  # trainable projection
```

This is what current code does, but may limit expressiveness.

#### Option B: Separate embeddings per method
- Each method maintains its OWN entity embeddings (not from R-GCN)
- Only use R-GCN for structure learning, not final embeddings
- More parameters but more faithful to original methods

#### Option C: Hybrid approach
- Start with R-GCN embeddings
- Add method-specific learnable parameters
- E.g., for ComplEx: use R-GCN as real part, learn imaginary part separately

---

## 6. Hyperparameter Issues

Current hyperparameters may not be optimal for each method:

| Hyperparameter | Current | TransE Optimal | ComplEx Optimal | ConvE Optimal |
|---|---|---|---|---|
| Learning rate | ? | 0.0005-0.001 | 0.001 | 0.0005-0.003 |
| Regularization | L2 | L2 + margin | N3 (λ=1e-7) | L2 |
| Loss function | BCE | Margin-based | BCE | BCE |
| Negative samples | ? | 2-10 | 2-10 | 1-N scoring |
| Batch size | ? | 2048 | 128-2048 | 128 |

---

## Recommendations Priority

### CRITICAL (Must Fix):
1. **TransE**: Add L2 normalization to entity embeddings
2. **ComplEx**: Make imaginary parts independent parameters (requires architecture change)
3. **All methods**: Add reciprocal relations to dataset

### HIGH Priority:
4. **TransE**: Switch to margin-based loss function
5. **ComplEx**: Add N3 regularization
6. **ConvE**: Verify batch norm mode handling
7. **All methods**: Hyperparameter tuning per method

### MEDIUM Priority:
8. Consider Option B or C for embedding architecture
9. Add self-adversarial negative sampling
10. Implement proper evaluation filtering

---

## Proposed Action Plan

I recommend we fix issues in this order:

1. **Quick wins** (can be done immediately):
   - Add L2 normalization to TransE scoring
   - Add reciprocal relations to data
   - Verify ConvE batch norm mode

2. **Medium effort** (requires code changes):
   - Implement margin-based loss for TransE
   - Add N3 regularization option for ComplEx
   - Hyperparameter tuning

3. **Requires approval** (architectural changes):
   - Redesign ComplEx to use independent real/imaginary embeddings
   - Consider separating embeddings from R-GCN for each method

Would you like me to proceed with implementing the quick wins first?
