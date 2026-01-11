# Comprehensive Fix Plan - Phương Án Toàn Diện

## 🔴 ROOT CAUSE ANALYSIS

### Critical Bug Found:
**Evaluation code (`myutils.py` line 220) hardcoded DistMult scoring for ALL methods!**

```python
# Current (WRONG):
emb_triplet = emb_s * emb_r * emb_o  # Distmult scoring!
scores = torch.sigmoid(torch.sum(emb_triplet, dim=1))
```

**Impact:**
- Training uses correct scoring function (TransE/ComplEx/ConvE)
- Evaluation uses DistMult scoring for all methods
- Embeddings trained with one function, evaluated with another
- Result: Terrible metrics despite good training loss

**Why Hits@1=0, Hits@3=0, Hits@10=1.0?**
- Embeddings learned for TransE/ComplEx/ConvE
- Evaluated with wrong (DistMult) scoring function
- Scores completely misaligned
- Some weak correlation → top-10 might be correct by chance

---

## 🎯 COMPREHENSIVE FIX STRATEGY

We need to fix **5 major components**:

### 1. ✅ Evaluation Code (CRITICAL - Root Cause)
### 2. ✅ TransE Implementation (Normalization + Loss)
### 3. ✅ ComplEx Implementation (Independent Embeddings + N3 Reg)
### 4. ✅ ConvE Implementation (Batch Norm + Dropout)
### 5. ✅ Data Pipeline (Reciprocal Relations)

---

## 📦 DETAILED IMPLEMENTATION PLAN

### COMPONENT 1: Fix Evaluation Code

#### Problem:
Evaluation hardcoded for DistMult only.

#### Solution:
Pass model's scoring function to evaluation.

#### Changes Needed:

**File: `myutils.py`**

**Change 1: Modify `perturb_and_get_filtered_rank` signature**
```python
# OLD:
def perturb_and_get_filtered_rank(emb, w, s, r, o, test_size, triplets_to_filter,
                                   neg_sample_size_eval, filter_o=True):

# NEW:
def perturb_and_get_filtered_rank(emb, w, s, r, o, test_size, triplets_to_filter,
                                   neg_sample_size_eval, score_function=None, filter_o=True):
```

**Change 2: Replace hardcoded scoring with function call**
```python
# OLD (line 220-221):
emb_triplet = emb_s * emb_r * emb_o  # Distmult
scores = torch.sigmoid(torch.sum(emb_triplet, dim=1))

# NEW:
if score_function is not None:
    # Call model's scoring function
    # Need to construct fake triplets for scoring
    batch_size = emb_o.shape[0]
    if filter_o:
        # Predicting object: (s, r, ?)
        s_indices = torch.full((batch_size,), target_s, dtype=torch.long)
        r_indices = torch.full((batch_size,), target_r, dtype=torch.long)
        o_indices = torch.tensor(candidate_nodes, dtype=torch.long)
    else:
        # Predicting subject: (?, r, o)
        s_indices = torch.tensor(candidate_nodes, dtype=torch.long)
        r_indices = torch.full((batch_size,), target_r, dtype=torch.long)
        o_indices = torch.full((batch_size,), target_o, dtype=torch.long)

    triplets = torch.stack([s_indices, r_indices, o_indices], dim=1)
    scores = score_function(emb, triplets)
else:
    # Fallback to DistMult (for backward compatibility)
    emb_triplet = emb_s * emb_r * emb_o
    scores = torch.sigmoid(torch.sum(emb_triplet, dim=1))
```

**Change 3: Update `_calc_mrr` to accept and pass score_function**
```python
def _calc_mrr(emb, w, test_triplets, total_data, batch_size, neg_sample_size_eval,
              hits, score_function=None, filter=False):
    # ... existing code ...

    if filter:
        ranks_s = perturb_and_get_filtered_rank(
            emb, w, s, r, o, test_size, triplets_to_filter,
            neg_sample_size_eval, score_function=score_function, filter_o=False
        )
        ranks_o = perturb_and_get_filtered_rank(
            emb, w, s, r, o, test_size, triplets_to_filter,
            neg_sample_size_eval, score_function=score_function
        )
```

**Change 4: Update `calc_mrr` signature**
```python
def calc_mrr(emb, w, test_triplets, total_data, batch_size=100,
             neg_sample_size_eval=20, hits=[1, 3, 10],
             score_function=None, eval_p="filtered"):
```

**File: `main.py` (all variants)**

Update evaluation calls:
```python
# OLD:
mr, mrr, hits_dict = myutils.calc_mrr(
    embeddings, model.relation_weights, test_data,
    total_data, batch_size=500, neg_sample_size_eval=neg_sample_size_eval
)

# NEW:
mr, mrr, hits_dict = myutils.calc_mrr(
    embeddings, model.relation_weights, test_data,
    total_data, batch_size=500, neg_sample_size_eval=neg_sample_size_eval,
    score_function=model.calculate_score  # Pass model's scoring function!
)
```

---

### COMPONENT 2: Fix TransE Implementation

#### Changes to `fuselinker-transe/model.py`:

**Fix 1: Add L2 Normalization to Scoring**
```python
def calculate_score(self, embeddings, triplets):
    """
    TransE scoring with proper L2 normalization.
    f(h,r,t) = -||h + r - t||_p
    """
    h = embeddings[triplets[:, 0]]
    r = self.relation_weights[triplets[:, 1]]
    t = embeddings[triplets[:, 2]]

    # CRITICAL: L2 normalize entity embeddings
    h = F.normalize(h, p=2, dim=1)
    t = F.normalize(t, p=2, dim=1)

    # TransE: -||h + r - t||_1
    score = -torch.norm(h + r - t, p=1, dim=1)
    return score
```

**Fix 2: Add Margin-Based Loss (Optional but Recommended)**
```python
def get_loss(self, graph, embeddings, triplets, labels):
    """
    Margin-based loss for TransE (more principled than BCE).
    """
    # Separate positive and negative samples
    pos_mask = labels == 1
    neg_mask = labels == 0

    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        # Fallback to BCE if batch doesn't have both pos and neg
        score = self.calculate_score(embeddings, triplets)
        prediction_loss = F.binary_cross_entropy_with_logits(score, labels)
    else:
        pos_scores = self.calculate_score(embeddings, triplets[pos_mask])
        neg_scores = self.calculate_score(embeddings, triplets[neg_mask])

        # Margin ranking loss: max(0, γ - score_pos + score_neg)
        margin = 1.0
        loss = torch.mean(F.relu(margin - pos_scores.mean() + neg_scores.mean()))
        prediction_loss = loss

    # Regularization
    reg_loss = self.regularization_loss(embeddings)
    return prediction_loss + self.regularization_param * reg_loss
```

**Fix 3: Normalize Relations on Initialization**
```python
def __init__(self, ...):
    # ... existing code ...

    # Initialize relation embeddings
    self.relation_weights = nn.Parameter(torch.Tensor(num_relations, hidden_dim))
    nn.init.xavier_uniform_(self.relation_weights, gain=nn.init.calculate_gain('relu'))

    # L2 normalize relation embeddings (optional but helps)
    with torch.no_grad():
        self.relation_weights.data = F.normalize(self.relation_weights.data, p=2, dim=1)

    print("Initialized random relation embeddings (L2 normalized).")
```

---

### COMPONENT 3: Fix ComplEx Implementation

#### Problem:
Imaginary parts are derived from real parts via linear transformation. Should be independent parameters.

#### Solution Options:

**Option A: Simplest - Project from R-GCN embeddings**
Current approach but make projection learnable and symmetric.

**Option B: Independent Embeddings (Most Faithful)**
Separate embeddings for real and imaginary, don't use R-GCN for final embeddings.

**Option C: Hybrid (Recommended)**
Use R-GCN as "real" part, learn "imaginary" offset.

#### Recommended: Option C (Hybrid Approach)

**File: `fuselinker-complex/model.py`**

```python
def __init__(self, input_dim, hidden_dim, num_relations, num_bases=-1,
             num_hidden_layers=1, dropout=0.0, use_cuda=False, regularization_param=0.0,
             pretrained_text_embeddings=None, pretrained_domain_embeddings=None,
             pretrained_relation_embeddings=None, freeze=False, w=0.5):
    super(LinkPredict, self).__init__()

    # R-GCN produces "real" embeddings
    self.rgcn = RGCN(input_dim, hidden_dim, hidden_dim, num_relations * 2, num_bases,
                     num_hidden_layers, dropout, use_cuda,
                     pretrained_text_embeddings=pretrained_text_embeddings,
                     pretrained_domain_embeddings=pretrained_domain_embeddings,
                     freeze=freeze, w=w)

    self.regularization_param = regularization_param
    self.hidden_dim = hidden_dim
    self.use_n3_reg = True  # Use N3 regularization for ComplEx

    # Relation embeddings (real and imaginary)
    self.relation_weights = nn.Parameter(torch.Tensor(num_relations, hidden_dim))
    self.relation_weights_imag = nn.Parameter(torch.Tensor(num_relations, hidden_dim))
    nn.init.xavier_uniform_(self.relation_weights, gain=nn.init.calculate_gain('relu'))
    nn.init.xavier_uniform_(self.relation_weights_imag, gain=nn.init.calculate_gain('relu'))

    # CRITICAL FIX: Independent imaginary embeddings for entities
    # Learn as offset from real embeddings (hybrid approach)
    self.entity_embeddings_imag = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
    nn.init.xavier_uniform_(self.entity_embeddings_imag, gain=nn.init.calculate_gain('relu'))

    print("Initialized ComplEx with independent imaginary entity embeddings.")

def calculate_score(self, embeddings, triplets):
    """
    ComplEx scoring with independent imaginary parts.
    f(h,r,t) = Re(<h, r, conj(t)>)
    """
    # Real parts from R-GCN
    h_real = embeddings[triplets[:, 0]]
    r_real = self.relation_weights[triplets[:, 1]]
    t_real = embeddings[triplets[:, 2]]

    # Imaginary parts from independent parameters
    h_imag = self.entity_embeddings_imag[triplets[:, 0]]
    r_imag = self.relation_weights_imag[triplets[:, 1]]
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
    """N3 regularization (nuclear 3-norm) - superior to L2 for ComplEx"""
    factor = 0.0

    # Get imaginary parts for all entities used in current batch
    # (In practice, regularize all parameters)
    factor += torch.mean(torch.abs(self.relation_weights) ** 3)
    factor += torch.mean(torch.abs(self.relation_weights_imag) ** 3)
    factor += torch.mean(torch.abs(self.entity_embeddings_imag) ** 3)

    return factor

def regularization_loss(self, embeddings):
    """Use N3 regularization for ComplEx"""
    if self.use_n3_reg:
        # N3 regularization
        reg = self.n3_regularization()
        # Also add L2 for R-GCN embeddings
        reg += torch.mean(embeddings.pow(2))
    else:
        # Standard L2
        reg = (torch.mean(embeddings.pow(2)) +
               torch.mean(self.relation_weights.pow(2)) +
               torch.mean(self.relation_weights_imag.pow(2)) +
               torch.mean(self.entity_embeddings_imag.pow(2)))
    return reg
```

---

### COMPONENT 4: Fix ConvE Implementation

#### Changes to `fuselinker-conve/model.py`:

**Fix 1: Ensure Proper Batch Norm Mode**

Add a method to switch to eval mode for batch norm layers:

```python
def set_eval_mode_for_inference(self):
    """Set batch norm layers to eval mode during inference"""
    self.bn0.eval()
    self.bn1.eval()
    self.bn2.eval()

def set_train_mode(self):
    """Set batch norm layers back to train mode"""
    self.bn0.train()
    self.bn1.train()
    self.bn2.train()
```

**Fix 2: Reorder Dropout and Batch Norm (if needed)**

Current order is OK, but verify:
```python
# Correct order:
x = self.bn0(stacked)        # 1. Batch Norm
x = self.input_dropout(x)     # 2. Dropout
x = self.conv1(x)             # 3. Convolution
x = self.bn1(x)               # 4. Batch Norm
x = F.relu(x)                 # 5. Activation
x = self.feature_map_dropout(x) # 6. Dropout
```

**Fix 3: Update evaluation in main.py**

```python
# Before evaluation:
model.set_eval_mode_for_inference()  # Force eval mode for batch norm

# Run evaluation
embeddings = model(graph, node_ids, rel_ids, norm)
mr, mrr, hits_dict = myutils.calc_mrr(...)

# After evaluation:
model.set_train_mode()  # Back to train mode
```

---

### COMPONENT 5: Add Reciprocal Relations

#### File: `data_loader.py`

Add method to augment data with reciprocal relations:

```python
def add_reciprocal_relations(data, num_relations):
    """
    For each triple (h, r, t), add reciprocal triple (t, r_inv, h).
    r_inv = r + num_relations

    Args:
        data: numpy array of shape (n, 3) with columns [head, relation, tail]
        num_relations: number of unique relations

    Returns:
        augmented_data: data with reciprocal triples added
        new_num_relations: num_relations * 2
    """
    # Create inverse triples
    inverse_data = data.copy()
    inverse_data[:, 0] = data[:, 2]  # head <- tail
    inverse_data[:, 2] = data[:, 0]  # tail <- head
    inverse_data[:, 1] = data[:, 1] + num_relations  # relation <- relation + offset

    # Concatenate original and inverse
    augmented_data = np.vstack([data, inverse_data])

    return augmented_data, num_relations * 2
```

#### File: `main.py` (all variants)

Apply reciprocal relations before creating knowledge graph:

```python
def main(args):
    # ... load train, valid, test ...

    train = pd.read_csv(train_path, sep='\t', header=None)
    valid = pd.read_csv(valid_path, sep='\t', header=None)
    test = pd.read_csv(test_path, sep='\t', header=None)

    # Convert to numpy for processing
    train_np = train.values
    valid_np = valid.values
    test_np = test.values

    # Add reciprocal relations
    num_relations_original = train_np[:, 1].max() + 1

    train_augmented, num_relations = add_reciprocal_relations(train_np, num_relations_original)
    valid_augmented, _ = add_reciprocal_relations(valid_np, num_relations_original)
    test_augmented, _ = add_reciprocal_relations(test_np, num_relations_original)

    # Convert back to DataFrame
    train = pd.DataFrame(train_augmented)
    valid = pd.DataFrame(valid_augmented)
    test = pd.DataFrame(test_augmented)

    print(f"Added reciprocal relations: {num_relations_original} -> {num_relations} relations")

    # Continue with rest of main()
    graph = pd.concat([train, valid, test])
    # ... rest of code ...
```

**Important:** Also update model instantiation to use doubled number of relations:
```python
model = LinkPredict(
    num_nodes,
    args.n_hidden,
    num_relations,  # Already doubled by add_reciprocal_relations
    # ... other args ...
)
```

---

## 🔧 IMPLEMENTATION ORDER

### Phase 1: Critical Fixes (Must Do First)
1. ✅ Fix evaluation code to accept score_function
2. ✅ Update all main.py to pass model.calculate_score
3. ✅ Test with DistMult first (should give same results as before)

### Phase 2: TransE Fixes
4. ✅ Add L2 normalization to TransE scoring
5. ✅ Test TransE (should improve significantly)
6. ⚙️ (Optional) Add margin-based loss

### Phase 3: Data Augmentation
7. ✅ Add reciprocal relations to data_loader.py
8. ✅ Update main.py to use reciprocal relations
9. ✅ Test all methods (should improve for all)

### Phase 4: ComplEx Fixes
10. ✅ Add independent imaginary embeddings
11. ✅ Add N3 regularization option
12. ✅ Test ComplEx

### Phase 5: ConvE Fixes
13. ✅ Add eval mode control for batch norm
14. ✅ Update evaluation code
15. ✅ Test ConvE

---

## 📊 EXPECTED IMPROVEMENTS

After all fixes:

| Method | Before | After Expected | Improvement |
|--------|--------|----------------|-------------|
| DistMult | MRR ~0.82 | MRR ~0.85 | +3% (reciprocal) |
| TransE | Hits@1=0 | MRR ~0.83 | +100% (from broken) |
| ComplEx | MRR ~0.20? | MRR ~0.86 | +320% |
| ConvE | MRR ~0.20? | MRR ~0.88 | +340% |

---

## ⚠️ BREAKING CHANGES

1. **Model checkpoints incompatible**: ComplEx architecture changes require retraining
2. **Data format changes**: Reciprocal relations double the number of relations
3. **Evaluation API changes**: All eval calls need score_function parameter

---

## 🎯 TESTING STRATEGY

### Test 1: Verify Evaluation Fix
```bash
# Should give SAME results as before for DistMult
python main.py --data suppkg --iterations 100 ...
```

### Test 2: Verify TransE Fix
```bash
# Should give Hits@1 > 0 now!
cd fuselinker-transe
python main.py --data suppkg --iterations 100 ...
```

### Test 3: Full Comparison
After all fixes, run all 4 methods and compare.

---

## 📝 DELIVERABLES

After implementation:
1. ✅ Fixed evaluation code in myutils.py
2. ✅ Fixed TransE with normalization
3. ✅ Fixed ComplEx with independent embeddings
4. ✅ Fixed ConvE with batch norm control
5. ✅ Data loader with reciprocal relations
6. ✅ Updated all main.py files
7. ✅ Testing scripts
8. ✅ Results comparison table

---

## 🚀 READY TO IMPLEMENT?

This is a comprehensive solution that fixes ALL identified issues:
- ✅ Root cause (evaluation bug)
- ✅ TransE normalization
- ✅ ComplEx architecture
- ✅ ConvE batch norm
- ✅ Reciprocal relations
- ✅ All best practices from research

Estimated time: 2-3 hours for full implementation and testing.

Approve to proceed?
