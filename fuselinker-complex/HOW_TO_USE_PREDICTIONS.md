# 🔗 Cách Sử Dụng Link Prediction - Xem Các Liên Kết Đã Được Fuse

## ❓ Vấn Đề

Sau khi train model với:
```bash
python main.py --data suppkg \
    --text_embedding_file pubmedbert_pretrained_embeddings_768.npy \
    --knowledge_embedding_file poincare_embeddings.npy \
    --num_hidden_layers 2 --iterations 40000 --w 0.75 \
    --model_state_file suppkg_model_state.pth
```

Bạn có file `suppkg_model_state.pth` nhưng **chưa biết liên kết nào đã được fuse/predict**.

## ✅ Giải Pháp: Sử Dụng Link Predictor

---

## 🚀 Cách 1: Quick Script (Khuyến Nghị)

### Bước 1: Run Prediction Script

```bash
cd fuselinker

python predict_new_links.py \
    --model suppkg_model_state.pth \
    --data suppkg \
    --top_k 100 \
    --min_score 0.7 \
    --output predicted_links.csv
```

### Output Bạn Sẽ Thấy:

```
============================================================
FUSELINKER LINK PREDICTION
============================================================
Model: suppkg_model_state.pth
Data: suppkg
Top K per relation: 100
Min score: 0.7
Output: predicted_links.csv
============================================================

Loading FuseLinker Model for Link Prediction
============================================================

[1/5] Loading data...
  ✓ Entities: 9000
  ✓ Relations: 15
  ✓ Existing links: 305986

[2/5] Loading embeddings...
  ✓ Text embeddings: (9000, 768)
  ✓ Domain embeddings: (9000, 50)

[3/5] Initializing model...
[4/5] Loading trained weights...
  ✓ Loaded from iteration 40000

[5/5] Generating embeddings...
  ✓ Embeddings shape: torch.Size([9000, 200])
  ✓ Relation weights shape: torch.Size([15, 200])

============================================================
✓ Model loaded successfully!
============================================================

[Relation 1/15]
Predicting links for relation 0...
100%|████████████████████| 9000/9000

[Relation 2/15]
...

============================================================
✓ Generated 1500 new link predictions
============================================================

============================================================
TOP 10 PREDICTED LINKS (New Fused Links)
============================================================

[1] Score: 0.9234
    C0003250_aapp (aapp)
      --[INTERACTS_WITH]-->
    C0011503_nnon (nnon)

[2] Score: 0.9156
    C0035179_orch (orch)
      --[TREATS]-->
    C0233488_dsyn (dsyn)

...

============================================================
✓ Exported 1500 predictions to predicted_links.csv
============================================================
```

### Bước 2: Xem Results

File `predicted_links.csv` sẽ có format:

| subject | relation | object | score | subject_type | object_type |
|---------|----------|--------|-------|--------------|-------------|
| C0003250_aapp | INTERACTS_WITH | C0011503_nnon | 0.9234 | aapp | nnon |
| C0035179_orch | TREATS | C0233488_dsyn | 0.9156 | orch | dsyn |
| ... | ... | ... | ... | ... | ... |

---

## 🎯 Cách 2: Python Script (Flexible)

### Script Đơn Giản

```python
from visualization.link_predictor import LinkPredictor

# Initialize predictor
predictor = LinkPredictor()

# Load trained model
predictor.load_model(
    model_state_path='suppkg_model_state.pth',
    data_dir='suppkg',
    text_embedding_path='suppkg/pubmedbert_pretrained_embeddings_768.npy',
    knowledge_embedding_path='suppkg/poincare_embeddings.npy',
    n_hidden=200,
    num_hidden_layers=2,
    w=0.75
)

# Predict new links
predictions = predictor.predict_new_links(
    top_k_per_relation=100,  # Top 100 per relation
    min_score=0.7            # Only predictions with score >= 0.7
)

# Export
predictor.export_predictions('predicted_links.csv')

# Print top 10
predictor.print_top_predictions(10)
```

### Advanced Usage - Filter Predictions

```python
# Filter by semantic types (chỉ diseases và drugs)
disease_drug_links = predictor.filter_predictions(
    semantic_types=['dsyn', 'phsu'],
    relations=['TREATS', 'PREVENTS'],
    min_score=0.8
)

print(f"Found {len(disease_drug_links)} disease-drug treatment links")
disease_drug_links.to_csv('disease_drug_predictions.csv', index=False)
```

### Analyze Predictions

```python
# Get statistics
stats = predictor.analyze_predictions()

print(f"Total predictions: {stats['total_predictions']}")
print(f"Average score: {stats['score_mean']:.4f}")
print(f"\nTop relations:")
for relation, count in stats['relations'].items():
    print(f"  {relation}: {count} predictions")
```

---

## 🎨 Cách 3: Visualize Predicted Links

### Visualize với Graph Visualizer

```python
from visualization.graph_visualizer import GraphVisualizer
import pandas as pd

# Load predictions
predictions = pd.read_csv('predicted_links.csv')

# Get top 100 predictions
top_predictions = predictions.head(100)

# Convert to graph format
viz = GraphVisualizer()
viz.load_from_dataframe(top_predictions)

# Customize
viz.filter_by_semantic_types(['dsyn', 'phsu'])  # diseases and drugs
viz.sample_nodes(200)

# Visualize
viz.create_pyvis_network(layout='hierarchical')
viz.show('predicted_links_graph.html')
```

### Compare Existing vs Predicted Links

```python
from visualization.link_predictor import LinkPredictor
from visualization.graph_visualizer import GraphVisualizer

# Load predictions
predictor = LinkPredictor()
predictor.load_model(...)
new_links = predictor.predict_new_links(top_k_per_relation=50, min_score=0.8)

# Visualize existing links
viz_existing = GraphVisualizer()
viz_existing.load_from_json('suppkg/visualization_outputs/train_graph.json')
viz_existing.sample_nodes(300)
viz_existing.create_pyvis_network()
viz_existing.show('existing_links.html')

# Visualize predicted links
viz_predicted = GraphVisualizer()
viz_predicted.load_from_dataframe(new_links)
viz_predicted.sample_nodes(300)
viz_predicted.create_pyvis_network()
viz_predicted.show('predicted_links.html')

print("Compare the two HTML files to see existing vs predicted links!")
```

---

## 📊 Use Cases

### Use Case 1: Tìm Drug Candidates cho Disease

```python
predictor = LinkPredictor()
predictor.load_model(...)

# Predict all links
predictions = predictor.predict_new_links(top_k_per_relation=200)

# Filter: diseases và drugs với TREATS relation
drug_treatments = predictor.filter_predictions(
    semantic_types=['dsyn', 'phsu'],
    relations=['TREATS'],
    min_score=0.85
)

# Tìm drugs cho một disease cụ thể
diabetes_drugs = drug_treatments[
    drug_treatments['subject'].str.contains('C0011849')  # Diabetes
]

print(f"Predicted drugs for diabetes: {len(diabetes_drugs)}")
print(diabetes_drugs[['object', 'score']].head(10))
```

### Use Case 2: Discover New Protein Interactions

```python
# Filter protein-protein interactions
protein_interactions = predictor.filter_predictions(
    semantic_types=['aapp'],  # proteins
    relations=['INTERACTS_WITH'],
    min_score=0.9
)

print(f"Discovered {len(protein_interactions)} new protein interactions")
```

### Use Case 3: Analyze Prediction Confidence

```python
import matplotlib.pyplot as plt

# Score distribution
predictions = predictor.predictions

plt.figure(figsize=(10, 6))
plt.hist(predictions['score'], bins=50, alpha=0.7)
plt.xlabel('Confidence Score')
plt.ylabel('Number of Predictions')
plt.title('Prediction Score Distribution')
plt.savefig('score_distribution.png')

# Predictions per relation
relation_counts = predictions['relation'].value_counts()
plt.figure(figsize=(12, 6))
relation_counts.plot(kind='bar')
plt.xlabel('Relation Type')
plt.ylabel('Number of Predictions')
plt.title('Predictions by Relation Type')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('predictions_by_relation.png')
```

---

## 🔍 Hiểu Output

### Predicted Link Format

Mỗi predicted link có:

1. **subject**: Entity nguồn (e.g., "C0003250_aapp")
2. **relation**: Loại quan hệ (e.g., "TREATS")
3. **object**: Entity đích (e.g., "C0011503_nnon")
4. **score**: Confidence score (0-1, càng cao càng confident)
5. **subject_type**: Semantic type của subject (e.g., "aapp" = protein)
6. **object_type**: Semantic type của object (e.g., "dsyn" = disease)

### Score Interpretation

- **0.9 - 1.0**: Very high confidence - rất có khả năng là true positive
- **0.8 - 0.9**: High confidence - có thể tin tưởng
- **0.7 - 0.8**: Moderate confidence - cần validation
- **< 0.7**: Low confidence - không khuyến nghị sử dụng

### Semantic Types

- **dsyn**: Disease or Syndrome
- **phsu**: Pharmacologic Substance (Drug)
- **aapp**: Amino Acid, Peptide, or Protein
- **gngm**: Gene or Genome
- **orch**: Organic Chemical
- **nnon**: Nucleic Acid/Nucleoside/Nucleotide
- **bacs**: Biologically Active Substance
- **sosy**: Sign or Symptom

---

## ⚙️ Advanced Options

### Custom Prediction Parameters

```bash
python predict_new_links.py \
    --model suppkg_model_state.pth \
    --data suppkg \
    --top_k 200 \           # More predictions per relation
    --min_score 0.85 \      # Higher threshold
    --filter_semantic_types dsyn phsu \  # Only diseases and drugs
    --filter_relations TREATS PREVENTS \ # Only treatment relations
    --output high_confidence_treatments.csv
```

### Batch Prediction for Specific Relations

```python
# Predict only for specific relations
predictor = LinkPredictor()
predictor.load_model(...)

# Get relation indices
treats_idx = None
for idx, rel in predictor.index2relation.items():
    if 'TREATS' in str(rel):
        treats_idx = idx
        break

# Predict only TREATS relation
treats_predictions = predictor.predict_links_for_relation(
    relation_idx=treats_idx,
    top_k=500,
    min_score=0.8
)

print(f"Found {len(treats_predictions)} TREATS predictions")
```

---

## 🐛 Troubleshooting

### Problem: "Model not loaded"
**Solution**: Call `predictor.load_model()` before predicting

### Problem: Prediction quá chậm
**Solution**:
- Giảm `top_k_per_relation`
- Tăng `min_score` threshold
- Predict cho specific relations thay vì all relations

### Problem: Memory error
**Solution**:
- Run trên GPU nếu có
- Giảm batch size trong code
- Predict từng relation một

### Problem: Scores đều thấp
**Solution**:
- Check model đã train đủ iterations chưa
- Verify model parameters match với lúc training
- Kiểm tra embeddings có load đúng không

---

## 📈 What's Next?

After getting predictions:

1. **Validate**: So sánh với domain knowledge
2. **Visualize**: Tạo interactive graphs
3. **Analyze**: Statistical analysis of patterns
4. **Export**: Share với team hoặc publish
5. **Iterate**: Retrain model với feedback

---

## 🎯 Summary

**Để xem liên kết đã được fuse:**

1. ✅ Train model → có `suppkg_model_state.pth`
2. ✅ Run `python predict_new_links.py --model suppkg_model_state.pth --data suppkg`
3. ✅ Xem file `predicted_links.csv` - đây là các **NEW FUSED LINKS**
4. ✅ Filter, visualize, analyze theo nhu cầu

**Key Points:**
- Predicted links = liên kết CHƯA có trong train/test data
- Score cao = model confident link này có khả năng true
- File CSV chứa tất cả predictions, có thể filter thêm
- Có thể visualize để explore patterns

---

**Happy Predicting! 🚀🔗**
