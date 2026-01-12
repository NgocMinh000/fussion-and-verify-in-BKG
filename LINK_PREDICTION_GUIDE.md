# 🔗 Link Prediction & Explainability Guide

## Tổng Quan

Hệ thống visualization và explainability hoàn chỉnh để:
- **Trích xuất** tất cả link predictions từ model đã train
- **Visualize** predicted triples với confidence scores
- **Giải thích** tại sao model predict được (explainability)
- **Explore** predictions interactively qua dashboard

---

## 🎯 Mục Đích

Chứng minh model đã link predict được bằng cách:
1. ✅ Hiển thị top-K predictions cho mỗi test query
2. ✅ So sánh predictions với ground truth
3. ✅ Phân tích embedding similarities
4. ✅ Tìm training patterns mà model học được
5. ✅ Giải thích reasoning của model

---

## 🚀 Quick Start

### Bước 1: Train Model (Đã Xong!)

Model ComplEx của bạn đã train và export visualization data:
```
suppkg/visualization_outputs/
├── entity_embeddings.npy
├── relation_embeddings.npy
├── graph_structure.json (chứa train/test triples)
└── ...
```

### Bước 2: Extract Link Predictions

```bash
conda activate fuselinker
cd ~/fussion-and-verify-in-BKG

# Extract predictions từ test set
python -m visualization.link_predictor \
    --viz_dir suppkg/visualization_outputs \
    --output_file suppkg/link_predictions.json \
    --num_queries 100 \
    --top_k 10 \
    --model_type complex \
    --show_samples 5
```

**Output:**
- File JSON với tất cả predictions
- Statistics (Hits@1, Hits@3, Hits@10, MRR)
- Sample predictions được print ra console

### Bước 3: Generate Explanations

```bash
# Tạo explanations cho predictions
python -m visualization.explainer \
    --prediction_file suppkg/link_predictions.json \
    --viz_dir suppkg/visualization_outputs \
    --output_file suppkg/prediction_explanations.json \
    --num_queries 10
```

**Output:**
- Embedding similarity analysis
- Training pattern analysis
- Nearest neighbors
- Similar query patterns

### Bước 4: Visualize Predictions

```bash
# Tạo static plots
python -m visualization.prediction_visualizer \
    --prediction_file suppkg/link_predictions.json \
    --output_dir suppkg/prediction_plots \
    --num_samples 10
```

**Output plots:**
- `rank_distribution.png` - Distribution of true answer ranks
- `score_vs_rank.png` - Prediction scores vs ranks
- `query_1_predictions.png`, `query_2_predictions.png`, ... - Individual query predictions
- `query_1_similarities.png`, ... - Similarity heatmaps
- `prediction_summary.txt` - Text summary report

### Bước 5: Interactive Dashboard

```bash
# Launch dashboard
streamlit run visualization/prediction_app.py
```

**Dashboard features:**
- Browse all queries và predictions
- Xem explanations chi tiết
- Interactive filtering và exploration
- Compare predictions across queries

---

## 📊 Output Examples

### Link Predictions JSON Format

```json
{
  "statistics": {
    "total_queries": 100,
    "hits_at_1": 0.72,
    "hits_at_3": 0.85,
    "hits_at_10": 0.93,
    "mrr": 0.78,
    "average_true_rank": 2.3
  },
  "predictions": [
    {
      "query": {
        "subject": {"idx": 45, "name": "Aspirin"},
        "relation": {"idx": 2, "name": "TREATS"},
        "true_object": {"idx": 120, "name": "Headache"},
        "true_object_rank": 1,
        "true_object_score": 0.94
      },
      "predictions": [
        {"rank": 1, "entity_name": "Headache", "score": 0.94, "is_correct": true},
        {"rank": 2, "entity_name": "Pain", "score": 0.89, "is_correct": false},
        {"rank": 3, "entity_name": "Fever", "score": 0.85, "is_correct": false},
        ...
      ]
    },
    ...
  ]
}
```

### Explanation JSON Format

```json
[
  {
    "query": {...},
    "embedding_analysis": [
      {
        "rank": 1,
        "entity_name": "Headache",
        "score": 0.94,
        "similarities": {
          "subject_object": 0.82,
          "subject_relation": 0.75,
          "relation_object": 0.88,
          "combined_alignment": 0.91
        },
        "reasoning": [
          "Strong alignment between (subject ⊙ relation) and object (similarity: 0.91)",
          "Object strongly associated with this relation (similarity: 0.88)"
        ]
      },
      ...
    ],
    "training_patterns": [...],
    "nearest_neighbors": [...],
    "similar_queries": [...]
  },
  ...
]
```

---

## 🔍 Explainability Features

### 1. Embedding Similarity Analysis

**Giải thích:** Tại sao prediction có score cao?

Phân tích:
- **Subject ↔ Object similarity**: Hai entities có giống nhau không?
- **Subject ↔ Relation similarity**: Subject có phù hợp với relation không?
- **Relation ↔ Object similarity**: Object có thường xuất hiện với relation này không?
- **Combined alignment**: `(subject ⊙ relation)` có align với object không? (DistMult scoring)

**Example reasoning:**
- "Strong alignment between (subject ⊙ relation) and object (similarity: 0.91)"
- "Subject and object are very similar entities (similarity: 0.82)"
- "Object strongly associated with this relation (similarity: 0.88)"

### 2. Training Pattern Analysis

**Giải thích:** Model học được gì từ training data?

Phân tích:
- Triple này có trong training không?
- Object này xuất hiện với relation này bao nhiêu lần?
- Subject đã thấy relation này chưa?
- Objects nào thường xuất hiện với relation này?

**Example patterns:**
- "✓ This exact triple was in training data"
- "This object appears with this relation 15 times in training (23.4% of all triples with this relation)"
- "Subject has 8 training examples with this relation"
- "This object never appeared with this relation in training (pure generalization)"

### 3. Nearest Neighbors

**Giải thích:** Entities nào tương tự trong embedding space?

Tìm top-5 entities gần nhất với subject/object trong embedding space.

**Use case:** Hiểu model nhóm entities như thế nào.

### 4. Similar Queries

**Giải thích:** Model generalize từ patterns nào?

Tìm (subject, relation) pairs tương tự trong training data.

**Use case:** Xem model có học được patterns tổng quát không.

---

## 📈 Visualizations Generated

### 1. Rank Distribution
**File:** `rank_distribution.png`

Bar chart hiển thị:
- Bao nhiêu queries có true answer ở rank 1 (Hits@1)
- Bao nhiêu queries ở rank 2-3 (Hits@3)
- Bao nhiêu queries ở rank 4-10 (Hits@10)
- Bao nhiêu queries > rank 10

**Colors:**
- Green: Rank 1 (perfect)
- Yellow: Rank 2-3 (good)
- Blue: Rank 4-10 (okay)
- Red: Rank >10 (poor)

### 2. Score vs Rank
**File:** `score_vs_rank.png`

Scatter plot:
- X-axis: True answer rank (log scale)
- Y-axis: Prediction score
- Points colored by rank

**Insight:** Scores cao thường correspond với ranks thấp (tốt).

### 3. Individual Query Predictions
**Files:** `query_1_predictions.png`, `query_2_predictions.png`, ...

Horizontal bar chart cho mỗi query:
- Top-10 predictions với scores
- Correct answer highlighted in green
- Other predictions in blue

**Shows:** Chi tiết predictions cho từng test query.

### 4. Similarity Heatmaps
**Files:** `query_1_similarities.png`, ...

Heatmap of embedding similarities:
- Rows: Similarity types (Subject↔Object, Subject↔Relation, etc.)
- Columns: Top-5 predicted entities
- Colors: Red (low) → Yellow → Green (high)

**Shows:** Tại sao predictions có scores cao.

### 5. Summary Report
**File:** `prediction_summary.txt`

Text file with:
- Overall statistics
- Sample predictions (top 10 queries)
- Detailed breakdown

---

## 🎨 Interactive Dashboard Features

### Statistics Tab
- Overall metrics (Hits@1, Hits@3, Hits@10, MRR)
- Rank distribution histogram
- Score distribution

### Query Explorer Tab
- Select any query from dropdown
- View top-K predictions with scores
- See true answer rank và score
- Interactive bar chart
- Detailed table with all predictions

### Explanation Tab (if available)
**4 sub-tabs:**

1. **Embedding Similarity**
   - Interactive heatmap
   - Similarity scores
   - Reasoning for each prediction

2. **Training Patterns**
   - Training statistics
   - Most common objects
   - Pattern analysis

3. **Nearest Neighbors**
   - Top-5 similar entities
   - Similarity scores

4. **Similar Queries**
   - Similar training patterns
   - Similarity scores

### Comparison Tab
- Score distribution across all queries
- Rank distribution
- Score vs Rank scatter plot

---

## 🔧 Advanced Usage

### Extract More Queries

```bash
# Analyze 500 test queries
python -m visualization.link_predictor \
    --num_queries 500 \
    --top_k 20
```

### Different Model Types

```bash
# For DistMult
python -m visualization.link_predictor --model_type distmult

# For TransE
python -m visualization.link_predictor --model_type transe

# For ComplEx
python -m visualization.link_predictor --model_type complex

# For ConvE
python -m visualization.link_predictor --model_type conve
```

### Explain Specific Queries

```bash
# Generate explanations for 20 queries
python -m visualization.explainer --num_queries 20
```

### Custom Output Paths

```bash
# Custom output paths
python -m visualization.link_predictor \
    --viz_dir path/to/viz_outputs \
    --output_file path/to/predictions.json

python -m visualization.explainer \
    --prediction_file path/to/predictions.json \
    --output_file path/to/explanations.json

python -m visualization.prediction_visualizer \
    --prediction_file path/to/predictions.json \
    --output_dir path/to/plots
```

---

## 💡 Use Cases

### Use Case 1: Chứng Minh Model Hoạt Động

**Goal:** Show model đã link predict được

**Steps:**
1. Extract predictions → `link_predictions.json`
2. Check Hits@1, Hits@3, Hits@10 → Should be high
3. Visualize → `rank_distribution.png` shows many rank-1 predictions
4. Show sample predictions → Green bars (correct) in top positions

**Result:** Model predicts correctly!

### Use Case 2: Hiểu Model Reasoning

**Goal:** Giải thích tại sao model predict được

**Steps:**
1. Generate explanations → `prediction_explanations.json`
2. Check embedding similarities → High combined alignment
3. Check training patterns → Model learned from similar examples
4. Check nearest neighbors → Subject is similar to other drugs

**Result:** Model generalizes from training patterns!

### Use Case 3: Debug Poor Predictions

**Goal:** Tìm hiểu tại sao một số predictions sai

**Steps:**
1. Filter queries with rank >10
2. Check embedding similarities → Low combined alignment?
3. Check training patterns → Subject never seen with this relation?
4. Check nearest neighbors → Subject is outlier?

**Result:** Identify weaknesses (e.g., rare relations, outlier entities)

### Use Case 4: Compare Models

**Goal:** So sánh DistMult vs ComplEx vs TransE vs ConvE

**Steps:**
1. Extract predictions for each model
2. Compare Hits@1, Hits@3, Hits@10, MRR
3. Visualize rank distributions
4. Check which model predicts which query types better

**Result:** Pick best model for production!

---

## 📋 Checklist

- [ ] Model đã train và export visualization data
- [ ] Extract predictions: `python -m visualization.link_predictor`
- [ ] Generate explanations: `python -m visualization.explainer`
- [ ] Create visualizations: `python -m visualization.prediction_visualizer`
- [ ] Launch dashboard: `streamlit run visualization/prediction_app.py`
- [ ] Review predictions và explanations
- [ ] Check Hits@1, Hits@3, Hits@10, MRR
- [ ] Understand model reasoning

---

## 🐛 Troubleshooting

### "File not found: suppkg/visualization_outputs"
→ Model chưa train. Run training với visualization export enabled.

### "No module named 'sklearn'"
```bash
pip install scikit-learn
```

### "Predictions seem random"
→ Check model type parameter matches your trained model (distmult/transe/complex/conve).

### Dashboard không load
```bash
# Install dependencies
pip install streamlit plotly

# Check files exist
ls suppkg/link_predictions.json
ls suppkg/prediction_explanations.json
```

### Explanations quá chậm
→ Reduce `--num_queries` to 5-10 for faster generation.

---

## 🎯 Expected Results

### Good Model (like your ComplEx: MRR 0.617)

**Predictions:**
- Hits@1: ~45-50%
- Hits@3: ~70-75%
- Hits@10: ~90-92%
- MRR: ~0.60-0.62

**Explanations:**
- High combined alignment (>0.7) for top predictions
- Training patterns show object frequently appears with relation
- Nearest neighbors make semantic sense
- Similar queries exist in training data

### Poor Model (needs improvement)

**Predictions:**
- Hits@1: <20%
- Hits@3: <40%
- Hits@10: <70%
- MRR: <0.40

**Explanations:**
- Low combined alignment (<0.5)
- Training patterns sparse or absent
- Nearest neighbors don't make sense
- No similar queries in training data

---

## 📚 Summary

**✅ Đã tạo:**
- `visualization/link_predictor.py` - Extract predictions
- `visualization/explainer.py` - Generate explanations
- `visualization/prediction_visualizer.py` - Create plots
- `visualization/prediction_app.py` - Interactive dashboard

**✅ Có thể làm:**
- Trích xuất tất cả link predictions
- Visualize predicted triples
- Giải thích tại sao model predict được
- Explore predictions interactively
- So sánh models

**✅ Chứng minh:**
- Model đã link predict được (Hits@1, Hits@3, Hits@10)
- Model có reasoning logic (embedding similarities)
- Model học được patterns (training analysis)
- Model generalize được (similar queries)

**🚀 Bước tiếp theo:**

```bash
# Extract và visualize predictions ngay!
conda activate fuselinker
cd ~/fussion-and-verify-in-BKG

python -m visualization.link_predictor \
    --viz_dir suppkg/visualization_outputs \
    --num_queries 100 \
    --top_k 10 \
    --model_type complex

python -m visualization.explainer \
    --prediction_file suppkg/link_predictions.json \
    --num_queries 10

python -m visualization.prediction_visualizer \
    --prediction_file suppkg/link_predictions.json

streamlit run visualization/prediction_app.py
```

**🎊 Hoàn thành!**
