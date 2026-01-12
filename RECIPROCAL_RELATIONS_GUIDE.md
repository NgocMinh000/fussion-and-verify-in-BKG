# 📘 Hướng Dẫn Sử Dụng Reciprocal Relations

## ❓ Reciprocal Relations Là Gì?

Reciprocal relations (quan hệ đảo ngược) là kỹ thuật data augmentation cho Knowledge Graph Embeddings:

**Ví dụ:**
```
Original triple: (DrugA, TREATS, DiseaseX)
Reciprocal triple: (DiseaseX, TREATS_INV, DrugA)
```

Với mỗi triple `(h, r, t)`, ta thêm triple đảo ngược `(t, r_inv, h)`.

## 🎯 Lợi Ích

- **Cải thiện performance**: +3-5% MRR trong nhiều dataset
- **Học được quan hệ hai chiều**: Model học cả chiều xuôi và ngược
- **Tốt cho asymmetric relations**: Như TREATS, CAUSES, AFFECTS

## ⚠️ VẤN ĐỀ QUAN TRỌNG: Iterations vs Epochs

### Hiểu Đúng Về Training

**Epoch** = 1 lần đi qua TOÀN BỘ training data
**Iteration** = 1 lần cập nhật weights (1 mini-batch)

```
Iterations per epoch = Số training samples / Batch size
```

### Vấn Đề Khi Dùng `--use_reciprocal`

Khi bật reciprocal relations:
1. ❌ Training data **GẤP ĐÔI** (244K → 489K triples)
2. ❌ Batch size **GIỮ NGUYÊN** (250)
3. ❌ Iterations per epoch **GẤP ĐÔI** (979 → 1,958)
4. ⚠️ Nếu giữ nguyên số iterations → Model chỉ train được **1/2 số epochs**!

### Ví Dụ Với Data suppkg

#### ❌ SAI: Cùng Số Iterations (Không Công Bằng)

```bash
# Test 1: Không reciprocal
--iterations 40000
→ 40,000 / 979 ≈ 41 epochs ✅
→ Model thấy mỗi triple ~41 lần

# Test 2: Có reciprocal  
--iterations 40000 --use_reciprocal
→ 40,000 / 1,958 ≈ 20 epochs ❌
→ Model chỉ thấy mỗi triple ~20 lần (GIẢM 1 NỬA!)
→ KẾT QUẢ XẤU HƠN LÀ DO CHƯA HỌC ĐỦ!
```

#### ✅ ĐÚNG: Cùng Số Epochs (Công Bằng)

```bash
# Test 1: Không reciprocal
--iterations 39166
→ 39,166 / 979 ≈ 40 epochs ✅

# Test 2: Có reciprocal
--iterations 78332 --use_reciprocal  
→ 78,332 / 1,958 ≈ 40 epochs ✅
→ CẢ HAI TRAIN 40 EPOCHS = SO SÁNH CÔNG BẰNG!
```

## 🧮 Tính Số Iterations Cần Thiết

Sử dụng script `calculate_iterations.py`:

```bash
python calculate_iterations.py
```

Output:
```
WITHOUT --use_reciprocal:
  Training triples: 244,788
  Iterations per epoch: 979.2
  To train 40 epochs: 39,166 iterations

WITH --use_reciprocal:
  Training triples: 489,576 (doubled)
  Iterations per epoch: 1958.3
  To train 40 epochs: 78,332 iterations

Ratio: 2.00x
```

## 📝 Lệnh Chạy Đúng

### ComplEx (Khuyến Nghị)

```bash
conda activate fuselinker
cd ~/fussion-and-verify-in-BKG/fuselinker-complex

# Baseline: Không reciprocal
python main.py \
    --data suppkg \
    --text_embedding_file pubmedbert_pretrained_embeddings_768.npy \
    --knowledge_embedding_file poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --iterations 39166 \
    --evaluate_every 2000 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --use_n3_reg \
    --model_state_file complex_baseline.pth \
    --use_cuda True

# With reciprocal: GẤP ĐÔI iterations
python main.py \
    --data suppkg \
    --text_embedding_file pubmedbert_pretrained_embeddings_768.npy \
    --knowledge_embedding_file poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --iterations 78332 \
    --evaluate_every 4000 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --use_n3_reg \
    --use_reciprocal \
    --model_state_file complex_reciprocal.pth \
    --use_cuda True
```

### DistMult

```bash
cd ~/fussion-and-verify-in-BKG/fuselinker

# Baseline
python main.py \
    --data suppkg \
    --text_embedding_file pubmedbert_pretrained_embeddings_768.npy \
    --knowledge_embedding_file poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --iterations 39166 \
    --evaluate_every 2000 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --model_state_file distmult_baseline.pth \
    --use_cuda True

# With reciprocal
python main.py \
    --data suppkg \
    --text_embedding_file pubmedbert_pretrained_embeddings_768.npy \
    --knowledge_embedding_file poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --iterations 78332 \
    --evaluate_every 4000 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --use_reciprocal \
    --model_state_file distmult_reciprocal.pth \
    --use_cuda True
```

### TransE

```bash
cd ~/fussion-and-verify-in-BKG/fuselinker-transe

# Baseline
python main.py \
    --data suppkg \
    --text_embedding_file pubmedbert_pretrained_embeddings_768.npy \
    --knowledge_embedding_file poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --iterations 39166 \
    --evaluate_every 2000 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --model_state_file transe_baseline.pth \
    --use_cuda True

# With reciprocal
python main.py \
    --data suppkg \
    --text_embedding_file pubmedbert_pretrained_embeddings_768.npy \
    --knowledge_embedding_file poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --iterations 78332 \
    --evaluate_every 4000 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --use_reciprocal \
    --model_state_file transe_reciprocal.pth \
    --use_cuda True
```

### ConvE

```bash
cd ~/fussion-and-verify-in-BKG/fuselinker-conve

# Baseline
python main.py \
    --data suppkg \
    --text_embedding_file pubmedbert_pretrained_embeddings_768.npy \
    --knowledge_embedding_file poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --iterations 39166 \
    --evaluate_every 2000 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --model_state_file conve_baseline.pth \
    --use_cuda True

# With reciprocal
python main.py \
    --data suppkg \
    --text_embedding_file pubmedbert_pretrained_embeddings_768.npy \
    --knowledge_embedding_file poincare_embeddings.npy \
    --num_hidden_layers 2 \
    --iterations 78332 \
    --evaluate_every 4000 \
    --neg_sample_size_eval 100 \
    --w 0.75 \
    --use_reciprocal \
    --model_state_file conve_reciprocal.pth \
    --use_cuda True
```

## 📊 Kỳ Vọng Kết Quả

Sau khi train với **CÙNG SỐ EPOCHS**:

| Method | Baseline MRR | +Reciprocal MRR | Improvement |
|--------|--------------|-----------------|-------------|
| DistMult | ~0.80 | ~0.83 | +3-5% |
| TransE | ~0.82 | ~0.85 | +3-5% |
| ComplEx (N3) | ~0.82 | ~0.86 | +4-6% |
| ConvE | ~0.85 | ~0.88 | +3-5% |

## ⚡ Quick Reference

### Công Thức Tính Nhanh

```
Iterations với reciprocal = Iterations không reciprocal × 2
```

### Rule of Thumb

- **Test nhanh**: Iterations nhỏ (vài nghìn) → không cần điều chỉnh
- **Production**: Luôn tăng GẤP ĐÔI iterations khi dùng reciprocal
- **So sánh**: PHẢI dùng cùng số epochs, không phải cùng iterations

## 🐛 Troubleshooting

### Vấn đề: Reciprocal cho kết quả XẤU HƠN

**Nguyên nhân 1**: Không tăng iterations
```bash
# ❌ SAI
--iterations 4000 --use_reciprocal

# ✅ ĐÚNG  
--iterations 8000 --use_reciprocal
```

**Nguyên nhân 2**: Dataset đã symmetric sẵn
- Một số dataset có relations đã symmetric tự nhiên
- Reciprocal có thể không giúp ích, hoặc thậm chí làm model confused
- Giải pháp: Thử cả hai và so sánh

**Nguyên nhân 3**: Model capacity không đủ
- Reciprocal tăng gấp đôi số relations
- Model cần capacity lớn hơn (tăng hidden_dim hoặc num_bases)

### Vấn đề: Training quá lâu

**Giải pháp**: Sử dụng early stopping
```bash
# Train với validation
--validate_every 2000
# Dừng khi validation MRR không tăng nữa
```

## 📚 Tài Liệu Tham Khảo

- Dettmers et al. "Convolutional 2D Knowledge Graph Embeddings" (2018)
- Lacroix et al. "Canonical Tensor Decomposition for Knowledge Base Completion" (2018)
- Trouillon et al. "Complex Embeddings for Simple Link Prediction" (2016)

## 🎓 Best Practices

1. ✅ **LUÔN train cùng số epochs** khi so sánh
2. ✅ Dùng reciprocal cho **asymmetric relations** (TREATS, CAUSES)
3. ✅ Thử cả **có và không có** reciprocal để xem cái nào tốt hơn
4. ✅ Sử dụng **N3 regularization cho ComplEx** khi dùng reciprocal
5. ✅ Monitor **validation metrics** để tránh overfitting

## 💡 Tips

- Bắt đầu với **iterations nhỏ** để test (vài nghìn)
- Sau khi confirm reciprocal có hiệu quả, train **full iterations**
- Luôn **save model** sau mỗi test để so sánh
- Document **tất cả experiments** với metrics

---

**Tóm Tắt**: Reciprocal relations tăng gấp đôi data → cần tăng gấp đôi iterations để train cùng số epochs! 🚀
