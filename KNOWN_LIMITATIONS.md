# VQA PROJECT - KNOWN LIMITATIONS & FUTURE IMPROVEMENTS

## ✅ FIXED ISSUES (Phase 1 Refactor)

### 1. Model Architecture
- ✅ **CNNEncoder**: Added `return_spatial` parameter for Spatial Attention
  - Now supports both global features (B, C) and spatial features (B, 49, C)
  - Auto-unfreezes from-scratch models (critical fix)
  
- ✅ **LSTMDecoder**: Initialized Cell State from context
  - Added `cell_proj` layer for better memory initialization
  - Both hidden and cell states now properly initialized

- ✅ **VQATrainer**: Scheduled Teacher Forcing
  - Linear decay from 1.0 → 0.5 over training
  - Helps model become more autonomous

---

## ⚠️ KNOWN LIMITATIONS (Require Data Re-generation)

### 1. Color Hallucination Problem
**Location**: `Data_prep/code/scripts/generate_vqa_annotations.py`

**Issue**:
```python
# Current implementation (WRONG):
color = random.choice(['black', 'white', 'brown', 'gray', 'orange'])
```

**Problem**:
- Colors are randomly assigned without looking at actual image
- Model learns spurious correlations (e.g., "dog" → "brown")
- Color recognition accuracy is meaningless

**Solutions**:
1. **Option A**: Use Color Detection (K-Means on image pixels)
   ```python
   from sklearn.cluster import KMeans
   dominant_color = extract_dominant_color(image)
   ```

2. **Option B**: Use CLIP/BLIP for color description
   ```python
   from transformers import BlipProcessor, BlipForConditionalGeneration
   color = blip_model.generate_color_description(image)
   ```

3. **Option C**: Remove color questions entirely
   - Focus on animal recognition, yes/no, counting only

**Impact**: Medium (affects 20,732 color questions = 20.7% of dataset)

---

### 2. Question Diversity Problem
**Location**: All annotation generation scripts

**Issue**:
- Fixed templates: "What animal is in the image?"
- Model overfits to sentence structure, not semantics

**Solutions**:
1. **Paraphrasing with LLM**:
   ```python
   templates = [
       "What animal is in the image?",
       "Which animal can you see?",
       "What type of animal is this?",
       "Identify the animal in this picture.",
       "What kind of creature is shown?"
   ]
   ```

2. **Use T5/GPT for generation**:
   ```python
   from transformers import T5ForConditionalGeneration
   question = t5_model.paraphrase(base_question)
   ```

**Impact**: High (affects model generalization)

---

### 3. Counting Limitation
**Location**: Model architecture (fundamental)

**Issue**:
- Global CNN features cannot count objects accurately
- ResNet/VGG compress spatial information → lose object count

**Why it fails**:
```
Input: Image with 3 dogs
CNN → Global Feature (2048-dim vector)
↓
Information about "3 separate objects" is LOST
↓
Model guesses based on statistics, not actual counting
```

**Solutions**:
1. **Object Detection Features** (Recommended):
   ```python
   # Use Faster R-CNN or YOLO to detect objects
   from torchvision.models.detection import fasterrcnn_resnet50_fpn
   detector = fasterrcnn_resnet50_fpn(pretrained=True)
   boxes, features = detector(image)
   count = len(boxes)  # Actual object count
   ```

2. **Spatial Attention** (Partial solution):
   - Use 7x7 feature maps instead of global pooling
   - Attention can focus on different regions
   - Still not perfect for counting

3. **Accept limitation**:
   - Document that counting is a known weakness
   - Use it to demonstrate CNN limitations

**Impact**: High (counting questions = 17.4% of dataset)

---

## 📊 IMPACT SUMMARY

| Issue | Severity | Affected Data | Fix Difficulty | Recommended Action |
|-------|----------|---------------|----------------|-------------------|
| **Color Hallucination** | Medium | 20.7% | Medium | Use K-Means or remove |
| **Question Diversity** | High | 100% | Easy | Add paraphrasing |
| **Counting Limitation** | High | 17.4% | Hard | Document limitation |

---

## 🎯 RECOMMENDED REFACTOR PRIORITY

### Phase 2 (Optional - Data Quality):
1. **High Priority**: Add question paraphrasing
   - Easy to implement
   - Huge impact on generalization
   
2. **Medium Priority**: Fix color detection
   - Use K-Means clustering
   - Or remove color questions

3. **Low Priority**: Counting
   - Document as known limitation
   - Or implement object detection (major refactor)

---

## 📝 NOTES FOR FUTURE WORK

### If Re-generating Data:
1. Use BLIP/CLIP for better captions
2. Implement proper color detection
3. Add more question templates
4. Consider using VQA v2 dataset instead

### If Keeping Current Data:
1. Document limitations in paper/report
2. Use as ablation study (show what doesn't work)
3. Focus analysis on animal recognition and yes/no questions

---

**Last Updated**: 2026-02-03
**Status**: Phase 1 Refactor Complete ✅
