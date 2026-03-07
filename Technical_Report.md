# Technical Report: Visual Question Answering on Animal Datasets

## 1. Introduction

Visual Question Answering (VQA) is one of the most challenging multimodal learning tasks in the field of Artificial Intelligence, as it requires a system to simultaneously comprehend visual representations and linguistic semantics to generate an accurate text-based answer.

While architectural advancements have greatly pushed the capabilities of VQA systems, the role of data distribution remains a critical factor in model generalization. Models trained on constrained, template-based datasets often exhibit strong performance within their narrow domain but fail drastically when confronted with the linguistic complexities of human-authored queries. To empirically investigate this disparity, this research narrows the VQA problem to a specific domain: querying properties related to animals (e.g., species recognition, color identification, object counting, and polar yes/no questions).

We systematically evaluate our models across two starkly contrasting data distributions:
1. **A Synthetic Template-Based Dataset (Dataset A)**: Consisting of over 100,000 syntactically constrained and artificially balanced question-answer pairs.
2. **An Unconstrained Natural Language Dataset (Dataset B)**: A subset of ~54,000 highly imbalanced, human-annotated queries extracted directly from the real-world VQA v2.0 benchmark.

### Research Objectives

The research is guided by three primary objectives:

**1. Architectural Evaluation:**  
To quantify the performance variance between a baseline Convolutional Neural Network and Long Short-Term Memory (CNN–LSTM) architecture, and an advanced variant integrated with a Spatial Attention mechanism.

**2. Transfer Learning Analysis:**  
To isolate and measure the impact of utilizing a global feature extractor (ResNet50 pretrained on ImageNet) versus forcing localized pixel-level optimization (training the CNN from scratch).

**3. Generalization vs. Inductive Bias:**  
To observe the network's degradation and generalization limits when transitioning from a predictable grammatical structure (Dataset A) to the chaotic, long-tail class distributions of human linguistics (Dataset B).

---

## 2. Theoretical Framework and Methods

### 2.1 Comprehensive Model Architecture

The proposed VQA system is formally defined as a mapping function from an image-question pair $(I, Q)$ to an answer sequence $A = (a_1, a_2, ..., a_T)$. The architecture consists of three highly coupled sub-networks: a Convolutional Neural Network (CNN) for spatial feature extraction, a Long Short-Term Memory (LSTM) network for sequential language formulation, and an Autoregressive Decoder augmented with a Spatial Attention mechanism.

#### A. Visual Feature Extraction (CNN Encoder)
The visual pipeline utilizes dual Convolutional Neural Network constructs depending on the experimental variant: a standard **ResNet50 (Residual Networks)** and a lightweight custom variant denoted as **ScratchCNN**. 

- **Pretrained Variants (Model 1 & 2):** We deploy the standard parameter-heavy ResNet50 architecture (using `Bottleneck` blocks) pretrained on ImageNet.
- **From-Scratch Variants (Model 3 & 4):** Training a complete ResNet50 from scratch mandates exorbitant computational overhead. Thus, we engineered **ScratchCNN**—a lightweight ResNet-style architecture built utilizing 2-layer `BasicBlock` modules (analogous to ResNet18). This custom module compresses the parameter space from ~23M down to ~11M, ensuring viable and rapid baseline optimization from random initialization.

Given an input image $I \in \mathbb{R}^{3 \times H_0 \times W_0}$, the CNN acts as a parameterized projection function $f_{cnn}$. Through both architectures, we extract the spatial activation maps directly from the terminal convolutional blocks, yielding a 3D Spatial Feature Map $V$:
$$ V = f_{cnn}(I) \in \mathbb{R}^{C \times H \times W} $$
where the channel depth sets to $C=2048$ for ResNet50 and $C=512$ for ScratchCNN. The spatial grid remains fixed at $H=7, W=7$. This tensor is unrolled along its spatial dimensions to form a sequence of $L = H \times W = 49$ region representations:
$$ V = [v_1, v_2, ..., v_L] $$
Here, each vector $v_i$ encapsulates the high-level semantic features of the $i$-th specific localized patch of the image.

#### B. Semantic Question Representation (LSTM Encoder)
The natural language question $Q = (w_1, w_2, ..., w_N)$ iteratively undergoes tokenization, where $N$ denotes the maximum sequence length. Each token $w_t$ is mapped into a dense low-dimensional embedding vector $e_t \in \mathbb{R}^{d_{emb}}$.

An LSTM network processes this linguistic sequence to capture long-range syntax dependencies and contextual structure. The standard LSTM gating mechanism regulates information flow at sequence step $t$:
$$ h^{q}_{t}, c^{q}_{t} = \text{LSTM}_{enc}(e_t, h^{q}_{t-1}, c^{q}_{t-1}) $$
The final hidden state vector $q = h^{q}_{N} \in \mathbb{R}^{d_q}$ acts as the **Global Sentence Context Vector**, encapsulating the entire semantic intent of the question.

#### C. Autoregressive Answer Generation (LSTM Decoder & Spatial Attention)
The decoder is constructed as a conditional language model maximizing the joint probability $\prod P(A | I, Q)$. 

In variants outfitted with **Spatial Attention**, the decoder dynamically computes an intermediate context vector at every generative step $t$. Let $h^{d}_{t-1}$ denote the decoder's previous hidden state. The alignment model computes an attention distribution $\alpha_{t} \in \mathbb{R}^L$ over the $L$ distinct image regions using a Multi-Layer Perceptron (MLP) scoring function:
$$ z_{t,i} = W_{att}^T \tanh(W_v v_i + W_h h^{d}_{t-1} + W_q q) $$
$$ \alpha_{t, i} = \text{softmax}(z_{t,i}) = \frac{\exp(z_{t,i})}{\sum_{j=1}^L \exp(z_{t,j})} $$
The attended visual context vector $c_t$ is calculated via the convex combination of all regional vectors weighted by their respective probability coefficients:
$$ c_t = \sum_{i=1}^L \alpha_{t, i} v_i $$
*Theoretical Implication:* This soft-attention mechanism forces the decoder to actively fixate (allocate higher probability masses) on explicitly localized visual boundaries corresponding to the spatial semantics explicitly requested by the question.

The internal state of the decoder LSTM is subsequently updated using the concatenation of the previous output word embedding $y_{t-1}$, the attended dynamic context $c_t$, and the static global question vector $q$:
$$ h^{d}_{t}, c^{d}_{t} = \text{LSTM}_{dec}([y_{t-1} \parallel c_t \parallel q], h^{d}_{t-1}, c^{d}_{t-1}) $$
Ultimately, a linear projection parameterized by $W_{out}$ followed by a softmax function predicts the probability mass over the fixed distinct vocabulary space $V_A$:
$$ P(a_t | a_{<t}, I, Q) = \text{softmax}(W_{out} h^{d}_{t} + b_{out}) $$

### 2.2 Experimental Configurations

Four distinct model pipelines were systematically isolated and trained to deduce algorithmic effectiveness:

| ID | Visual Extractor | Initial Weights | Context Routing |
| :--- | :--- | :--- | :--- |
| **Model 1** | ResNet50 | ImageNet Pretrained | Static Global Features |
| **Model 2** | ResNet50 | ImageNet Pretrained | **Spatial Attention ($c_t$)** |
| **Model 3** | ResNet50 | Random Init (From-Scratch) | Static Global Features |
| **Model 4** | ResNet50 | Random Init (From-Scratch) | **Spatial Attention ($c_t$)** |

### 2.3 Dataset Origin and Construction

To properly assess generalization boundaries, the data pipeline was engineered iteratively from external resources.

**Synthetic Dataset Construction (Dataset A)**  
This dataset was syntactically synthesized utilizing raw base images strictly sourced from the **Microsoft COCO (Common Objects in Context)** 2014 dataset. Instead of relying on human interrogators, question-answer pairs were automatically generated employing grammatical templates bounding 10 specific mammalian categories (e.g., *cat, dog, elephant, horse, zebra*). The bounding box annotations defining object coordinates and categorical counts were enriched through a secondary deterministic pipeline utilizing **YOLO object detection** complemented by **K-Means clustering** to mathematically define continuous color boundaries.

**Unconstrained Natural Dataset Construction (Dataset B)**  
To juxtapose the synthetic distributions constraint, this secondary dataset was algorithmically filtered directly from the official **VQA v2.0 Benchmark**. The VQA v2.0 benchmark contains millions of human-authored questions constructed to actively counter visual priors (requiring deep visual validation rather than statistical guessing). We constrained the dataset solely by parsing instances where the question strings organically contained predefined animal taxonomy keywords.

### 2.4 Dataset Characteristics

Two datasets were used to analyze the influence of data distribution.

**Dataset A: Synthetic Template Dataset**
- **Scale:** Total 103,660 QA pairs (Train: 72,560 | Val: 15,545 | Test: 15,555).
- **Characteristics:**
  - Automatically generated via syntactical rule engines.
  - Restricted lexical space optimized tightly around 10 distinct mammal categories.
  - Perfectly zero-entropy continuous distribution among categorical answers.
- **Top Answer Frequency:** $`yes`$ (14,512), $`no`$ (14,512), $`1`$ (12,752)

**Dataset B: VQA v2.0 Unconstrained Subset**
- **Scale:** Total 54,283 QA pairs (Train: 38,119 | Val: 8,029 | Test: 8,135).
- **Characteristics:**
  - Pure Human-annotated queries filtered mechanically from MS-COCO.
  - Highly chaotic parsing structures with unbounded Natural Language complexities.
  - Extreme Long-Tail imbalance where polar answers ("Yes/No") artificially bias the optimization target.
- **Top Answer Frequency:** $`no`$ (7,492), $`yes`$ (7,133), $`2`$ (1,623)

### 2.4 Training Configuration (Hyperparameters)

The objective function strives to mitigate the empirical risk.

- **Optimizer:** Adam Optimizer (Initial $\alpha = 0.001$).
- **Loss Mapping Component:** Standard CrossEntropyLoss computed exclusively over valid sequence ranges while padding tokens are explicitly masked (`ignore_index = <PAD>`).
- **Exposure Bias Strategy (Teacher Forcing):** 
  - Ground-truth forcing probability $p_{tf} = 0.5$ at initialization.
  - The parameter undergoes deterministic Linear Decay relative to epoch convergence intervals.
- **Gradient Clipping Threshold:** L2 Norm constraint of $||g|| \le 5.0$ to combat recurrent unbounded gradients.
- **Batch Parallelism:** 32 operational samples / Batch.

---

## 3. Results

Quantitative comparisons dictate testing mechanisms targeting Exact Match Accuracies and Harmonic means. Standardized parameters include:
- Exact Match Accuracy
- Micro Average F1 Score
- BLEU-1 Unigram Precision
- Validation Loss Surface

### 3.1 Synthesis Constraint (Dataset A)

| Model Matrix | Epoch | Val Accuracy | Val Loss | Val F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| Model 1 (Pre. CNN / Base) | 12 | 62.35% | 0.2448 | 62.35 |
| **Model 2 (Pre. CNN / Attn)** | 12 | **71.30%** | **0.2190** | **71.30** |
| Model 3 (Scratch / Base) | 12 | 53.72% | 0.3104 | 61.82 |
| Model 4 (Scratch / Attn) | 12 | 56.24% | 0.3004 | 56.26 |

**Remarks:** Deep transfer learning exhibits a stark dominance here. Model 2 reached state convergence maximizing at roughly ~71%, yielding an absolute variance delta of **+8.95% Accuracy** purely contributed by integrating soft-attention alignment.

### 3.2 Open-World Constraints (Dataset B)

| Model Matrix | Epoch | Val Accuracy | Val Loss | BLEU-1 | Val F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Model 1 | 10 | 27.03% | 1.0738 | 29.91 | 31.13 |
| Model 2 | 10 | 28.11% | 1.0280 | 30.08 | 31.71 |
| **Model 3** | 10 | 27.67% | **0.9482** | 30.76 | **38.84** |
| **Model 4** | 10 | **28.26%** | 0.9523 | **31.15** | 33.77 |

**Remarks:** Accuracies effectively plateaued universally below the 30% barrier limit. Astonishingly, scratch-initialization variants (Models 3 & 4) sustained lower cumulative testing loss mappings and a substantially higher macro F1 index.

---

## 4. Discussion

### 4.1 Generalization Gap vs. Template Inductive Bias
Because Dataset A utilizes syntactically generated prompts, parameter optimization fell prone to severe **Template Overfitting**. The CNN-LSTM network effortlessly recognized continuous surface patterns associated directly with dominant contextual signals bounding visual geometries. However, subsequent testing deployments on Dataset B collapsed overall exact alignments downward by ~43 percent. The network failed to interpolate semantical ambiguity not pre-observed in structural loops, validating fundamental structural weakness.

### 4.2 Local Sub-Optimal Pretraining Limitations
Standard CNN weights calibrated exclusively for ImageNet rely heavily upon macroscopic object classification. Animal properties specified during VQA tests (e.g. geometric counting, isolated sub-sections like "giraffe's neck color") demand heavily localized fractional patch understanding. Therefore, non-initialized CNN variants organically backpropagated the Cross-Entropy risk to force micro-focus extraction locally, hence theoretically justifying the minor F1 lead registered by the From-Scratch designs on complex data.

### 4.3 Decoder Collapse Constraints (Mode Collapse)
Continuous monitoring reveals model degradation cascades around the 9th training epoch (Dataset B), bottoming at ~9% accuracy parameters before partially recovering. Core observations postulate:
1. **Attention Homogenization:** The soft probability weights over the 49 visual fractions scattered broadly, stripping localized discriminative power.
2. **Exposure Bias Compounding:** The rigid reduction mechanism forcing Teacher Probabilities $p \rightarrow 0$ decoupled stability at output recurrent transitions, causing continuous false token repetition (e.g., repeating terminal `<PAD>` or generalized "yes" items).
3. **Cross Entropy Saturation:** In instances observing extreme label-density discrepancy, non-normalized baseline Cross-Entropy failed to project substantial penalties restricting minor-case misclassification.

---

## 5. Conclusion

This research systematically constructed and evaluated a CNN-LSTM architecture for the Visual Question Answering (VQA) task. We assessed four distinct model variants based on a matrix of dual structural characteristics: (1) the inclusion of a Spatial Attention mechanism, and (2) the usage of a Pretrained global feature extractor versus evaluating a From-scratch lightweight CNN initialized organically. Both network paths strictly fed the evaluated semantic question into a sequential LSTM-decoder to formulate responses.

The models were rigorously compared utilizing targeted Evaluation Metrics including Exact Match Accuracy, Micro/Macro F1 Scores, BLEU-1 Unigram Precision, and Validation Loss. Based on these metrics, the empirical analysis establishes the following comparative answers:

**1. Attention vs. No Attention:**
Comparing architectures using evaluation metrics demonstrates that **Spatial Attention broadly enhances exact alignments and structural confidence**. On the synthetic Dataset A, Spatial Attention generated a massive **+8.95% exact-match Accuracy boost** inside Pretrained networks (scoring 71.30%). While on the highly unconstrained Dataset B, Attention remained crucial for solidifying optimization, lifting the from-scratch network to the highest registered Accuracy score (28.26%) across all trials. 

**2. Pretrained vs. From-Scratch Initialization:**
The evaluation metrics highlight a heavy dependence on **Data Distribution** when selecting initialization paths. 
- On simpler, structured data where global context dominates, the **Pretrained ResNet50 overwhelmingly succeeds** (Model 2 scoring 71.30% Accuracy vs. Model 4's 56.24%). 
- Conversely, within highly chaotic, open-world human linguistics (Dataset B), macroscopic Pretrained ImageNet weights failed to adapt to micro-localization requirements. Here, the lightweight **From-scratch ScratchCNN forced to learn pixel-level relationships locally yielded superior Harmonic F1 Scores** (38.84% vs 31.71%), proving it generalizes better to long-tail textual concepts despite having fewer parameters.

Ultimately, solving true abstract reasoning in VQA requires traversing beyond recurrent structural modeling. Future systematic corrections recommend transitioning towards high-parameter **Transformer-based (ViT) Cross-Attention Modules**.
