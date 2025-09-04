# CT-CLIP Embedding Space Analysis

## Overview
CT-CLIP (Clinical Text-Guided Contrastive Learning for CT Image Analysis) creates a joint embedding space where CT images and their corresponding clinical text descriptions are aligned through contrastive learning. This document analyzes how the embedding space is formed and structured.

## CT-CLIP vs Original CLIP: Key Differences

### Domain Adaptation
| Aspect | Original CLIP | CT-CLIP |
|--------|---------------|----------|
| **Input Data** | 2D natural images + captions | 3D chest CT volumes + radiology reports |
| **Text Encoder** | General language model | BiomedVLP-CXR-BERT (medical domain-specific) |
| **Image Processing** | 2D patches from RGB images | 3D volume processing with mean pooling |
| **Training Data** | 400M image-text pairs from web | 50,188 CT volumes with professional radiology reports (CT-RATE dataset) |
| **Vocabulary** | General English | Medical terminology and radiology-specific language |

### Architectural Modifications
1. **3D Vision Transformer**: Modified to handle volumetric CT data instead of 2D images
2. **Medical Text Encoder**: Uses specialized medical BERT variant trained on radiology reports
3. **Patch Size Adaptation**: Larger patches (32x32) to handle CT volume characteristics
4. **Pooling Strategy**: Mean pooling over spatial dimensions for volume representation
5. **Patch Dropout**: 50% dropout rate for robustness (higher than typical CLIP)

### Training Enhancements
- **Multi-abnormality Labels**: Supports 18 different pathology classifications
- **Three Training Variants**:
  - Zero-shot CT-CLIP: Pure contrastive learning
  - VocabFine: Fine-tuned on medical vocabulary
  - ClassFine: Fine-tuned with classification head (faster inference)
- **Dataset**: CT-RATE with 25,692 unique CT volumes from 21,304 patients
- **Hardware Requirements**: A100 GPU with 80GB VRAM for batch size 8 (due to 3D volume processing)

## Embedding Space Architecture

### 1. Dual-Encoder Architecture

CT-CLIP uses two separate encoders that project different modalities into a shared latent space:

#### Text Encoder
- **Base Model**: BERT-based architecture (`microsoft/BiomedVLP-CXR-BERT-specialized`)
- **Architecture Components**:
  - Token embedding layer
  - Transformer blocks (depth configurable, default 6 layers)
  - Optional: Absolute position embeddings or Rotary position embeddings
  - CLS token for sequence representation
- **Input**: Tokenized clinical text (max sequence length: 256-512 tokens)
- **Output Dimension**: Configurable (default: 512)

#### Vision Encoder
- **Architecture**: Vision Transformer (ViT)
- **Key Features**:
  - Patch-based encoding (default patch size: 32x32)
  - Patch dropout for regularization (default: 50%)
  - Positional embeddings for spatial information
  - Mean pooling over spatial dimensions for final representation
- **Input**: CT images (default size: 256x256)
- **Output Dimension**: Configurable (default: 512)

### 2. Embedding Projection Layers

Both encoders output embeddings that are projected into a shared latent space:

```
Text Embedding → Linear(dim_text, dim_latent) → L2 Normalization → Text Latent
Image Embedding → Linear(dim_image, dim_latent) → L2 Normalization → Image Latent
```

Key characteristics:
- **Dimensionality Reduction**: Projects from encoder dimensions to shared latent dimension
- **L2 Normalization**: Ensures embeddings lie on a unit hypersphere
- **Optional Extra Projections**: For decoupled contrastive learning (CLOOB-style)

### 3. Contrastive Learning Mechanism

The embedding space is shaped through contrastive loss that:
- **Pulls together** matched image-text pairs
- **Pushes apart** unmatched pairs

#### Mathematical Foundation

##### Loss Formulation
```python
# Temperature-scaled similarity (logit scale)
temp = exp(self.temperature)  # Learnable logit scale
sim = (text_latent @ image_latent.T) * temp

# Symmetric InfoNCE Loss
text_to_image_loss = -log(exp(sim[i,i]) / sum(exp(sim[i,:])))
image_to_text_loss = -log(exp(sim[i,i]) / sum(exp(sim[:,i])))
total_loss = (text_to_image_loss + image_to_text_loss) / 2
```

##### Geometric Properties
- **Unit Hypersphere**: L2 normalization projects all embeddings onto S^(d-1)
- **Cosine Similarity**: Inner product equals cosine similarity after normalization
- **von Mises-Fisher Distribution**: Embeddings follow vMF-like distribution on sphere
- **Alignment-Uniformity Trade-off**: Loss simultaneously optimizes for:
  - Alignment: Positive pairs have high similarity
  - Uniformity: Embeddings are uniformly distributed on hypersphere

#### Temperature Parameter Analysis
- **Implementation**: `self.temperature` is a learnable parameter, exponential ensures positivity
- **Initial Value**: Default 1.0 → exp(1) ≈ 2.718 (equivalent to τ ≈ 0.368)
- **Effect on Distribution**:
  - High temperature (low τ): Sharper distribution, stronger focus on hardest negatives
  - Low temperature (high τ): Smoother distribution, considers all negatives equally
- **Optimal Range**: For medical domain with smaller batches:
  - Recommended initialization: `log(1/0.07) ≈ 2.65` (CLIP standard)
  - Clamping range: `[log(1/30), log(100)]` for stability

#### Key Components:
- **Bidirectional Loss**: Both text→image and image→text similarities are optimized
- **Symmetric Design**: Equal weight to both modality directions
- **Negative Sampling**: In-batch negatives provide uniformity signal

### 4. Embedding Space Properties

#### Multimodal Alignment
The space aligns visual features with semantic concepts from text:
- CT scan patterns map to clinical descriptions
- Similar pathologies cluster together
- Severity levels form gradients in the space

#### Hierarchical Structure
Based on t-SNE visualization analysis:
1. **Primary Clustering**: Presence/absence of pathologies
2. **Secondary Clustering**: Number of pathologies (severity)
3. **Fine-grained Structure**: Specific pathology types

#### Pathology Grouping Pattern
The embedding space naturally organizes samples by pathology count:
- No pathologies (healthy cases)
- 1-3 pathologies (mild cases)
- 4-6 pathologies (moderate cases)  
- 7-9 pathologies (moderate-severe cases)
- 10-12 pathologies (severe cases)
- >13 pathologies (critical cases)

### 5. Advanced Features

#### Fine-grained Matching (FILIP-style)
- **Token-level Alignment**: 
  - When `use_all_token_embeds=True`, computes token × patch similarity matrix
  - Aggregation: `max` pooling followed by `masked mean` for final score
  - Benefits: Better localization of pathologies, word-level alignment with specific regions
  - Trade-offs: Higher memory/computation cost, requires careful masking
- **Spatial Attention**: Image patches align with specific text tokens describing local features
- **Clinical Advantage**: Captures fine-grained pathology-description correspondences

#### CLS Token Matching (Standard)
- **Global Representation**: Uses `enc_text[:,0,:]` (first token) and global pooled image
- **Efficiency**: Lower memory footprint, faster inference
- **Use Case**: Suitable for global classification and coarse-grained retrieval
- **Limitation**: May miss local pathology-text alignments

#### Decoupled Contrastive Learning (CLOOB)
- **Positive Decoupling**: When `decoupled_contrastive_learning=True`:
  - Removes positive pair from denominator
  - Reduces denominator bias from hard positives
  - Stabilizes training dynamics
- **Dual Projections**: When `extra_latent_projection=True`:
  - Separate projection heads for text→image vs image→text
  - Addresses modality asymmetry in medical domain
  - Particularly effective for imbalanced medical terminology

#### Multi-view Learning
- **Implementation**: 
  - `aug_text` and `aug_image` provide augmented views
  - Combined with main loss via `multiview_loss_weight`
- **Medical Augmentations**:
  - Image: Limited geometric transforms (preserve anatomical structure)
  - Text: Synonym replacement, sentence reordering (preserve clinical meaning)
- **Robustness Benefits**:
  - Scanner/reconstruction variations
  - Report writing style differences
  - Improves embedding uniformity

#### Self-Supervised Components
- **MLM (Masked Language Modeling)**: 
  - Text encoder pre-training with medical corpus
  - Weight: `text_ssl_loss_weight` (default 0.05)
  - Improves medical terminology understanding
- **Visual SSL**: 
  - SimSiam or SimCLR for image encoder
  - Weight: `image_ssl_loss_weight` (default 0.05)
  - Enhances visual feature invariance
- **Joint Training**: Combined loss = CL + α·MLM + β·Visual_SSL + γ·Multiview

### 6. Embedding Extraction and Usage

#### For Downstream Tasks
```python
# Extract text embeddings
text_tokens = model.tokenize(text)
text_embeddings = model.text_transformer(text_tokens)
text_latents = model.to_text_latent(text_embeddings)

# Extract image embeddings  
image_embeddings = model.visual_transformer(images)
image_latents = model.to_visual_latent(image_embeddings)

# Compute similarity
similarity = (text_latents @ image_latents.T) * temperature
```

#### Zero-shot Classification
The learned embedding space enables:
- Disease classification without task-specific training
- Cross-modal retrieval (text→image, image→text)
- Similarity-based diagnosis

### 7. Visualization and Analysis

#### t-SNE Visualization
The `tsne_latents.py` script reveals:
- **Primary Structure**: Clear separation between pathological and healthy cases
- **Severity Gradient**: Progressive clustering from healthy → mild → severe cases
- **Pathology-specific Regions**: Different pathologies occupy distinct regions in 2D projection
- **Mixed Pathology Positioning**: Cases with multiple pathologies appear in boundary regions

#### t-SNE Interpretation Guidelines
- **Local Structure Emphasis**: t-SNE preserves local neighborhoods better than global distances
- **Perplexity Impact**: Values 5-50 affect cluster tightness (higher = smoother boundaries)
- **Validation**: Cross-verify with UMAP and cosine distance matrices
- **Caution**: Global distances in t-SNE can be misleading; use for qualitative insights only

#### Embedding Quality Metrics
- **Alignment Score**: Mean cosine similarity of positive pairs (target: >0.3)
- **Uniformity Metric**: Average pairwise similarity exp(-t||x-y||²) (target: <-2.0)
- **Silhouette Score**: Cluster coherence measure (-1 to 1, higher is better)
- **Retrieval Metrics**:
  - R@1, R@5, R@10 for both text→image and image→text
  - Mean Reciprocal Rank (MRR)
  - Normalized Discounted Cumulative Gain (NDCG)

#### Medical Domain Characteristics
- **Terminology Specialization**: Medical BERT captures nuanced clinical vocabulary
- **Hierarchical Disease Structure**: Natural emergence of severity-based hierarchy
- **Cross-pathology Relationships**: Related conditions cluster nearby (e.g., pneumonia-consolidation)
- **Distribution Shift Robustness**: Multi-view and SSL components handle scanner variations

## Technical Implementation Details

### Embedding Dimensions
- Text encoder output: `dim_text` (default: 512)
- Image encoder output: `dim_image` (default: 512)  
- Shared latent space: `dim_latent` (default: 512)

### Normalization Strategy
- L2 normalization ensures embeddings have unit norm
- Enables use of cosine similarity for matching
- Stabilizes training dynamics

### Temperature Scaling
- Learnable parameter (initialized at 1.0)
- Controls the concentration of similarity distribution
- Critical for contrastive learning convergence

## Applications of the Embedding Space

1. **Zero-shot Disease Detection**: Classify diseases using text descriptions
2. **Report Generation**: Find similar cases for reference
3. **Quality Control**: Detect mismatched reports and images
4. **Multimodal Retrieval**: Search images using text queries
5. **Clustering Analysis**: Discover disease patterns and subtypes
6. **CT-CHAT Integration**: Powers visual-language chat model for 3D chest CT volumes

## Performance Characteristics

### Inference Speed
- **Zero-shot CT-CLIP**: ~1.5 seconds for 18 pathology assessment
- **VocabFine**: ~1.5 seconds for 18 pathology assessment  
- **ClassFine**: ~0.5 seconds for 18 pathology assessment (3x faster)

### Pathology Coverage
Supports 18 different pathologies including:
- Medical material
- Arterial wall calcification
- Cardiomegaly
- Pericardial effusion
- Coronary artery wall calcification
- Hiatal hernia
- Lymphadenopathy
- Emphysema
- Atelectasis
- Lung nodule
- Lung opacity
- Pulmonary fibrotic sequela
- Pleural effusion
- Mosaic attenuation pattern
- Peribronchial thickening
- Consolidation
- Bronchiectasis
- Interlobular septal thickening

## Key Insights

1. **Clinical Semantic Capture**: The embedding space successfully aligns clinical concepts across modalities
2. **Severity-based Organization**: Natural emergence of pathology severity gradients without explicit supervision
3. **Zero-shot Capability**: Learns transferable medical concepts enabling unseen pathology detection
4. **Localization vs Efficiency Trade-off**: Fine-grained matching improves localization but increases computational cost
5. **Robustness Through Diversity**: Multi-view learning and SSL create invariance to clinical variations
6. **Temperature Criticality**: Proper temperature tuning is essential for medical domain (smaller batches, specialized vocabulary)
7. **Modality Asymmetry**: CLOOB-style decoupling addresses inherent text-image distribution differences in medical data
8. **3D Adaptation Success**: Mean pooling strategy effectively handles volumetric nature of CT data

## Practical Recommendations

### For Model Training
1. **Temperature Initialization**: Start with `log(1/0.07)` and clamp to `[log(1/30), log(100)]`
2. **Batch Size**: Maintain ≥8 for sufficient negatives (requires high VRAM for 3D data)
3. **Loss Weights**: Keep SSL weights at 0.05-0.1, multiview at 0.1-0.2
4. **Fine-grained vs CLS**: Choose based on downstream task (localization vs classification)

### For Embedding Quality
1. **Monitor Metrics**: Track alignment, uniformity, and retrieval metrics during training
2. **Visualization**: Use both t-SNE and UMAP with multiple perplexity values
3. **Validation**: Test on held-out pathologies for zero-shot evaluation
4. **Augmentation**: Carefully design medical-appropriate augmentations

### For Downstream Applications
1. **Zero-shot**: Leverage for rapid prototyping on new pathologies
2. **Fine-tuning**: Use ClassFine variant for production speed requirements
3. **Retrieval**: Implement efficient nearest neighbor search (e.g., FAISS) for large-scale deployment
4. **Interpretability**: Use fine-grained matching for explainable AI requirements

## Code Implementation Analysis

### 1. Core Architecture Implementation

#### Text Processing Pipeline (ct_clip.py)
```python
# Text encoding with medical BERT (line 685)
text_embeddings = self.text_transformer(text.input_ids, attention_mask=text.attention_mask)
enc_text = text_embeddings[0]  # Shape: (batch, seq_len, 768)

# CLS token extraction (line 762)
text_embeds = enc_text[:,0,:]  # First token (CLS)

# Projection to shared latent space (line 765)
text_latents = self.to_text_latent(text_embeds)  # Linear(768, 512)

# L2 normalization to unit hypersphere (line 771)
text_latents = l2norm(text_latents)
```

#### Vision Processing Pipeline with CTViT
```python
# 3D CT encoding (line 715)
enc_image = self.visual_transformer(image, return_encoded_tokens=True)
# Returns: (batch, time, height, width, dim)

# Temporal averaging (line 724)
enc_image = torch.mean(enc_image, dim=1)  # Aggregate across slices

# Spatial flattening (line 740)
enc_image = enc_image.view(batch_size, -1)  # Flatten to 1D

# Projection and normalization (lines 767, 771)
image_latents = self.to_visual_latent(enc_image)
image_latents = l2norm(image_latents)
```

### 2. Contrastive Loss Implementation

#### Temperature-Scaled InfoNCE Loss
```python
# Temperature parameter (line 796)
temp = self.temperature.exp()  # Learnable logit scale

# Similarity computation (line 845, CLS-only path)
text_to_image = einsum('m t d, n i d -> m n t i', text_latents, image_latents) * temp

# Exponentiation (line 858)
text_to_image_exp = torch.exp(text_to_image)

# Positive pairs extraction (line 861)
text_to_image_pos = matrix_diag(text_to_image_exp)

# Optional CLOOB decoupling (lines 865-868)
if self.decoupled_contrastive_learning:
    pos_mask = torch.eye(batch_size)
    text_to_image_exp.masked_fill_(pos_mask, 0.)  # Remove positive from denominator

# Loss calculation (lines 873-878)
text_to_image_loss = (-log(text_to_image_pos) + log(text_to_image_exp.sum(dim=-1))).mean()
image_to_text_loss = (-log(image_to_text_pos) + log(image_to_text_exp.sum(dim=-1))).mean()
cl_loss = (text_to_image_loss + image_to_text_loss) / 2
```

### 3. Training and Inference

#### Training Loop (CTCLIPTrainer.py)
```python
# Data loading and tokenization (lines 244-251)
video, text = next(self.dl_iter)
text_tokens = self.tokenizer(text, return_tensors="pt", 
                            padding="max_length", max_length=512)

# Forward pass with loss (line 255)
with self.accelerator.autocast():
    loss = self.CTClip(text_tokens, video, return_loss=True, device=device)

# Optimization (lines 257-263)
self.accelerator.backward(loss)
self.accelerator.clip_grad_norm_(self.CTClip.parameters(), self.max_grad_norm)
self.optim.step()
```

#### Zero-shot Inference (zero_shot.py)
```python
# For each pathology, create binary prompts (lines 133-144)
for pathology in pathologies:
    text = [f"{pathology} is present.", f"{pathology} is not present."]
    text_tokens = self.tokenizer(text, return_tensors="pt", max_length=512)
    
    # Get similarity scores
    output = model(text_tokens, valid_data.cuda(), device=device)
    
    # Apply softmax for probability
    output = apply_softmax(output)  # Binary probability
    predictedlabels.append(output[0])  # Probability of presence
```

## Final Comprehensive Analysis

### 1. Key Innovations for Medical Domain

#### 3D Medical Image Processing
- **CTViT Integration**: Uses specialized 3D Vision Transformer that processes volumetric data directly
- **Temporal Aggregation**: Mean pooling across CT slices preserves global pathology patterns
- **Spatial Flattening**: Converts 3D features to 1D for efficient projection to latent space

#### Medical Text Understanding
- **Domain-Specific BERT**: BiomedVLP-CXR-BERT pretrained on radiology reports
- **Binary Prompt Design**: "X is present" vs "X is not present" for clear medical decisions
- **18 Independent Pathologies**: Each disease evaluated separately for multi-label classification

### 2. Embedding Space Characteristics

#### Geometric Properties
- **Unit Hypersphere**: L2 normalization ensures all embeddings lie on S^(d-1)
- **Cosine Similarity**: Inner product equals cosine similarity after normalization
- **Temperature Scaling**: Learnable parameter controls distribution sharpness (default: exp(1) ≈ 2.718)

#### Learning Dynamics
- **Alignment**: Positive pairs pulled together through InfoNCE loss
- **Uniformity**: Embeddings distributed evenly on hypersphere
- **Bidirectional Optimization**: Both text→image and image→text directions

### 3. Implementation Insights

#### Why CLS Token Only?
- **Efficiency**: Reduces memory from O(n×m) to O(1) for similarity computation
- **Global Semantics**: BERT CLS token captures document-level meaning
- **Stability**: Single vector per sample simplifies loss calculation

#### CTViT Temporal Averaging Significance
- **Information Integration**: Combines information across all CT slices
- **Noise Reduction**: Averages out slice-specific artifacts
- **Memory Efficiency**: Reduces 3D volume to 2D representation

#### 18 Independent Pathologies Design
- **Advantages**: 
  - Easy zero-shot extension to new diseases
  - No class imbalance issues
  - Simple binary decisions
- **Limitations**:
  - Ignores disease co-occurrence patterns
  - Requires separate threshold calibration
  - Higher inference time (18 forward passes)

### 4. Practical Usage Guide

#### Embedding Extraction Pattern
```python
# Extract text embedding
text_tokens = tokenizer(text, return_tensors="pt")
text_features = model.text_transformer(text_tokens)[0][:,0,:]
text_embedding = l2norm(model.to_text_latent(text_features))

# Extract image embedding
image_features = model.visual_transformer(ct_volume, return_encoded_tokens=True)
image_features = torch.mean(image_features, dim=1).flatten(1)
image_embedding = l2norm(model.to_visual_latent(image_features))

# Compute similarity
similarity = (text_embedding @ image_embedding.T) * model.temperature.exp()
```

#### Zero-shot vs Fine-tuning Decision
- **Use Zero-shot when**:
  - Limited labeled data (<1000 samples)
  - Need to detect new/rare pathologies
  - Rapid prototyping required
  
- **Use Fine-tuning when**:
  - Deployment speed critical (ClassFine 3x faster)
  - Institution-specific adaptation needed
  - High accuracy on known pathologies required

### 5. Performance and Limitations

#### Resource Requirements
- **Memory**: A100 80GB for batch size 8 (3D volumes + large dim_image)
- **Computation**: ~1.5s for zero-shot, ~0.5s for ClassFine (18 pathologies)
- **Storage**: Model weights ~2GB, CT volumes ~100-500MB each

#### Current Limitations
1. **No Disease Correlation**: Independent binary classification misses co-occurrence patterns
2. **Fixed Temperature**: No adaptive scaling based on sample difficulty
3. **Simple Pooling**: Mean averaging may lose important slice-specific information
4. **Memory Intensive**: Full 3D processing requires high-end GPUs

#### Improvement Opportunities
1. **Multi-label Correlation**: Model disease co-occurrence with graph networks
2. **Learned Pooling**: Attention-based temporal aggregation
3. **Efficient Architecture**: Pruning or quantization for deployment
4. **Slice-level Attention**: Identify which slices contribute to diagnosis
5. **Prompt Engineering**: Optimize medical prompt templates

### 6. Key Takeaways for Other Projects

#### When to Use CT-CLIP Architecture
- 3D medical imaging tasks with limited labels
- Need for interpretable text-image alignment
- Multi-label classification with class imbalance
- Cross-modal retrieval in medical domain

#### Critical Implementation Details
- Set `dim_image` to match flattened CTViT output (e.g., 24×24×512=294,912)
- Initialize temperature with `log(1/0.07)` for medical domain
- Use batch size ≥8 for sufficient negative samples
- Consider ClassFine variant for production deployment

#### Success Factors
- Domain-specific text encoder is crucial
- Temporal averaging works well for volumetric data
- Binary prompts simplify multi-label classification
- L2 normalization stabilizes training

## References
- Original CLIP paper (Radford et al., 2021)
- CLOOB improvements (Wortsman et al., 2021)
- BiomedVLP-CXR-BERT (Boecking et al., 2022)
- CT-RATE dataset and CT-CLIP paper (Hamamci et al., 2024)