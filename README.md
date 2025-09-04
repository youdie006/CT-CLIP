# CT-CLIP: Clinical Text-Guided Contrastive Learning for CT Image Analysis

## Overview

CT-CLIP is a medical AI model that creates a joint embedding space where 3D chest CT volumes and their corresponding clinical text descriptions are aligned through contrastive learning. This repository contains a comprehensive analysis of the CT-CLIP model architecture, implementation, and embedding space characteristics.

**Note**: This is an analysis and study repository. The original CT-CLIP implementation and all copyrights belong to the original authors at [https://github.com/ibrahimethemhamamci/CT-CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP).

## Key Features

### Medical Domain Optimization
- **3D Vision Transformer (CTViT)**: Specialized for volumetric CT data processing
- **Medical BERT**: BiomedVLP-CXR-BERT pretrained on radiology reports
- **18 Pathology Detection**: Zero-shot classification for chest abnormalities
- **CT-RATE Dataset**: 50,188 CT volumes with professional radiology reports

### Architecture Highlights
| Component | Original CLIP | CT-CLIP |
|-----------|--------------|----------|
| Input Data | 2D RGB images | 3D CT volumes |
| Text Encoder | General BERT | Medical BiomedVLP-CXR-BERT |
| Vision Encoder | 2D ViT | 3D CTViT with temporal pooling |
| Training Data | 400M web pairs | 50K medical CT-report pairs |
| Inference Speed | - | 0.5-1.5 seconds per case |

## Model Architecture

### Dual-Encoder Design
```
CT Volume → CTViT → Temporal Averaging → Projection → L2 Norm → Image Embedding
                                                               ↓
                                                        Cosine Similarity
                                                               ↑
Clinical Text → Medical BERT → CLS Token → Projection → L2 Norm → Text Embedding
```

### Embedding Space Properties
- **Geometry**: Unit hypersphere (S^511) with L2 normalization
- **Similarity**: Temperature-scaled cosine similarity
- **Loss**: Bidirectional InfoNCE with optional CLOOB decoupling
- **Organization**: Hierarchical clustering by pathology severity

## Implementation Details

### Core Processing Pipeline

#### Text Processing
```python
# Medical BERT encoding
text_embeddings = medical_bert(text_tokens)
text_features = text_embeddings[0][:,0,:]  # CLS token
text_latent = l2norm(projection(text_features))
```

#### Vision Processing
```python
# 3D CT encoding with CTViT
ct_tokens = ctvit(ct_volume, return_encoded_tokens=True)  # (B,T,H,W,D)
ct_features = torch.mean(ct_tokens, dim=1)  # Temporal averaging
ct_features = ct_features.flatten(1)  # Spatial flattening
image_latent = l2norm(projection(ct_features))
```

### Training Configurations

| Variant | Method | Speed | Use Case |
|---------|--------|-------|----------|
| Zero-shot | Pure contrastive learning | 1.5s | New pathologies, limited labels |
| VocabFine | Medical vocabulary tuning | 1.5s | Improved medical understanding |
| ClassFine | Classification head | 0.5s | Production deployment |

## Usage Guide

### Zero-shot Disease Detection
```python
# For each pathology, create binary prompts
pathologies = ['Emphysema', 'Atelectasis', 'Lung nodule', ...]
for pathology in pathologies:
    prompts = [f"{pathology} is present.", f"{pathology} is not present."]
    similarity = model(prompts, ct_volume)
    probability = softmax(similarity)
```

### Embedding Extraction
```python
# Extract embeddings for downstream tasks
text_embedding = model.encode_text(clinical_report)
image_embedding = model.encode_image(ct_volume)
similarity_score = (text_embedding @ image_embedding.T) * temperature
```

## Performance Characteristics

### Resource Requirements
- **GPU**: NVIDIA A100 80GB (for batch size 8)
- **Memory**: ~294,912 dimensional flattened features
- **Storage**: ~2GB model weights

### Inference Performance
- **18 Pathologies Assessment**: 0.5-1.5 seconds
- **Batch Processing**: Supported with sufficient VRAM
- **Accuracy**: State-of-the-art on CT-RATE benchmark

## Key Innovations

1. **3D Medical Adaptation**: Temporal averaging elegantly handles volumetric data
2. **Domain-Specific Encoders**: Medical BERT captures radiology terminology
3. **Binary Prompt Design**: Simple "present/not present" for each pathology
4. **Temperature Learning**: Adaptive logit scaling for medical domain
5. **Flexible Training**: Three variants for different deployment needs

## Limitations and Future Work

### Current Limitations
- Independent pathology classification (ignores co-occurrence)
- Fixed temperature scaling
- High memory requirements for 3D processing
- Simple mean pooling may lose slice-specific information

### Improvement Opportunities
- Multi-label correlation modeling
- Attention-based temporal aggregation
- Model compression for edge deployment
- Slice-level explainability

## File Structure

```
CT-CLIP/
├── CT_CLIP/
│   └── ct_clip/
│       └── ct_clip.py          # Main CTCLIP model implementation
├── scripts/
│   ├── CTCLIPTrainer.py        # Training workflow
│   ├── zero_shot.py            # Zero-shot inference
│   └── tsne_latents.py         # Embedding visualization
├── transformer_maskgit/
│   └── ctvit.py                # 3D Vision Transformer
├── EMBEDDING_ANALYSIS.md        # Detailed embedding space analysis
└── README_ORIGINAL.md           # Original repository documentation
```

## Analysis Documents

- **[EMBEDDING_ANALYSIS.md](EMBEDDING_ANALYSIS.md)**: Comprehensive analysis of the embedding space formation, mathematical foundations, and implementation details

## Requirements

```bash
# Navigate to transformer_maskgit and install
cd transformer_maskgit
pip install -e .

# Navigate to CT_CLIP and install
cd ../CT_CLIP
pip install -e .
```

## Citation

If you use CT-CLIP in your research, please cite the original paper:

```bibtex
@article{hamamci2024ctclip,
  title={CT-CLIP: A Foundation Model for Chest CT Volumes},
  author={Hamamci, Ibrahim Ethem and others},
  journal={arXiv preprint arXiv:2403.17834},
  year={2024}
}
```

## License and Copyright

**Important**: This repository is for educational and research purposes only. All original code, models, and datasets are property of the original CT-CLIP authors. Please refer to the original repository at [https://github.com/ibrahimethemhamamci/CT-CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP) for:

- Original implementation and source code
- Pretrained model weights
- CT-RATE dataset access
- Official documentation
- Licensing information (CC-BY-NC-SA)

This analysis repository does not claim any ownership of the CT-CLIP model, code, or associated materials.

## Acknowledgments

Special thanks to the original CT-CLIP authors for their groundbreaking work in medical AI and for making their research publicly available. This analysis aims to help researchers understand and utilize CT-CLIP effectively in their own medical imaging projects.