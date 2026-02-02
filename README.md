# SAE for CLIP (MFTI Course 2)

## What is CLIP?
CLIP (Contrastive Language-Image Pretraining) is a vision-language model trained on image-text pairs with a contrastive objective.
It learns aligned embeddings for images and texts in a shared space where matching pairs are close.
This alignment lets a single model support both visual and textual semantics.
Zero-shot classification works by embedding an image and comparing it to embeddings of text prompts.
The highest similarity between the image embedding and prompt embeddings gives the predicted class.
This makes CLIP a strong general-purpose feature extractor without task-specific training.
In this project we use OpenCLIP ViT-B/32 (laion2b_s34b_b79k).
It is CLIP-compatible and provides 512-dim image embeddings for further analysis.

## Related Work

Our project builds on recent work on interpreting multimodal representations using Sparse Autoencoders (SAE).
In "Towards Multimodal Interpretability: Learning Sparse Autoencoders for CLIP", the authors train SAEs on CLIP embeddings
and show that individual latent units correspond to interpretable visual and semantic concepts.
This work demonstrates that CLIP representations contain disentangled structure that can be uncovered through sparse coding.

In "Interpreting and Steering Features in Images", SAE features learned from CLIP are further used to control image generation
in Kandinsky2.2 by selectively amplifying or suppressing specific latent activations.
This shows that interpretable SAE features are not only descriptive, but can also be used for model steering.

In this project, we reimplement the core SAE training and analysis pipeline for CLIP image embeddings in a simplified and
reproducible setting, and later apply the same principles to study interpretability and controllability of visual features.

## Results

# Dataset and Embeddings
We use Flickr30k images and extract image embeddings with OpenCLIP ViT-B/32 (laion2b_s34b_b79k).
Each image is represented by a 512-dimensional CLIP embedding.

# Sparse Autoencoder Training
A Sparse Autoencoder (SAE) is trained on the CLIP image embeddings with the following setup:
- Dictionary size: 4096
- Optimizer: Adam
- Learning rate: 1e-3
- Batch size: 256
- Epochs: 5
- Sparsity regularization: L1 penalty

### SAE Training Metrics

| L1 coefficient (λ) | Dictionary size | Avg. L0 (active latents) | Reconstruction MSE | R2 score (Explained Variance) |
| ------------------ | --------------- | ------------------------- | ------------------ | ----------------------------- |
| 1e-3               | 4096            | ≈ 2050                    | ≈ 6e-6             | ≈ 0.995                       |
| 1e-2               | 4096            | ≈ 900                     | ≈ 1.5e-5           | ≈ 0.988                       |
| 3e-2               | 4096            | ≈ 570                     | ≈ 3.0e-5           | ≈ 0.977                       |

As λ increases, the average number of active latents drops, indicating stronger sparsity.
This sparsity comes with a gradual increase in reconstruction error and a small decrease in explained variance.
Overall, the metrics show a clear sparsity–reconstruction trade-off while preserving most of the variance.

We evaluate three sparsity levels:
1) λ = 1e-3
2) λ = 1e-2
3) λ = 3e-2

Increasing the L1 coefficient leads to stronger sparsity (lower average number of active latents) at the cost of reconstruction quality.

## Zero-Shot Classification Evaluation
We evaluate the effect of SAE reconstruction on zero-shot classification accuracy using CIFAR-10 and CIFAR-100.

We evaluate zero-shot performance on CIFAR-10 and CIFAR-100 to cover datasets of different difficulty.
CIFAR-10 contains a small number of coarse-grained object categories, making it an easier benchmark.
CIFAR-100 is significantly more challenging due to fine-grained classes and higher inter-class similarity.
Together, these datasets allow us to assess how SAE reconstruction affects CLIP performance across varying levels of semantic complexity.

Results for λ = 3e-2 (best interpretability):
| Dataset   | CLIP Baseline | CLIP + SAE |
| --------- | ------------- | ---------- |
| CIFAR-10  | 0.9366        | 0.9187     |
| CIFAR-100 | 0.7569        | 0.7182     |

As expected, reconstruction through the SAE slightly reduces accuracy, but performance remains competitive while gaining interpretability.

## Interpretation of SAE Latents
To analyze interpretability, we extract top-activating images for individual SAE latents and visualize them as image collages.

Each latent corresponds to a coherent, human-interpretable visual concept, demonstrating that the SAE successfully disentangles CLIP representations.

Examples of learned latents:
Latent 0: Outdoor and urban scenes (streets, buildings, open spaces)
Latent 1: Food and table scenes (meals, kitchens, cafes)
Latent 2: Human portraits and close-ups
Latent 3: Animals and pets
Latent 4: Crowds and group activities
Latent 5: Vehicles and transportation
Latent 6: Water-related scenes (pools, swimming, seaside)
Latent 7: Indoor human activities
Latent 8: Events and performances

These visualizations show that individual SAE neurons correspond to semantic features rather than low-level visual patterns.

## Discussion
The results demonstrate a clear trade-off:
- Higher sparsity → stronger interpretability
- Slight degradation in downstream task accuracy

Despite this, the SAE preserves most of the semantic information present in CLIP embeddings while providing a transparent and analyzable latent space.

This supports the hypothesis that CLIP representations contain disentangled semantic structure that can be uncovered using sparse coding methods.

## Conclusion
In this project, we:
1. Extracted CLIP image embeddings from a large-scale dataset
2. Trained a Sparse Autoencoder to obtain sparse, interpretable representations
3. Evaluated the effect of SAE reconstruction on zero-shot classification
4. Demonstrated that individual SAE latents correspond to meaningful visual concepts

Overall, Sparse Autoencoders provide an effective tool for interpreting and analyzing CLIP representations with minimal loss in performance.
