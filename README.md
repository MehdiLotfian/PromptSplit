# PromptSplit: Revealing Prompt-Level Disagreement in Generative Models

This repository provides the official implementation of **PromptSplit**, a kernel-based framework for detecting and analyzing prompt-dependent disagreements between generative AI models, as described in the accompanying paper.

**[PromptSplit: Revealing Prompt-Level Disagreement in Generative Models](https://arxiv.org/abs/2602.04009)**  

## Overview

PromptSplit enables systematic comparison of prompt-guided generative models in vision and language domains, such as text-to-image and text-to-text systems. By constructing joint prompt-output representations via tensor-product embeddings, it computes kernel covariance differences to identify prompt categories leading to divergent behaviors. For scalability, the framework incorporates random-projection approximations, achieving O(nr² + r³) computational complexity with an O(1/r²) accuracy bound.

This codebase supports embedding extraction for images and text, kernel-based analysis, and random projection methods, making it extensible for evaluating models like Stable Diffusion, Kandinsky, PixArt, Qwen, and Gemma.

As an illustration, Figure 1 below demonstrates PromptSplit applied to text-to-text models (Qwen3 vs. Gemma3) on NQ-Open questions, highlighting top disagreement modes with representative prompts and outputs.

![Figure 1: PromptSplit Framework](./figures/qwen_gemma_nq.png)

---

## Repository Structure

```
.
├── algorithm/
│   ├── random_projection.py   # Implements random projection methods
│   └── kernel_method.py       # Implements kernel-based methods
├── utils/
│   ├── ...                   # Scripts for image generation, text generation, and embedding extraction
├── README.md                 # Project documentation (this file)
└── LICENSE                   # MIT License details
```

- **algorithm/**: Contains core algorithmic implementations:
  - `random_projection.py`: Dimensionality reduction and kernel approximation via random projections.
  - `kernel_method.py`: Kernel-based similarity and feature analysis methods.
- **utils/**: Utility scripts for:
  - Image generation
  - Text generation
  - Embedding extraction
  (Add your scripts here as needed.)

---

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/unified-kernel-eval.git
   cd unified-kernel-eval
   ```

2. **Install dependencies:**
   - Python 3.8+
   - numpy

   You can use `pip` or `conda` to install requirements.

3. **Usage:**
   - See the scripts in `utils/` for data generation and embedding extraction.
   - Use the methods in `algorithm/` for kernel-based evaluation and analysis.


---

## Citation

If you use this code or ideas from the paper, please cite:

```
@article{lotfian2026promptsplit,
  title={PromptSplit: Revealing Prompt-Level Disagreement in Generative Models},
  author={Lotfian, Mehdi and Jalali, Mohammad and Farnia, Farzan},
  journal={arXiv preprint arXiv:2602.04009},
  year={2026}
}
```

---

## License

This repository is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contributing

We welcome contributions! Please open an issue for questions, bug reports, or feature requests, or submit a pull request with improvements. For major changes, discuss them in an issue first to align with the project's goals.
