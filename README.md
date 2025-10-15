# ldpc-post-selection

**Python package for decoding quantum LDPC codes with cluster-based post-selection**, which is an approach introduced in the paper ["Efficient Post-Selection for General Quantum LDPC Codes"](https://arxiv.org/abs/2510.05795).

## Introduction

In quantum error correction, **post-selection** is a powerful technique to boost the effective fidelity of a computation by discarding low-confidence results. However, the representative **[logical gap](https://doi.org/10.1103/PRXQuantum.5.010302) (or [complementary gap](https://doi.org/10.1038/s41467-025-59714-1)) method** faces major obstacles: it can be computationally expensive (scaling exponentially with the number of logical qubits) and is limited to specific codes like the surface code that allow the MWPM decoder.

[Our work](https://arxiv.org/abs/2510.05795) introduces a new approach to overcome these limitations. We've developed **efficient heuristic decoding confidence metrics based on error cluster statistics from clustering-based decoders (such as [BP+LSD](https://doi.org/10.1038/s41467-025-63214-7)) and utilized them for post-selection**.

Key accomplishments include:
- **General Applicability:** Our method works for a broad range of quantum low-density parity check (QLDPC) codes, not just surface codes.
- **Drastic Error Reduction:** We demonstrate orders of magnitude reduction in logical error rates with only modest abort rates. For example, with the [[144, 12, 12]] bivariate bicycle code, we achieve a ~1000x reduction in the memory logical error rate with just a 1% abort rate at a physical error rate of 0.1%.
- **Real-Time Integration:** We integrated our strategy with real-time decoding via the sliding-window framework, featuring early mid-circuit abort decisions. Crucially, this shows performance that matches or even surpasses global decoding strategies, exhibiting favorable scaling in the number of rounds.

Our post-selection strategies are expected to be particularly useful for offline resource state generation processes (such as magic state preparation) with QLDPC codes.

## Installation

Requires Python >= 3.11

**WARNING:** It’s strongly recommended to use a virtual environment, **especially if the [`ldpc` package](https://github.com/quantumgizmos/ldpc) is already installed**. This package includes a **[modified version of `ldpc`](https://github.com/seokhyung-lee/ldpc)** (where BP+LSD is modified to run LSD regardless of BP converegence & return additional stat info), which can conflict with an existing installation. We will send a pull request to the original repo shortly to resolve this problem.

```bash
git clone --recurse-submodules git@github.com:seokhyung-lee/ldpc-post-selection.git
cd ldpc-post-selection
pip install -e .
# Optional: install codes used for the numerical analyses in our paper.
pip install -e ./simulations
```

## Usage

See the [`basic_usage.ipynb` notebook](examples/basic_usage.ipynb).

## Citation

If you want to cite this package in an academic work, please cite the [arXiv preprint](https://arxiv.org/abs/2510.05795):

```bibtex
@misc{lee2025efficient,
      title={Efficient Post-Selection for General Quantum LDPC Codes}, 
      author={Seok-Hyung Lee and Lucas English and Stephen D. Bartlett},
      year={2025},
      eprint={2510.05795},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2510.05795}, 
}
```

## License

This repository is distributed under the MIT license. Please see the LICENSE file for more details.

## Acknowledgements

This work is supported by the Australian Research Council via the Centre of Excellence in Engineered Quantum Systems (EQUS) Project No. CE170100009, and by the Intelligence Advanced Research Projects Activity (IARPA) through the Entangled Logical Qubits program Cooperative Agreement Number W911NF-23-2-0223.