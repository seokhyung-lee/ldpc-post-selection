# ldpc-post-selection

Decoding with post-selection for general quantum LDPC codes

## Installation

Requires Python >= 3.11

**WARNING**: It’s recommended to use a virtual environment, especially if the [`ldpc` package](https://github.com/quantumgizmos/ldpc) is already installed. This package includes a [modified version of `ldpc`](https://github.com/seokhyung-lee/ldpc) (where BP+LSD is modified to run LSD regardless of BP converegence & return additional stat info), which can conflict with an existing installation.

```bash
git clone --recurse-submodules git@github.com:seokhyung-lee/ldpc-post-selection.git
cd ldpc-post-selection
pip install -e .
```