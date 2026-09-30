# iRevRNN

This repository contains a PyTorch reversible-RNN implementation, C++/CUDA
extension sources, data-preparation utilities, and MNIST/action example
scripts. It is a source checkout rather than a packaged Python distribution:
there is no root `requirements.txt`, `pyproject.toml`, or `setup.cfg`.

## Build requirements

The extension build in `model/setup.py` defines both a C++ extension and a
CUDA extension. Building it therefore requires a PyTorch installation
compatible with the local compiler toolchain and CUDA toolkit, in addition to
Python setuptools.

```bash
python -m pip install torch numpy
cd model
python setup.py build_ext --inplace
```

The extension sources are under `model/torch_irevrnn/`. The `irevrnn` module
imports the compiled C++ and CUDA modules, so the extension build is needed
before importing it. This repository does not provide a tested platform or
toolchain matrix. The example scripts also import `torchvision` and
`torch.utils.tensorboard`; those example dependencies are not declared in a
repository dependency manifest.

## Examples and data

- `mnist_main.py` requests MNIST through `torchvision.datasets.MNIST` with
  `download=True`, so the dataset is fetched at runtime rather than stored in
  this repository. As checked in, however, that script imports
  `irevrnn_mnist_model`, which is not present in the tree; the available
  similarly named file is `irevrnn_mnist_action_model.py`. The MNIST example
  therefore cannot run from this checkout without resolving that missing
  module.
- `action_main.py` and the files in `action/` implement an action-data
  pipeline. The action dataset itself is not included; the training code also
  uses CUDA operations directly.
- `model/torch_irevrnn/irevrnn.py` contains the `IRevRNN` implementation.

MNIST is a public dataset accessed through torchvision. No action dataset or
trained checkpoint is supplied here, and no particular external action-data
or checkpoint source is specified by this repository.

## Source layout

- `model/torch_irevrnn/` — Python, C++, and CUDA implementation.
- `model/setup.py` — native extension build definition.
- `mnist_main.py` — MNIST experiment script (currently references a missing
  module; see above).
- `action_main.py`, `irevrnn_mnist_action_model.py` — action example code.
- `action/` — action-data readers and conversion utilities.
