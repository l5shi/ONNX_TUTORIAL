# ONNX_TUTORIAL
This repository is used to build ONNX model from scratch.




# Installation

## Binaries

A binary build of ONNX is available from [Conda](https://conda.io), in [conda-forge](https://conda-forge.org/):

```
conda install -c conda-forge onnx
```

## Source

You will need an install of protobuf and numpy to build ONNX.  One easy
way to get these dependencies is via
[Anaconda](https://www.anaconda.com/download/):

```
# Use conda-forge protobuf, as default doesn't come with protoc
conda install -c conda-forge protobuf numpy
```

You can then install ONNX from PyPi (Note: Set environment variable `ONNX_ML=1` for onnx-ml):

```
pip install onnx
```

You can also build and install ONNX locally from source code:

```
git clone https://github.com/onnx/onnx.git
cd onnx
git submodule update --init --recursive
python setup.py install
```

Note: When installing in a non-Anaconda environment, make sure to install the Protobuf compiler before running the pip installation of onnx. For example, on Ubuntu:

```
sudo apt-get install protobuf-compiler libprotoc-dev
pip install onnx
```

After installation, run

```
python -c "import onnx"
```

to verify it works.  Note that this command does not work from
a source checkout directory; in this case you'll see:

```
ModuleNotFoundError: No module named 'onnx.onnx_cpp2py_export'
```

Change into another directory to fix this error.

# Testing

ONNX uses [pytest](https://docs.pytest.org) as test driver. In order to run tests, first you need to install pytest:

```
pip install pytest nbval
```

After installing pytest, do

```
pytest
```

to run tests.

# Development

Check out [contributor guide](https://github.com/onnx/onnx/blob/master/docs/CONTRIBUTING.md) for instructions.

# License

[MIT License](LICENSE)

