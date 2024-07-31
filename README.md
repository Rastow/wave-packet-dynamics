# wave-packet-dynamics

## Installation
The package is available on the 
[Python Package Index](https://pypi.org/project/wave-packet-dynamics/)
and can be installed using the following command.
```bash
pip install wave-packet-dynamics
```

## Features
This package provides a framework for simulating the time evolution of
a single wave packet in an arbitrary time-independent potential.
Currently, only one-dimensional simulations are supported. Additionally,
the package provides a simple interface for visualizing the results of
the simulations.

## Documentation
The [documentation](https://wave-packet-dynamics.readthedocs.io) is made
with [Material for MkDocs](https://github.com/squidfunk/mkdocs-material)
and is hosted by [ReadTheDocs](https://about.readthedocs.com/).

## Development
The project uses [PDM](https://github.com/pdm-project/pdm/) as the 
package manager. After cloning the repository and installing PDM, run 
the following commands to create a virtual environment and install the 
development dependencies.
```bash
pdm venv create
pdm install
```
Afterward, several useful commands for formatting, linting, testing and 
type checking are available.
```bash
pdm run --list
```

## License

The project is distributed under the terms of the 
[BSD-3-Clause](https://spdx.org/licenses/BSD-3-Clause.html) license.