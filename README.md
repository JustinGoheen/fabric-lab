# Fabric Lab

## Overview

Fabric Lab is a public template for artificial intelligence and machine learning research projects using Lightning AI's [Lightning Fabric](https://lightning.ai/docs/fabric/stable/).

The recommended way for Fabric Lab users to create new repos is with the [use this template](https://github.com/new?template_name=fabric-lab&template_owner=JustinGoheen) button.

The example uses a language transformer borrowed from the PyTorch Lightning demos.

## Source Module

`fabriclab.core` contains code for the Lightning Module and Trainer.

`fabriclab.components` contains tasks grouped by purpose for cohesion.

`fabriclab.pipeline` contains code for data acquistion and preprocessing, and building a TorchDataset.

`fabriclab.serve` contains code for model serving APIs built with [FastAPI](https://fastapi.tiangolo.com/project-generation/#machine-learning-models-with-spacy-and-fastapi).

`fabriclab.cli` contains code for the command line interface built with [Typer](https://typer.tiangolo.com/)and [Rich](https://rich.readthedocs.io/en/stable/).

`fabriclab.app` contains code for data apps built with Streamlit.

`fabriclab.config` assists with project, trainer, and sweep configurations.

## Project Root

`data` directory should be used to cache the TorchDataset and training splits locally if the size of the dataset allows for local storage. additionally, this directory should be used to cache predictions during HPO sweeps.

`docs` directory should be used to store technical documentation.

`logs` directory will store logs generated from experiment managers and profilers.

`models` directory will store training checkpoints and the pre-trained ONNX model.

`notebooks` directory can be used to present exploratory data analysis, explain math concepts, and create a presentation notebook to accompany a conference style paper.

`requirements` directory should mirror base requirements and extras found in setup.cfg. the requirements directory and _requirements.txt_ at root are required by the basic CircleCI GitHub Action.

`tests` module contains unit and integration tests targeted by pytest.

`setup.py` `setup.cfg` `pyproject.toml` and `MANIFEST.ini` assist with packaging the Python project.

`.pre-commit-config.yaml` is required by pre-commit to install its git-hooks.

## Base Requirements and Extras

Fabric Lab installs minimal requirements out of the box, and provides extras to make creating robust virtual environments easier. To view the requirements, in [setup.cfg](setup.cfg), see `install_requires` for the base requirements and `options.extras_require` for the available extras.

The recommended install is as follows:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[full, { domain extra(s) }]"
```

where { domain extra(s) } is one of, or some combination of (vision, text, audio, rl, forecast) e.g.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[full, text]"
```

## Using Fabric Lab

After installing fabric-lab, you can run the demo with:

```sh
lab run demo
```
