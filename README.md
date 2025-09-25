# Project Lumina

Project Lumina is a collection of Fraud Detection algorithms using Graph Neural Networks.
It uses **MLFlow** for tracking model training and validation, and the models are developed using **PyTorch**.
The models are containerized using Docker and deployed on a K8s cluster with API endpoints.

# Dependencies

The python package dependencies are all defined in the `requirements.txt` with the correct version.
In order to create the python virtual environment and install all the deps, run the command

```bash
make installdeps
```

# Usage

In order to download the dataset and preprocess the data you can run the command

```bash
make prepdata
```

In order to train a model, first define it in `src/models.py` and add the configuration for
it to `config.yaml`. Then run the command

```bash
make train
```

This make recipe will make sure the data is downloaded and preprocessed, and trains all the
models defined and configured.
