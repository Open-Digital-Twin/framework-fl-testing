# A Framework for testing Federated Learning algorithms using an edge-like environment

This repository contains the implementation for the article: <add-link>

## Introduction

Federated Learning (FL) is a machine learning paradigm where many clients cooperatively train a single centralized model while keeping their data private and decentralized. FL is commonly used in edge computing, which involves placing computer workloads (both hardware and software) as close as possible to the edge, where the data is being created and where actions are occurring, enabling faster response times, greater data privacy, and reduced data transfer costs. However, due to the heterogeneous data distributions/contents of clients, it is non-trivial to accurately evaluate the contributions of local models in global centralized model
aggregation. This has been a major challenge in FL, commonly known as data imbalance or class imbalance. Previous work has been proposed to address this issue such as Deep Reinforcement Learning (DRL) algorithms to dynamically learn the weight of the
contributions of each client at each round. Testing and assessing this FL algorithms can be a very difficult and complex task due to the distributed nature of the systems. In this work, a literature review of these concepts is presented in order to introduce the reader enough context of this challenge and a distributed edge-like environment framework is proposed to assess FL algorithms in a more easy and scalable way.

## Repository Structure

```
├── client      ----> Source code for FL client
├── docker      ----> Dockerfiles for images of client and server
├── docs        ----> Documentation
├── graphs
├── images
├── kubernetes  ----> Helm charts used to deploy experiments to kubernetes
├── scripts     ----> Scripts used to create FL testing environment
├── server      ----> Source code for FL server
├── .gitignore
├── client.env  ----> Environment variables for clients
├── common.env  ----> Common variables for clients and server
├── docker-composer.yaml    ----> Docker Composer configuration
├── Makefile    ----> Makefile for quick setup configuration
├── README.md
└── server.env

```
## Instalation

The project was created using [Poetry](https://python-poetry.org/). To locally install dependencies for either the server and client:

```
make install-dependencies
```

## Usage

To use the framework locally, you can use [Docker Compose](https://docs.docker.com/compose/).

```
docker-compose up
```

This will spin up locally, using Docker, the clients and servers defined in the [docker-compose.yaml](docker-composer.yaml) file.


## Configuration

Client and server configuration are done through the `.env` files, which will define the environment variables needed for the experiment. 

### common.env

Has all common configuration done in both client and server:

- CERTIFICATES_PATH `str`
    - Path for SSL certficates (don't change)
- EXPERIMENT_PATH `str`
    - Experiment output path
- SERVER_ADDRESS `str`
    - Server internal IP address (don't change)
- EXPERIMENT_NAME `str`
    - Name for the experiment you will run
- CLIENT_LOCAL_EPOCHS `int`
    - Number of client local training epochs
- SERVER_NAME=fl-framework-server
    - Name for the server container
- DATASET_NAME `str`
    - Dataset to be used. Options include `cifar-10`, `cifar-100` and `fmnist`
- DATASET_NIID `int`
    - If the dataset distribution shoud be Non-IID or not. Should be `1` or `0`
- DATASET_PARTITION `str`
    - How is the classes distribution in the clients
- DATASET_BALANCE `int`
    - If the number of samples is unbalanced or balanced 
- DATASET_BATCHES `int`
    - Number of batches to divide the dataset
- MODEL `str`
    - Which model will be used for the experiment. 
- DATASET_CLASS_PER_CLIENT `int`
    - Number of classes per client.

### server.env

Has all configuration for the server.

- SERVER_NUM_ROUNDS `int`
    - Number of server rounds
- SERVER_STRATEGY `str`
    - Which global model aggregation strategy to be used
- FRACTION_FIT `int`
    - Percentage of fit clients to fit
- MIN_FIT_CLIENTS `int`
    - Mininum number of clients to fit
- MIN_AVAILABLE_CLIENTS `int`
    - Mininum number of clients needed to start training


