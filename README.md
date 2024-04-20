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


## Usage

Instructions for using the project should go here.

## Contributing

Instructions for contributing to the project should go here.

## License

Information about the license of the project should go here.