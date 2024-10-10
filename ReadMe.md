# Mitigation of Over-Squashing in Graph Neural Networks

## Overview

This project focuses on mitigating the issue of over-squashing in Graph Neural Networks (GNNs). Over-squashing occurs when important information is lost during the aggregation of node features in GNNs, which can degrade model performance, particularly on large and complex graphs. This repository implements two novel methods to address this challenge: **FoSR (Feature of Structural Relevance)** and **SDRF (Spectral Decomposition with Residual Filtering)**.

## Methods

### FoSR (Feature of Structural Relevance)
FoSR aims to enhance the representation of node features by considering their structural relevance within the graph. This method focuses on improving information retention during the message-passing phase of GNNs.

### SDRF (Spectral Decomposition with Residual Filtering)
SDRF employs spectral decomposition techniques to filter out noise and enhance the flow of relevant information. This method leverages the graph's spectral properties to minimize the impact of over-squashing.

## Measurement Metrics

To evaluate the effectiveness of the proposed methods, two primary metrics are utilized:

1. **Average Spectral Gap**: This metric measures the separation between the eigenvalues of the graph Laplacian, providing insights into the graph's connectivity and potential information flow.

2. **Commute Time**: This metric assesses the expected time it takes for a random walker to traverse between two nodes in the graph, offering a measure of the overall efficiency of information propagation.

## Dataset

The proposed methods are tested on the **QM9 dataset**, which consists of a diverse set of molecular graphs. This dataset is widely used in graph-based machine learning tasks, making it suitable for assessing the performance of GNNs.

## Installation

Experiment part will come soon
