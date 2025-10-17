# Tabular Architecture Playground

This repository is my personal exploration into building a general-purpose, pre-trained model for tabular data. The core idea is to train a single, powerful neural network on a vast amount of synthetically generated datasets, with the goal of creating a model that can solve new, unseen tabular problems "in-context" without the need for fine-tuning.

## The Core Idea: In-Context Learning for Tables

Traditional machine learning involves training a specific model for each specific task (e.g., a churn model, a credit risk model). This project explores a different paradigm:

1.  **Phase 1: Pre-training.** A neural network architecture is trained on thousands of diverse, synthetically generated tabular classification problems. The model is not learning to solve any single problem, but rather learning a general-purpose *algorithm* for how to deduce patterns from tabular data.

2.  **Phase 2: Inference.** The single, pre-trained model is presented with a new, unseen problem. The entire training set for this new problem is provided as a "prompt" or "context" to the model in a single forward pass. The model uses its learned algorithm to infer the underlying patterns from the prompt and make predictions on the test samples provided.

The primary goal is to experiment with different neural network architectures to see which ones are most effective at learning this general "tabular reasoning" algorithm.

## Project Structure

This project is organized into a few key components:

-   `data_generator.py`: A script responsible for creating an infinite stream of diverse, synthetic tabular classification datasets. It uses `scikit-learn` to generate data with various underlying structures (linear, clustered, non-linear) and complexities.

-   `architectures/`: This directory will contain the implementation of the different neural network models I am experimenting with. Each architecture will be designed to handle a sequence of data points and perform in-context learning.

-   `train.py`: The main script for pre-training a chosen architecture on the synthetic data generated on the fly.

-   `evaluate.py`: A script to benchmark the pre-trained models on a suite of real-world datasets (e.g., from the UCI Machine Learning Repository) to evaluate their generalization capabilities.

## My Architectural Experiments

The central part of this project is to implement and compare different architectures. My plan is to start with a baseline and then explore more creative ideas.

### 1. The Baseline: Row-Based Transformer

-   **Concept:** Treat each data point (a row) as a "token" in a sequence. An initial embedding layer converts each row into a high-dimensional vector, and a standard Transformer Encoder processes the entire sequence of rows.
-   **Hypothesis:** The self-attention mechanism will allow the model to learn complex relationships and similarities between all data points in the provided context, effectively creating a powerful, learned version of an algorithm like k-nearest neighbors.

### 2. The Follow-up: Hybrid (CNN + Transformer)

-   **Concept:** Before feeding the sequence of row embeddings into the Transformer, first pass it through several layers of 1D Convolutional Neural Networks (CNNs).
-   **Hypothesis:** The CNN layers can act as "local pattern extractors," identifying meaningful features among small, local groups of data points. The Transformer can then use these richer, pre-processed features to model the more complex, global relationships across the entire dataset.

Further architectural ideas may be explored after establishing a solid baseline with these two.

## My Development Plan

1.  **[Done]** **Build a robust synthetic data generator.** The quality and diversity of the pre-training data are fundamental to the project's success.
2.  **[In Progress]** **Implement the Row-Based Transformer baseline.** This involves creating the row-embedding module and integrating it with a standard Transformer Encoder.
3.  **Develop the pre-training loop.** This script will generate data on the fly and train the model, calculating loss only on the "test" portion of the in-context prompt.
4.  **Establish an evaluation harness.** Create a standardized way to test the pre-trained model on a set of real-world tabular datasets to measure its zero-shot performance.
5.  **Iterate and experiment.** Once the baseline is established, I will implement and test other architectures, like the Hybrid CNN model, and compare their performance.

## How to Use

*(This section will be updated as the project progresses)*

Currently, the project is in the development phase. You can inspect `data_generator.py` to see how the synthetic data is created.

```bash
# To see an example of a generated dataset
python data_generator.py
```

---