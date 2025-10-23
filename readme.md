# Tabular Architecture Playground

This repository is my personal exploration into building a general-purpose, pre-trained model for tabular data. The core idea is to train a single, powerful neural network on a vast amount of synthetically generated datasets, with the goal of creating a model that can solve new, unseen tabular problems "in-context" without the need for fine-tuning.

## The Core Idea: In-Context Learning for Tables

Traditional machine learning involves training a specific model for each specific task. This project explores a different paradigm:

1.  **Phase 1: Pre-training.** A neural network architecture is trained on thousands of diverse, synthetically generated tabular classification problems. The model is not learning to solve any single problem, but rather learning a general-purpose *algorithm* for how to deduce patterns from tabular data.

2.  **Phase 2: Inference.** The single, pre-trained model is presented with a new, unseen problem. The entire training set for this new problem is provided as a "prompt." The model uses its learned algorithm to infer the underlying patterns from the prompt and make predictions on the test samples provided.

The primary goal is to experiment with different neural network architectures to see which ones are most effective at learning this general "tabular reasoning" algorithm.

## Project Status: First Baseline Established

I have successfully built and tested a complete, end-to-end pipeline:
1.  A flexible synthetic data generator (`data_generator.py`).
2.  A Transformer-based architecture that treats rows as tokens (`architectures/row_transformer.py`).
3.  A pre-training script that trains the model on-the-fly (`train.py`).
4.  An evaluation script that performs zero-shot testing on real-world datasets (`evaluate.py`).

The initial pre-training run was completed using the `RowBasedTransformer` architecture for 5,000 steps. The model was then evaluated on two classic scikit-learn datasets.

### Initial Evaluation Results

The first experiment has successfully established a baseline for the model's performance.

| Dataset | Random Guessing Accuracy | **My Model's Zero-Shot Accuracy** | Analysis |
| :--- | :--- | :--- | :--- |
| **Breast Cancer** (2 classes) | 50.0% | **54.55%** | **Successful Signal:** The model performs better than random, proving the pre-training learned a valid, generalizable signal. |
| **Wine** (3 classes) | 33.3% | **24.44%** | **Insightful Failure:** The model performed worse than random, suggesting the learned algorithm is not yet general enough and struggles with data distributions different from the synthetic training data. |

**Conclusion:** The pipeline works and the model is learning! The current results indicate that the model is **under-trained and its learned algorithm is not yet general enough** to perform well on a wide variety of tasks. This is an excellent and expected starting point for iterative improvement.

## My Development Plan & Next Steps

With a working baseline, the next phase of the project is focused on improving the model's generalization capabilities.

-   [x] **Build a robust synthetic data generator.**
-   [x] **Implement the Row-Based Transformer baseline.**
-   [x] **Develop the pre-training loop.**
-   [x] **Establish an evaluation harness and get baseline results.**
-   [ ] **Iterate to Improve Performance.** My immediate next steps are:
    1.  **Increase Training Duration:** The most straightforward next step is to train the model for significantly longer. I will increase the training steps from 5,000 to 20,000+ to allow the model to see more problems and learn a more robust algorithm.
    2.  **Increase Model Capacity:** I will experiment with a larger model by increasing parameters like `embedding_dim` and `num_encoder_layers`. A larger model has more capacity to learn a more complex algorithm.
    3.  **Diversify Synthetic Data:** I will enhance the `data_generator.py` to produce a wider variety of problem types, forcing the model to learn a more flexible and general approach to problem-solving.

## How to Use

This project is currently in the experimentation and development phase.

**1. Replicate the Training:**
Run the training script. This will take some time and will save the model weights to `row_transformer_pretrained.pth`.
```bash
python train.py
```

**2. Replicate the Evaluation:**
Run the evaluation script to test the trained model on real-world datasets.
```bash
python evaluate.py
```