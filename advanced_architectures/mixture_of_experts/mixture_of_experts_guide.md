# Mixture of Experts (MoE)

Mixture of Experts (MoE) is a neural network architecture technique designed to increase model capacity (number of parameters) without a proportional increase in computational cost per input. It achieves this by replacing certain dense layers (typically the feed-forward networks in Transformers) with multiple parallel "expert" sub-networks and dynamically routing each input token to only a small subset of these experts.

## Core Idea

Imagine instead of having one large feed-forward network (FFN) process every token, you have several smaller, specialized FFNs (the "experts"). For each incoming token, a small "gating network" or "router" looks at the token and decides which one or two experts are best suited to process it. The token is then only sent to those selected experts.

## Architecture Components

1.  **Experts:** These are the individual neural networks (often FFNs) that perform the core computation. There can be many experts (e.g., 8, 16, 64, or more).
2.  **Gating Network (Router):** A smaller network (e.g., a simple linear layer) that takes the token representation as input and outputs probabilities or scores indicating how suitable each expert is for processing that token.
3.  **Dispatcher:** Routes the token to the selected expert(s) based on the gating network's output (typically the top-k, where k is often 1 or 2).
4.  **Combiner:** Aggregates the outputs from the selected expert(s) for each token, often by taking a weighted sum based on the gating network's scores.

## Benefits

*   **Scalability:** Allows for building models with trillions of parameters while keeping the compute cost per token manageable. Only the activated experts contribute to the FLOPs for a given token.
*   **Computational Efficiency:** For the same number of *total* parameters, an MoE model requires significantly fewer FLOPs per token during inference compared to a dense model.
*   **Potential for Specialization:** Experts might learn to specialize in different types of inputs, patterns, or knowledge domains, although the extent and nature of this specialization are still areas of research.

## Challenges

*   **Load Balancing:** A major challenge is ensuring that tokens are distributed relatively evenly across experts during training. If the gating network consistently sends most tokens to only a few experts, those experts become bottlenecks, and others are underutilized. This is typically addressed with auxiliary "load balancing" loss functions during training that encourage uniform routing.
*   **Training Complexity:** MoE models are more complex to train due to the routing mechanism and the need for load balancing.
*   **Communication Overhead (Distributed Training):** In large-scale distributed settings, routing tokens to experts located on different devices can introduce significant communication overhead.
*   **Inference Complexity:** While FLOPs per token are lower, the conditional computation and routing add complexity to the inference process.

## Implementation

MoE layers typically replace the FFN layers within a Transformer block. The gating mechanism selects the top-k experts (k=1 or k=2 are common), computes the outputs from those experts, and combines them using the gating scores as weights. 