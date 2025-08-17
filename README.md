🔁 Gated Feedback MLP

An experimental Multi-Layer Perceptron (MLP) architecture with gated feedback connections that iteratively refine representations. Inspired by ideas from recurrent networks, feedback CNNs, and deep equilibrium models, this project explores whether simple feedback loops in MLPs can improve performance on standard benchmarks.

✨ Features

🔹 Basic MLP backbone with configurable number of layers.

🔹 Gated feedback mechanism: hidden states are fed back into the model, allowing iterative refinement.

🔹 Leaky ReLU activations for improved stability over vanilla ReLU.

🔹 PyTorch implementation, modular and easy to extend.

🔹 Experiments on:

CIFAR-10 (image classification)

MNIST (digit recognition)

Two Moons (toy 2D dataset)

📊 Results Summary
Dataset	Single-Pass Accuracy	Iterative Best Accuracy	Notes
CIFAR-10	~46% (2-layer MLP)	~56% (after 4 iterations)	Largest gains seen
MNIST	Moderate	Small improvements	Converges quickly
Two Moons	Baseline separation	Improved cluster boundaries	More defined decision boundary
