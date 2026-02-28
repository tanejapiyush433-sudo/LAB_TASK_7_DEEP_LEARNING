# LAB_TASK_7_DEEP_LEARNING

📘 UCS761 – Lab 7
From Numbers to Vision: Building, Breaking, Comparing

Name: Piyush Taneja
Roll No: 102303307
Course: UCS761

🔍 Overview

This lab focuses on implementing neural networks completely from scratch using NumPy, without using:

PyTorch

TensorFlow

Keras

Scikit-learn neural network modules

Automatic differentiation tools

The goal of this assignment is not just to make models run, but to deeply understand:

Representation learning

Effect of depth

Activation behavior

Optimization strategies

Gradient stability

Dense vs Convolutional scaling

Generalization

🧠 Part 1 – Deep Networks on Numeric Data
Dataset

Synthetic dataset generated as:

𝑥
1
∼
𝑈
(
−
2
,
2
)
x
1
	​

∼U(−2,2)
𝑥
2
∼
𝑈
(
−
2
,
2
)
x
2
	​

∼U(−2,2)

Target:

𝑦
=
{
1
	
if 
𝑥
1
2
+
𝑥
2
2
>
1.5


0
	
otherwise
y={
1
0
	​

if x
1
2
	​

+x
2
2
	​

>1.5
otherwise
	​


Data split:

70% Training

15% Validation

15% Test

Architectures Built

2-layer network (2–8–1)

5-layer network (2–8–8–8–8–1)

10-layer network (2–8–8–8–8–8–8–8–8–1)

Hidden activations tested:

ReLU

Sigmoid

Optimizers tested:

SGD

Momentum

Adam

Key Observations

Increasing depth does not always improve validation performance.

Sigmoid networks suffer from vanishing gradients in deeper architectures.

ReLU maintains gradient flow better in deep networks.

Adam converges significantly faster than SGD and Momentum.

🖼 Part 2 – Dense vs CNN
Dataset

Synthetic 8×8 image dataset:

Class 0 → Vertical center line

Class 1 → Horizontal center line

Added Gaussian noise (σ = 0.1)

Models Compared

Dense Network (Flattened 64 inputs)

CNN (3×3 convolution + ReLU + Flatten + Dense)

Parameter Comparison
Model	Parameters
2-layer Dense	33
5-layer Dense	249
10-layer Dense	537
CNN	47
Key Observations

CNN achieves higher test accuracy with fewer parameters.

Dense networks overfit more easily.

CNN generalizes better due to:

Parameter sharing

Local receptive fields

Spatial feature extraction

⚙ Part 3 – Optimizer Comparison (CNN)

Optimizers compared:

SGD

Momentum

Adam

Convergence Ranking

Adam (Fastest)

Momentum

SGD (Slowest)

Observations

Adam reaches near-zero loss within 1–2 epochs.

Momentum accelerates convergence compared to SGD.

SGD is stable but slower.

Optimizer choice significantly affects convergence speed.
