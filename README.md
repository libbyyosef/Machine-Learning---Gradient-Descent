# Gradient-Descent

This repository contains an implementation of a generic Gradient Descent algorithm along with visualizations of its performance on different objective functions. The Gradient Descent algorithm is implemented in Python, and various learning rate strategies and objective functions are explored.

**GradientDescent Class**

The GradientDescent class is the core component of this implementation, providing a flexible framework for applying the Gradient Descent algorithm to various optimization problems. It facilitates the exploration of different solution types and allows for detailed investigation of algorithm properties through customizable callbacks.

**Learning Rate Strategies**

**Constant (Fixed) Learning Rate (FixedLR):** This strategy maintains a constant learning rate throughout the optimization process. It provides stability but may converge slowly or struggle to adapt to changing gradients.

**Exponentially Decaying Learning Rate (ExponentialLR):** This strategy dynamically adjusts the learning rate by exponentially decaying it over time. It allows for faster convergence in the initial stages of optimization while ensuring stability in later stages.

**Objective Functions (Modules)**

The implementation includes various objective functions derived from the BaseModule class, which defines a generic abstract form for any objective to be minimized using Gradient Descent. These functions, including compute_output and compute_jacobian, enable the computation of the function value and its derivative at a given point, facilitating optimization.

**Regularized Logistic Regression**

The implementation extends Gradient Descent to solve a regularized logistic regression optimization problem. This involves:

- **LogisticModule:** This module computes the negative log-likelihood of logistic regression, capturing the probability of outcomes given input features and model parameters.
- **RegularizedModule:** This module incorporates regularization terms, such as L1 and L2 regularization, to prevent overfitting and improve generalization.
- **LogisticRegression Class:** This class encapsulates the logistic regression model, integrating the Gradient Descent implementation with the logistic module and regularization techniques.


![Screenshot 2024-02-01 030212](https://github.com/libbyyosef/Machine-Learning---Gradient-Descent-/assets/36642026/d7e7bf6f-ea3e-4c87-a88a-40fef84a7546)


![Screenshot 2024-02-01 030218](https://github.com/libbyyosef/Machine-Learning---Gradient-Descent-/assets/36642026/a98afe2f-f8d8-46ae-b511-fee2b1f8f544)


![Screenshot 2024-02-01 030245](https://github.com/libbyyosef/Machine-Learning---Gradient-Descent-/assets/36642026/3c68bbb6-598e-4b32-ad1c-e27b9548540b)


![Screenshot 2024-02-01 030252](https://github.com/libbyyosef/Machine-Learning---Gradient-Descent-/assets/36642026/116ba533-832f-4986-ba19-91dc252ed394)


![Screenshot 2024-02-01 030258](https://github.com/libbyyosef/Machine-Learning---Gradient-Descent-/assets/36642026/5fc9d4ae-d17e-477f-b26b-38679d6758d0)


![Screenshot 2024-02-01 030310](https://github.com/libbyyosef/Machine-Learning---Gradient-Descent-/assets/36642026/dce92567-3d14-4665-899b-d9775e94c66f)


![Screenshot 2024-02-01 030317](https://github.com/libbyyosef/Machine-Learning---Gradient-Descent-/assets/36642026/f974beed-f56f-4afa-b144-d94c8aabd26a)


![Screenshot 2024-02-01 030323](https://github.com/libbyyosef/Machine-Learning---Gradient-Descent-/assets/36642026/58a859db-1b9b-4fd7-800a-5cdfd14c7cfa)


![Screenshot 2024-02-01 030328](https://github.com/libbyyosef/Machine-Learning---Gradient-Descent-/assets/36642026/f6784bc8-4015-4ca5-9ea6-9f6828f43a8f)


![Screenshot 2024-02-01 030334](https://github.com/libbyyosef/Machine-Learning---Gradient-Descent-/assets/36642026/9beb6a22-0f9f-4878-839a-53e3f5379a8b)


![Screenshot 2024-02-01 030341](https://github.com/libbyyosef/Machine-Learning---Gradient-Descent-/assets/36642026/fe69aa6e-cd32-474c-88b0-1c1649169016)


![Screenshot 2024-02-01 030347](https://github.com/libbyyosef/Machine-Learning---Gradient-Descent-/assets/36642026/d05d686c-652e-49ee-9b2f-53801e9af2af)






