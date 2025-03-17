# DA6401 - Assignment 1

## Links
- [Weights & Biases Report](https://wandb.ai/ns25z061-indian-institute-of-technology-madras/fashion_mnist_ffnn/reports/DA6401-Assignment-1--VmlldzoxMTgzMTQ5NQ?accessToken=9bqx0bksr03ba80v4u2008ee88xseq52qlrzrpe8rn1tcn4nmldqkpenx9qt6dmg)
- [GitHub Repository](https://github.com/SherryS997/DA6401-Assignment-1)

## Code Files

### [train.py](https://github.com/SherryS997/DA6401-Assignment-1/blob/main/train.py)
Main training script for Fashion-MNIST classification with command-line arguments. Handles model training, evaluation, and logging metrics to Weights & Biases.

### [implementations.py](https://github.com/SherryS997/DA6401-Assignment-1/blob/main/implementations.py)
Core implementation of neural network components including activation functions, loss functions, optimizers, and the FeedForwardNeuralNetwork class. Also contains data preprocessing utilities.

### [sweep.py](https://github.com/SherryS997/DA6401-Assignment-1/blob/main/sweep.py)
Hyperparameter optimization script using Optuna and Weights & Biases. Performs systematic search for optimal model architecture and training parameters.

### [compare.py](https://github.com/SherryS997/DA6401-Assignment-1/blob/main/compare.py)
Comparison script for evaluating different loss functions (Cross Entropy vs Mean Squared Error). Trains models with identical hyperparameters to isolate impact of loss function selection.

### [cm.py](https://github.com/SherryS997/DA6401-Assignment-1/blob/main/cm.py)
Visualization script that generates and logs confusion matrices and performance metrics. Creates detailed visualizations of model predictions and class-wise performance.

### [wandb_plots.py](https://github.com/SherryS997/DA6401-Assignment-1/blob/main/wandb_plots.py)
Helper script to create and log sample dataset visualizations to Weights & Biases. Displays representative examples of the Fashion-MNIST dataset classes.