import argparse
import wandb
from keras.datasets import fashion_mnist
import numpy as np
from implementations import FeedForwardNeuralNetwork, preprocess_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-wp", "--wandb_project", type=str, default="myprojectname", 
                        help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we", "--wandb_entity", type=str, default="myname", 
                        help="Wandb Entity used to track experiments in the Weights & Biases dashboard")
    
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist", 
                        help="Dataset to use")
    parser.add_argument("-e", "--epochs", type=int, default=10, 
                        help="Number of epochs to train neural network")
    parser.add_argument("-b", "--batch_size", type=int, default=16, 
                        help="Batch size used to train neural network")
    
    parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], default="cross_entropy", 
                        help="Loss function to use")
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="rmsprop", 
                        help="Optimizer to use")
    
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0004117813426264185, 
                        help="Learning rate used to optimize model parameters")
    parser.add_argument("-m", "--momentum", type=float, default=0.5, 
                        help="Momentum used by momentum and nag optimizers")
    parser.add_argument("-beta", "--beta", type=float, default=0.5, 
                        help="Beta used by rmsprop optimizer")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.5, 
                        help="Beta1 used by adam and nadam optimizers")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.5, 
                        help="Beta2 used by adam and nadam optimizers")
    parser.add_argument("-eps", "--epsilon", type=float, default=0.000001, 
                        help="Epsilon used by optimizers")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.00012243241893688942, 
                        help="Weight decay used by optimizers")
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "Xavier"], default="Xavier", 
                        help="Weight initialization method")
    
    parser.add_argument("-nhl", "--num_layers", type=int, default=3, 
                        help="Number of hidden layers used in feedforward neural network")
    parser.add_argument("-sz", "--hidden_size", type=int, default=32, 
                        help="Number of hidden neurons in a feedforward layer")
    
    parser.add_argument("-a", "--activation", type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], default="tanh", 
                        help="Activation function to use")
    args = parser.parse_args()
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": args.optimizer,
            "learning_rate": args.learning_rate,
            "activation": args.activation,
            "weight_init": args.weight_init,
            "weight_decay": args.weight_decay,
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size
        }
    )
    
    # Load dataset
    if args.dataset == "fashion_mnist":
        (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    else:  # Handle MNIST if needed
        from keras.datasets import mnist
        (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    
    # Split training data into training and validation sets (90% train, 10% validation)
    num_train_samples = X_train_full.shape[0]
    val_size = int(0.1 * num_train_samples)
    X_val, y_val = X_train_full[:val_size], y_train_full[:val_size]
    X_train, y_train = X_train_full[val_size:], y_train_full[val_size:]
    
    # Preprocess data
    X_train_processed, y_train_onehot, X_val_processed, y_val_onehot = preprocess_data(X_train, y_train, X_val, y_val)
    X_test_processed, y_test_onehot, _, _ = preprocess_data(X_test, y_test, X_test, y_test)
    
    # Set up model architecture
    input_size = X_train_processed.shape[1]
    output_size = 10
    hidden_layers = [args.hidden_size] * args.num_layers
    
    # Initialize the model
    model = FeedForwardNeuralNetwork(
        input_size=input_size,
        output_size=output_size,
        hidden_layers=hidden_layers,
        activation=args.activation.lower(),  # Convert to lowercase to match implementation
        loss=args.loss,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        weight_init=args.weight_init.lower(),  # Convert to lowercase to match implementation
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        beta=args.beta,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon
    )
    
    # Train the model
    history = model.fit(
        X_train_processed, y_train_onehot,
        epochs=args.epochs, batch_size=args.batch_size,
        X_val=X_val_processed, y_val=y_val_onehot
    )
    
    # Log metrics during training
    for epoch in range(args.epochs):
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": history["train_loss"][epoch],
            "val_loss": history["val_loss"][epoch] if history["val_loss"] else np.nan,
            "train_accuracy": history["train_accuracy"][epoch],
            "val_accuracy": history["val_accuracy"][epoch]
        })
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test_processed, y_test_onehot)
    wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy})
    
    print(f"Training completed with test accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()