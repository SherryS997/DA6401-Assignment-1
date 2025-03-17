import numpy as np
import wandb
from tensorflow.keras.datasets import fashion_mnist
from implementations import FeedForwardNeuralNetwork, preprocess_data

# Load Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Preprocess data
X_train_processed, y_train_onehot, X_test_processed, y_test_onehot = preprocess_data(
    X_train, y_train, X_test, y_test
)

input_size = X_train_processed.shape[1]
output_size = 10

# Best hyperparameters from the sweep
best_params = {
    "activation": "relu",
    "batch_size": 16,
    "epochs": 5,
    "hidden_layers": [128, 128, 128, 128, 128],
    "learning_rate": 0.0004137305229308512,
    "optimizer": "adam",
    "weight_decay": 0.00011828729376021434,
    "weight_init": "xavier"
}

# Function to train and evaluate model with a specific loss function
def train_and_evaluate(loss_function):
    # Initialize wandb
    run = wandb.init(
        project="fashion_mnist_ffnn", 
        name=f"loss_comparison_{loss_function}",
        reinit=True
    )
    
    # Log hyperparameters
    wandb_config = best_params.copy()
    wandb_config["loss"] = loss_function
    wandb.config.update(wandb_config)
    
    print(f"\nTraining model with {loss_function} loss function:")
    
    # Initialize model
    model = FeedForwardNeuralNetwork(
        input_size=input_size,
        output_size=output_size,
        hidden_layers=best_params["hidden_layers"],
        activation=best_params["activation"],
        loss=loss_function,
        optimizer=best_params["optimizer"],
        learning_rate=best_params["learning_rate"],
        weight_init=best_params["weight_init"],
        weight_decay=best_params["weight_decay"]
    )
    
    # Train model
    history = model.fit(
        X_train_processed, y_train_onehot,
        epochs=best_params["epochs"],
        batch_size=best_params["batch_size"]
    )
    
    # Log training metrics per epoch
    for epoch in range(best_params["epochs"]):
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": history["train_loss"][epoch],
            "train_accuracy": history["train_accuracy"][epoch]
        })
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test_processed, y_test_onehot)
    print(f"Test accuracy with {loss_function}: {test_accuracy:.4f}")
    print(f"Test loss with {loss_function}: {test_loss:.4f}")
    
    # Log final metrics
    wandb.log({
        "final_train_loss": history["train_loss"][-1],
        "final_train_accuracy": history["train_accuracy"][-1],
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    })
    
    # Calculate predictions for test data
    y_pred = model.predict(X_test_processed)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test_onehot, axis=1)
    
    # Log confusion matrix data
    wandb.log({
        "correct_predictions": (y_pred_classes == y_test_classes).sum(),
        "total_predictions": len(y_test_classes),
        "loss_function": loss_function
    })
    
    # Finish wandb run
    run.finish()
    
    return {
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "train_history": history
    }

def main():
    # Train and evaluate with Cross Entropy loss
    print("Starting comparison of Cross Entropy vs Mean Squared Error loss functions")
    ce_results = train_and_evaluate("cross_entropy")
    
    # Train and evaluate with Mean Squared Error loss
    mse_results = train_and_evaluate("mean_squared_error")
    
    # Print comparison summary
    print("\nComparison Summary:")
    print(f"Cross Entropy - Test Accuracy: {ce_results['test_accuracy']:.4f}, Test Loss: {ce_results['test_loss']:.4f}")
    print(f"Mean Squared Error - Test Accuracy: {mse_results['test_accuracy']:.4f}, Test Loss: {mse_results['test_loss']:.4f}")
    
    accuracy_diff = ce_results['test_accuracy'] - mse_results['test_accuracy']
    print(f"Accuracy Difference (CE - MSE): {accuracy_diff:.4f}")
    
    # Create a final wandb run to log comparison directly
    run = wandb.init(project="fashion_mnist_ffnn", name="loss_comparison_summary", reinit=True)
    
    # Log comparison metrics
    wandb.log({
        "cross_entropy_test_accuracy": ce_results['test_accuracy'],
        "mean_squared_error_test_accuracy": mse_results['test_accuracy'],
        "accuracy_difference": accuracy_diff,
        "cross_entropy_final_train_loss": ce_results['train_history']['train_loss'][-1],
        "mean_squared_error_final_train_loss": mse_results['train_history']['train_loss'][-1],
    })
    
    run.finish()
    print("\nComparison completed. Results logged to wandb.")

if __name__ == "__main__":
    main()