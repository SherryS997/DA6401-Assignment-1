import wandb
import optuna
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from implementations import FeedForwardNeuralNetwork, preprocess_data

# Load Fashion MNIST dataset globally to avoid reloading for each trial
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# Split training data into training and validation sets (90% train, 10% validation)
num_train_samples = X_train_full.shape[0]
val_size = int(0.1 * num_train_samples)
X_val, y_val = X_train_full[:val_size], y_train_full[:val_size]
X_train, y_train = X_train_full[val_size:], y_train_full[val_size:]

# Preprocess data
X_train_processed, y_train_onehot, X_val_processed, y_val_onehot = preprocess_data(X_train, y_train, X_val, y_val)
X_test_processed, y_test_onehot, _, _ = preprocess_data(X_test, y_test, X_test, y_test)

input_size = X_train_processed.shape[1]
output_size = 10  # 10 classes for Fashion MNIST

def objective(trial):
    # Initialize wandb for this trial
    run = wandb.init(project="fashion_mnist_ffnn_optuna", reinit=True)
    
    # Sample hyperparameters using Optuna
    epochs = trial.suggest_categorical("epochs", [5, 10])
    
    # Create hidden layers configuration
    hidden_layers_type = trial.suggest_categorical("hidden_layers_type", ["small", "medium", "large"])
    if hidden_layers_type == "small":
        hidden_layers = [32] * 3
    elif hidden_layers_type == "medium":
        hidden_layers = [64] * 4
    else:
        hidden_layers = [128] * 5
    
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    optimizer = trial.suggest_categorical("optimizer", ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    weight_init = trial.suggest_categorical("weight_init", ["random", "xavier"])
    activation = trial.suggest_categorical("activation", ["sigmoid", "tanh", "relu"])
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 0.5, log=True)
    
    # Log hyperparameters to wandb
    config = {
        "epochs": epochs,
        "hidden_layers": hidden_layers,
        "learning_rate": learning_rate,
        "optimizer": optimizer,
        "batch_size": batch_size,
        "weight_init": weight_init,
        "activation": activation,
        "weight_decay": weight_decay,
        "trial_number": trial.number
    }
    wandb.config.update(config)
    
    # Initialize and train the FeedForwardNeuralNetwork
    model = FeedForwardNeuralNetwork(
        input_size=input_size,
        output_size=output_size,
        hidden_layers=hidden_layers,
        activation=activation,
        loss="cross_entropy",
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_init=weight_init,
        weight_decay=weight_decay
    )

    # Train the model
    history = model.fit(
        X_train_processed, y_train_onehot,
        epochs=epochs, batch_size=batch_size,
        X_val=X_val_processed, y_val=y_val_onehot
    )
    
    # Get final validation accuracy for optimization
    val_accuracy = history["val_accuracy"][-1] if history["val_accuracy"] else 0
    
    # Report intermediate values for pruning
    for epoch in range(len(history["val_accuracy"])):
        trial.report(history["val_accuracy"][epoch], epoch)
        
        # Early stopping if the trial isn't promising
        if trial.should_prune():
            wandb.log({"pruned": True})
            run.finish()
            raise optuna.exceptions.TrialPruned()
    
    # Log final metrics to wandb
    wandb.log({
        "train_loss": history["train_loss"][-1],
        "val_loss": history["val_loss"][-1] if history["val_loss"] else np.nan,
        "train_accuracy": history["train_accuracy"][-1],
        "val_accuracy": val_accuracy,
        "epochs_completed": epochs
    })
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test_processed, y_test_onehot)
    wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy})
    
    run.finish()
    return val_accuracy

def main():
    # Create an Optuna study for maximizing validation accuracy
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
        study_name="fashion_mnist_optimization"
    )
    
    # Run optimization with 100 trials
    study.optimize(objective, n_trials=100)
    
    # Print optimization results
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value (validation accuracy): {best_trial.value:.4f}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Optionally, save study results
    optuna.visualization.plot_param_importances(study)
    
    # Run a final trial with the best parameters and log to wandb
    run = wandb.init(project="fashion_mnist_ffnn_optuna", name="best_model")
    wandb.config.update(best_trial.params)
    
    # Train with best hyperparameters
    hidden_layers_type = best_trial.params["hidden_layers_type"]
    if hidden_layers_type == "small":
        hidden_layers = [32] * 3
    elif hidden_layers_type == "medium":
        hidden_layers = [64] * 4
    else:
        hidden_layers = [128] * 5
        
    best_model = FeedForwardNeuralNetwork(
        input_size=input_size,
        output_size=output_size,
        hidden_layers=hidden_layers,
        activation=best_trial.params["activation"],
        loss="cross_entropy",
        optimizer=best_trial.params["optimizer"],
        learning_rate=best_trial.params["learning_rate"],
        weight_init=best_trial.params["weight_init"],
        weight_decay=best_trial.params["weight_decay"]
    )
    
    history = best_model.fit(
        X_train_processed, y_train_onehot,
        epochs=best_trial.params["epochs"],
        batch_size=best_trial.params["batch_size"],
        X_val=X_val_processed, y_val=y_val_onehot
    )
    
    # Log final test metrics
    test_loss, test_accuracy = best_model.evaluate(X_test_processed, y_test_onehot)
    wandb.log({
        "final_test_loss": test_loss,
        "final_test_accuracy": test_accuracy
    })
    run.finish()

if __name__ == "__main__":
    main()