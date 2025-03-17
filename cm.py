import numpy as np
import matplotlib.pyplot as plt
import wandb
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.datasets import fashion_mnist
from implementations import FeedForwardNeuralNetwork, preprocess_data
import matplotlib
matplotlib.use("Agg")  

# Initialize wandb
run = wandb.init(project="fashion_mnist_ffnn", name="best_model_evaluation")

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

# Log best parameters to wandb
wandb.config.update(best_params)

# Initialize the model with best hyperparameters
best_model = FeedForwardNeuralNetwork(
    input_size=input_size,
    output_size=output_size,
    hidden_layers=best_params["hidden_layers"],
    activation=best_params["activation"],
    loss="cross_entropy",
    optimizer=best_params["optimizer"],
    learning_rate=best_params["learning_rate"],
    weight_init=best_params["weight_init"],
    weight_decay=best_params["weight_decay"]
)

# Train the model
print("Training the best model...")
history = best_model.fit(
    X_train_processed, y_train_onehot,
    epochs=best_params["epochs"],
    batch_size=best_params["batch_size"]
)

# Evaluate on test set
test_loss, test_accuracy = best_model.evaluate(X_test_processed, y_test_onehot)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Log test metrics to wandb
wandb.log({
    "test_accuracy": test_accuracy,
    "test_loss": test_loss
})

# Get predictions on test set
y_pred_proba = best_model.predict(X_test_processed)
y_pred = np.argmax(y_pred_proba, axis=1)
y_test_classes = np.argmax(y_test_onehot, axis=1)

# Class names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Compute confusion matrix
cm = confusion_matrix(y_test_classes, y_pred)

# Creative visualization of confusion matrix
plt.figure(figsize=(12, 10))

# Create a custom colormap from blue to purple to yellow
colors = [(0.1, 0.1, 0.6), (0.5, 0, 0.5), (0.9, 0.9, 0.1)]
custom_cmap = sns.blend_palette(colors, as_cmap=True)

# Plot the confusion matrix with seaborn for better visual appeal
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d',
    cmap=custom_cmap,
    linewidths=1,
    square=True,
    cbar_kws={"shrink": 0.8},
    xticklabels=class_names,
    yticklabels=class_names
)

# Add an elegant border to the heatmap
plt.gca().patch.set_edgecolor('black')
plt.gca().patch.set_linewidth(1.5)

# Modify appearance
plt.title('Confusion Matrix for Fashion-MNIST Classification', fontsize=16, pad=15)
plt.ylabel('True Label', fontsize=12, labelpad=10)
plt.xlabel('Predicted Label', fontsize=12, labelpad=10)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)

# Add performance summary as text annotation
classification_rep = classification_report(y_test_classes, y_pred, target_names=class_names, output_dict=True)
plt.figtext(0.5, -0.05, f"Overall Test Accuracy: {test_accuracy:.4f}", 
            ha="center", fontsize=12, bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.8})

# Save and log the confusion matrix
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
wandb.log({"confusion_matrix": wandb.Image('confusion_matrix.png')})

# Generate some sample misclassifications
misclassified_indices = np.where(y_pred != y_test_classes)[0]
sample_size = min(10, len(misclassified_indices))
sample_indices = np.random.choice(misclassified_indices, sample_size, replace=False)

# Reshape test images back to original shape for visualization
X_test_images = X_test.reshape(X_test.shape[0], 28, 28)

# Create a figure to show misclassified examples
if sample_size > 0:
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Sample Misclassifications", fontsize=16)
    
    for i, idx in enumerate(sample_indices):
        ax = axes.flat[i] if sample_size > 1 else axes
        ax.imshow(X_test_images[idx], cmap='viridis')
        ax.set_title(f"True: {class_names[y_test_classes[idx]]}\nPred: {class_names[y_pred[idx]]}")
        ax.axis('off')
    
    # If we have fewer than 10 examples, hide the extra subplots
    if sample_size < 10:
        for i in range(sample_size, 10):
            axes.flat[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('misclassifications.png', dpi=300, bbox_inches='tight')
    wandb.log({"misclassified_examples": wandb.Image('misclassifications.png')})

# Per-class metrics visualization
# Fixed: The classification_report uses class names as keys, not indices
precision = []
recall = []
f1_score = []

# Get metrics for each class from the classification report
for i, class_name in enumerate(class_names):
    precision.append(classification_rep[class_name]['precision'])
    recall.append(classification_rep[class_name]['recall'])
    f1_score.append(classification_rep[class_name]['f1-score'])

# Create a per-class performance chart
plt.figure(figsize=(14, 8))
x = np.arange(len(class_names))
width = 0.25

plt.bar(x - width, precision, width, label='Precision', color='#5DA5DA', edgecolor='black', linewidth=1)
plt.bar(x, recall, width, label='Recall', color='#FAA43A', edgecolor='black', linewidth=1)
plt.bar(x + width, f1_score, width, label='F1-Score', color='#60BD68', edgecolor='black', linewidth=1)

plt.xlabel('Fashion Item Class', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Performance Metrics by Class', fontsize=16)
plt.xticks(x, class_names, rotation=45, ha='right')
plt.ylim(0, 1.05)
plt.legend()

# Add a grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('class_performance.png', dpi=300, bbox_inches='tight')
wandb.log({"class_performance": wandb.Image('class_performance.png')})

# Finish wandb run
wandb.finish()

print("Evaluation completed. Results logged to wandb.")