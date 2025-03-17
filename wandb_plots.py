import wandb
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import matplotlib
matplotlib.use("Agg")  

wandb.init(project="fashion_mnist_ffnn", resume="allow")

(train_images, train_labels), (_, _) = fashion_mnist.load_data()

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

sample_images = []
sample_labels = []
for class_id in range(10):
    idx = np.where(train_labels == class_id)[0][0]
    sample_images.append(train_images[idx])
    sample_labels.append(class_names[class_id])

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle("Fashion-MNIST Samples", fontsize=16)
for i, ax in enumerate(axes.flat):
    ax.imshow(sample_images[i], cmap='gray')
    ax.set_title(sample_labels[i])
    ax.axis("off")

# Log plot to wandb sweep
wandb.log({"fashion_mnist_grid": wandb.Image(fig)})

# Close plot
plt.close(fig)

# Finish wandb run
wandb.finish()
