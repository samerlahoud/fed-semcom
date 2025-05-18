import tensorflow as tf
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import tensorflow.image as tf_image

# Load CIFAR-10
(x_train_full, y_train_full), (x_test, _) = cifar10.load_data()
x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Define non-IID clients
client_classes = {
    0: [0, 1],
    1: [2, 3],
    2: [4, 5],
}

def get_client_data(classes):
    indices = np.isin(y_train_full.flatten(), classes)
    return x_train_full[indices], y_train_full[indices]

client_data = {i: get_client_data(classes) for i, classes in client_classes.items()}

# Semantic encoder-decoder
def create_semantic_model():
    inputs = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(8, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(4, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(8, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(16, 3, strides=2, activation="relu", padding="same")(x)
    outputs = layers.Conv2D(3, 3, activation="sigmoid", padding="same")(x)
    return models.Model(inputs, outputs)

# FedLol weighting
def compute_fedlol_weights(losses):
    total_loss = np.sum(losses)
    return [(total_loss - lk) / ((len(losses) - 1) * total_loss) for lk in losses]

# Federated training parameters
NUM_ROUNDS = 5
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

global_model = create_semantic_model()
loss_fn = tf.keras.losses.MeanSquaredError()
global_weights = global_model.get_weights()
test_losses = []

print("\n=== Starting Federated Training with FedLol ===")
for rnd in range(NUM_ROUNDS):
    print(f"\n--- Global Round {rnd + 1} ---")
    local_weights = []
    local_losses = []

    for client_id, (x_client, _) in client_data.items():
        x_train, x_val = train_test_split(x_client, test_size=0.1, random_state=rnd)
        local_model = create_semantic_model()
        local_model.set_weights(global_weights)
        local_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
        local_model.compile(optimizer=local_optimizer, loss=loss_fn)

        local_model.fit(x_train, x_train, epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        val_loss = local_model.evaluate(x_val, x_val, verbose=0)
        print(f"Client {client_id} Validation Loss: {val_loss:.4f}")

        local_weights.append(local_model.get_weights())
        local_losses.append(val_loss)

    fedlol_weights = compute_fedlol_weights(local_losses)
    print(f"FedLol Weights: {[round(w, 4) for w in fedlol_weights]}")

    new_weights = []
    for weights_list in zip(*local_weights):
        aggregated = np.sum([w * fedlol_weights[i] for i, w in enumerate(weights_list)], axis=0)
        new_weights.append(aggregated)

    global_weights = new_weights
    global_model.set_weights(global_weights)

    # ✅ Compile before evaluation
    global_model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss=loss_fn)
    test_loss = global_model.evaluate(x_test, x_test, verbose=0)
    test_losses.append(test_loss)
    print(f"Global Model Test Loss: {test_loss:.4f}")

# Plot test loss over rounds
plt.figure(figsize=(6, 4))
plt.plot(range(1, NUM_ROUNDS + 1), test_losses, marker='o')
plt.title("Test Loss over Rounds (FedLol)")
plt.xlabel("Global Round")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("test_loss_fedlol.png")

# Reconstruct test images
reconstructed = global_model.predict(x_test[:100])
reconstructed = np.clip(reconstructed, 0.0, 1.0)

# Compute final PSNR and SSIM
psnr = tf_image.psnr(x_test[:100], reconstructed, max_val=1.0).numpy().mean()
ssim = tf_image.ssim(x_test[:100], reconstructed, max_val=1.0).numpy().mean()

print(f"\n=== Final Evaluation ===")
print(f"PSNR: {psnr:.2f} dB")
print(f"MS-SSIM: {ssim:.4f}")

# Visualize reconstructed images
plt.figure(figsize=(12, 4))
for i in range(8):
    plt.subplot(2, 8, i + 1)
    plt.imshow(x_test[i])
    plt.axis('off')
    if i == 0:
        plt.title("Original")

    plt.subplot(2, 8, i + 9)
    plt.imshow(reconstructed[i])
    plt.axis('off')
    if i == 0:
        plt.title("Reconstructed")

plt.suptitle("Original vs Reconstructed Images (FedLol Global Model)")
plt.tight_layout()
plt.show()
plt.savefig("reconstructed_images_fedlol.png")