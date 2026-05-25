import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Setting random seed for reproducibility
tf.random.set_seed(23)

#Loading in MNIST dataset from tensorflow keras
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#Preprocessing (only normalization right now)
#Normalization
x_train_full, x_test = x_train_full / 255.0, x_test / 255.0

#Flattening image (28x28) into 784x1
x_train_full = x_train_full.reshape(x_train_full.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

#Train/val split
train_size = 0.9
val_size = 0.1
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full, test_size=val_size, random_state=23, shuffle=True
)
print(f'Training data shape: {x_train.shape}')
print(f'Validation data shape: {x_val.shape}')
print(f'Testing data shape: {x_test.shape}')

#Model hyperparameters
learning_rate = 0.001
epochs = 20
batch_size = 32

#Optional regularization featues
dropout = True
dropout_rate = 0.3
L2_regularization = True
L1_regularization = False

#L2/L1 Regularization settings
reg_type = "None"
if L1_regularization and L2_regularization:
    regularizer = tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)
    reg_type = "L1_L2"
elif L1_regularization:
    regularizer = tf.keras.regularizers.l1(0.01)
    reg_type = "L1"
elif L2_regularization:
    regularizer = tf.keras.regularizers.l2(0.01)
    reg_type = "L2"
else:
    regularizer = None

#Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

#Model init
layers = [
     tf.keras.layers.Dense(256, activation='relu', input_shape=(784,), kernel_regularizer=regularizer)
]
if dropout:
    layers.append(tf.keras.layers.Dropout(dropout_rate))

layers.append(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizer))
if dropout:
    layers.append(tf.keras.layers.Dropout(dropout_rate))

layers.append(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizer))
if dropout:
    layers.append(tf.keras.layers.Dropout(dropout_rate))

layers.append(tf.keras.layers.Dense(10, activation='softmax'))

model = tf.keras.models.Sequential(layers)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#Model Info
hidden_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)][:-1]
num_hidden_layers = len(hidden_layers)
hidden_units = [layer.units for layer in hidden_layers]
model.summary()

#Progress Bar + history tracking + early stopping
history = model.fit(
    x_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

#Loss Reporting
best_epoch = history.history['val_loss'].index(min(history.history['val_loss']))
print(f'\nBest Epoch: {best_epoch + 1}')
print(f'Training Loss at Best Epoch: {history.history["loss"][best_epoch]:.4f}')
print(f'Validation Loss at Best Epoch: {history.history["val_loss"][best_epoch]:.4f}')
print(f'Final Logged Training Loss: {history.history["loss"][-1]:.4f}')
print(f'Final Logged Validation Loss: {history.history["val_loss"][-1]:.4f}')

#Evaluation of the restored (best) model
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print(f'\nRestored Model Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
print(f'Restored Model Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

#Loss and accuracy plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['loss'], label='Training Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].axvline(best_epoch, color='r', linestyle='--', label=f'Best Epoch: {best_epoch + 1}')
axes[0].set_title('Loss vs Epoch')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['accuracy'], label='Training Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[1].axvline(best_epoch, color='r', linestyle='--', label=f'Best Epoch: {best_epoch + 1}')
axes[1].set_title('Accuracy vs Epoch')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.suptitle(
    f'MNIST MLP Training Curves '
    f'(LR={learning_rate}, Batch={batch_size}, Dropout={dropout_rate if dropout else "None"}, Reg={reg_type})'
)
plt.tight_layout()
plt.show()
