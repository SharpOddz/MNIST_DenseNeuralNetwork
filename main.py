import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Setting random seed for reproducibility
tf.random.set_seed(23)

#Loading in MNIST dataset from tensorflow keras
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#Preprocessing
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

#Model hyperparameters
learning_rate = 0.001
epochs = 20
batch_size = 32

#Optional regularization features - Toggle these to True to enable
dropout_rate = 0.2
L2_regularization = True
L2_reg_weight = 0.001
L1_regularization = False
L1_reg_weight = 0.001

#L2/L1 Regularization settings (If neither are set to true then L1/L2 regularization are turned off)
reg_type = "None"
if L1_regularization and L2_regularization:
    regularizer = tf.keras.regularizers.l1_l2(l1=L1_reg_weight, l2=L2_reg_weight)
    reg_type = "L1_L2"
elif L1_regularization:
    regularizer = tf.keras.regularizers.l1(L1_reg_weight)
    reg_type = "L1"
elif L2_regularization:
    regularizer = tf.keras.regularizers.l2(L2_reg_weight)
    reg_type = "L2"
else:
    regularizer = None

#Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)
lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1
)

#Model init
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(512, kernel_regularizer=regularizer),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(dropout_rate),
    
    tf.keras.layers.Dense(256, kernel_regularizer=regularizer),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(dropout_rate),
    
    tf.keras.layers.Dense(128, kernel_regularizer=regularizer),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(dropout_rate),
    
    tf.keras.layers.Dense(64, kernel_regularizer=regularizer),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(dropout_rate),
    
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#Model Training + History
history = model.fit(
    x_train, y_train, 
    epochs=epochs, 
    batch_size=batch_size, 
    validation_data=(x_val, y_val), 
    callbacks=[early_stopping, lr_reduction],
    verbose=1
)

#Evaluation (IMPORTANT: Only test on the test set once the hyperparameters are chosen)
train_metrics = model.evaluate(x_train, y_train, verbose=0)
val_metrics = model.evaluate(x_val, y_val, verbose=0)
#test_metrics = model.evaluate(x_test, y_test, verbose=0)

print(f"Train Loss: {train_metrics[0]:.4f}, Train Accuracy: {train_metrics[1]:.4f}")
print(f"Val Loss:   {val_metrics[0]:.4f}, Val Accuracy:   {val_metrics[1]:.4f}")
#print(f"Test Loss:  {test_metrics[0]:.4f}, Test Accuracy:  {test_metrics[1]:.4f}")

#Plotting
best_epoch = history.history['val_loss'].index(min(history.history['val_loss']))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['loss'], label='Train')
axes[0].plot(history.history['val_loss'], label='Val')
axes[0].set_title('SCCE Loss vs Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('SCCE Loss')
axes[0].legend()

axes[1].plot(history.history['accuracy'], label='Train')
axes[1].plot(history.history['val_accuracy'], label='Val')
axes[1].set_title('Accuracy vs Epochs')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.suptitle(f'MNIST Training Plot')
plt.show()
