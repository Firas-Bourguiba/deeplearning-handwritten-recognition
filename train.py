import tensorflow as tf

def train_and_save():
    # Load the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize the data
    X_train, X_test = X_train / 255.0, X_test / 255.0

    # Build the neural network model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test))

    # Save the model to disk
    model.save('model.h5')

    # Evaluate the model
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(f"Validation loss: {val_loss}, Validation accuracy: {val_acc}")

if __name__ == '__main__':
    train_and_save()
