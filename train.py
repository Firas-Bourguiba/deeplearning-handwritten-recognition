import tensorflow as tf #importer tensorflow pour la creation du model et l'entrainement

def train_and_save(): # def d'une fonction qui encapsule le processus d'entrainement et de sauvegarde du model
    
    mnist = tf.keras.datasets.mnist  #importer le dataset mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()  # découpage des doonées MNIST en données d'entrainement ('X_train', 'y_train') et données de test ('X_test', 'y_test')

    
    X_train, X_test = X_train / 255.0, X_test / 255.0 #  # Normalisation des données en divisant par 255.0 pour qu'elles soient comprises entre 0 et 1 pour accélérer l'entrainement

    # Build the neural network model
    model = tf.keras.models.Sequential(                    # Création d'un modèle séquentiel pour creer des modeles couche par couche pour la plupart des problèmes
        tf.keras.layers.Flatten(input_shape=(28, 28)),      # Aplatir les données d'entrée en un vecteur 1D ou transfromation de l'image 28x28 (2d) en un vecteur 1D de 784
        tf.keras.layers.Dense(128, activation='relu'),      # Ajout d'une couche dense'(entierement connécté) de 128 neurones avec une fonction d'activation 'relu'
        tf.keras.layers.Dense(128, activation='relu'),      # Ajout d'une autre couche dense de 128 neurones avec une fonction d'activation 'relu' pour introduire de la non-linéarité
        tf.keras.layers.Dense(10, activation='softmax'))     # Ajout d'une couche de sortie de 10 neurones un pour chaque classe de chiffres (0-9) avec une fonction d'activation 'softmax' pour la classification multiclasse

    # Compilation du model en spécifiant l'optimiseur adam, la fonction de perte 'sparse_categorical_crossentropy' et la métrique 'accuracy' (précision)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # entraiement du model sur les données d'entrainement 'X_train', 'y_train' avec 3 époques et validation sur les données de test 'X_test', 'y_test' pour chaque époque
    model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test))

    # Sauvegarde du model en format hdf5
    model.save('model.h5')

    # Evaluation du model sur les données de test 'X_test', 'y_test' et affichage de la perte et de la précision
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(f"Validation loss: {val_loss}, Validation accuracy: {val_acc}") # affichage de la perte et de la précision

if __name__ == '__main__': # si le script est exécuté directement, appeler la fonction 'train_and_save'
    train_and_save()
