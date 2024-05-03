from flask import Flask, request, render_template  #importatiion de flask pour la création d'une application web
import tensorflow as tf  #importer tensorflow pour la creation du model et l'entrainement
import numpy as np   #importation de numpy pour le calcul scientifique avec un support pour les tableaux et les matrices
import cv2    #importation de cv2 pour la manipulation d'images

app = Flask(__name__)
model = tf.keras.models.load_model('model.h5') # Chargement du model sauvegardé hdf5

@app.route('/', methods=['GET']) #route pour la page d'accueil
def index():
    # Render the upload HTML form
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict_digit():                       #Fonction pour prédire le chiffre d'une image envoyée. Si aucune image n'est envoyée, elle affiche de nouveau le formulaire sans prédiction.
    file = request.files.get('image')
    if not file:
        return render_template("index.html", prediction=None)

    # Read the image through OpenCV, convert to grayscale, and resize
    image = np.array(cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)) #Lecture de l'image via OpenCV, conversion en niveaux de gris et décodage de l'image à partir des données brutes envoyées.
    image = cv2.resize(image, (28, 28)) # redimensionnement de l'image en 28x28
    image = np.invert(image)  # Invert colors
    image = image.reshape(1, 28, 28, 1)  # Reshape for the model

    # Normalize the image
    image = image / 255.0

    # Predict the digit
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)

    # Render the form again, but with the prediction
    return render_template("index.html", prediction=int(predicted_digit))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=1)
