from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)
model = tf.keras.models.load_model('model.h5')

@app.route('/', methods=['GET'])
def index():
    # Render the upload HTML form
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict_digit():
    file = request.files.get('image')
    if not file:
        return render_template("index.html", prediction=None)

    # Read the image through OpenCV, convert to grayscale, and resize
    image = np.array(cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE))
    image = cv2.resize(image, (28, 28))
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
