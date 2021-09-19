import io
import os
import sys

from PIL import Image
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

from modeling.inference import MNISTInference

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.secret_key = "super secret key"
model = MNISTInference('/home/lionel/Desktop/MLE/mlops/weights/mnist_model.pt')


@app.route("/predict", methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            print('found file')

        try:
            # Read image contents
            contents = file.read()
            pil_image = Image.open(io.BytesIO(contents))
            pil_image = pil_image.convert('L')

            # Resize image to expected input shape
            pil_image = pil_image.resize((28, 28))

            result = model.predict(pil_image)
            return {'label': str(result)}
        except:
            e = sys.exc_info()[1]
            return {'error': str(e)}

if __name__ == "__main__":
    app.run(debug=False, threaded=False, host='0.0.0.0', port=5000)
    # curl -X POST -F 'file=@/home/lionel/Desktop/MLE/mlops/data/mnist/test/4/67.png' http://192.168.2.12:5000/predict

