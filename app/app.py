import os
from secrets import token_hex

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request

from load import init
from utils import load_prep_img

class_names = []

model = init()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 4
app.config["UPLOAD_DIRECTORY"] = "uploads"
app.config["DEBUG"] = os.environ.get("DEBUG")
app.config["ENV"] = os.environ.get("ENV")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY")


@app.route("/index", methods=["GET"])
def index():
	return render_template("index.html")

@app.route("/predict", methods=["POST"])
def prediction():
	img_data = request.files.get("image")
	img_data_fname = os.path.join(app.config["UPLOADS_DIRECTORY"], str(token_hex(10)) + os.path.splitext(img_data.filename)[1],)
	img_data.save(img_data_fname)	
	loaded_img = load_prep_img(img_data_fname)

	pred = model.predict(tf.expand_dims(loaded_img, axis=0))

	if len(pred[0]) > 1:
		pred_class = class_names[tf.argmax(pred[0])]
	else:
		pred_class = class_names[(int(tf.round(pred)))]
	return {"message": pred_class}

