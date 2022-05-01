import tensorflow_hub as hub
import keras
import tensorflow as tf

def init():
	model = tf.keras.models.load_model(
			("./model/resnet_model.h5"),
			custom_objects={"KerasLayer":hub.KerasLayer})
	return model
