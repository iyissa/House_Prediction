import tensorflow as tf

def load_prep_img(fname, image_shape=224):
	img = tf.io.read_file(fname)
	img = tf.image.decode_img(img)
	img = tf.image.resize(img, size=[image_shape, image_shape])
	img = img/255.
	return img


