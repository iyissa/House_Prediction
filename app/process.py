import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

class_names = ['Bathroom' 'Bedroom' 'Dinning' 'Kitchen' 'Livingroom']
loaded_model = tf.keras.models.load_model(("/home/ayomide/Documents/Projects/House_Prediction/Resnet_model.h5"),custom_objects={'KerasLayer':hub.KerasLayer})

def load_and_prep_image(filename, img_shape=224):

  """
  Reads an image from filename, turns it to a tensor and reshapes it 
  to (img_shape, img_shape, colour_channels)
  """
  
  # Read in the image 
  img = tf.io.read_file(filename)
  # Decord the read file into a tensor 
  img = tf.image.decode_image(img)
  #Resize the image 
  img = tf.image.resize(img, size=[img_shape, img_shape])
  # Rescale the image (get all values standardized)
  img = img/255.
  return img 

def pred_and_plot (model, filename, class_names = class_names):
  """
  Imports an image located at filename, makes a prediction with the model specified
  and plot the image with the predicted class at the title
  """

  # Import the target image and preprocess it 
  img = load_and_prep_image(filename)

  # Make a prediction
  pred = model.predict(tf.expand_dims(img, axis=0))

  # Add in logic for multi-class
  if len(pred[0]) > 1:
    pred_class = class_names[tf.argmax(pred[0])]
  else:
    pred_class = class_names[(int(tf.round(pred)))]

  # Get the predicted class
  # pred_class = class_names[(int(tf.round(pred)))]

  #P Plot the image and predicted class 
  plt.imshow(img)
  plt.title(f"Prediction:{pred_class}")
  plt.axis(False);

pred_and_plot(model=loaded_model, filename="bed.jpeg",class_names = class_names)
