from tensorflow import keras
import numpy as np

model = keras.models.load_model("flowers17_model.h5")

image_path = "flower1.jpg"
image = keras.preprocessing.image.load_img(image_path, target_size=(64, 64))
image_array = keras.preprocessing.image.img_to_array(image)
image_array = np.expand_dims(image_array, axis=0)
image_array = image_array / 255.0

predictions = model.predict(image_array)
predicted_class = np.argmax(predictions)

print("Predicted class for the image:", predicted_class)