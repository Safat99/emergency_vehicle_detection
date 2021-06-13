#ei code ta ekdom vgg16 er 1000 ta class er train korano model er upore

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model = VGG16()
model.summary()

import os 
for file in os.listdir('obj'):
    print(file)
    full_path = 'obj/' + file

    image = load_img(full_path, target_size(224,224))
    image = img_to_array(image)
    image = image.reshape((1,image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input
    y_pred = model.predict(image)
    label = decode_predictions(y_pred)#convert the probabilities to class label
    label = decode_predictions(y_pred, top = 1)
    print(label)
    print() 
  