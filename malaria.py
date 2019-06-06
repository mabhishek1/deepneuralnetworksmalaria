#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 17:33:15 2019

@author: abhishek
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense



classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))




classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))



classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))



classifier.add(Flatten())


classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



from keras.preprocessing.image import ImageDataGenerator

traindatagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2,
zoom_range = 0.2, horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

trainingset = traindatagen.flow_from_directory('cell_images/train', target_size = (64, 64),
batch_size = 32, class_mode = 'binary')

testset = test_datagen.flow_from_directory('cell_images/test', target_size = (64, 64),
batch_size = 32, class_mode = 'binary')






classifier.fit_generator(trainingset,
                        steps_per_epoch = 5000,
                        epochs = 10,
                        validation_data = testset,
                        validation_steps = 2000
                        )







# serialize model to JSON
model_json = classifier.to_json()
with open("classifier.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("model.h5")
print("Saved model to disk")




from keras.models import model_from_json

# load json and create model
json_file = open('classifier.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))



trainingset.class_indices

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('cell_images/C51AP12thinF_IMG_20150724_153313_cell_106.png', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)

if result[0][0] == 1:
    prediction = 'Uninfected'
else:
    prediction = 'Parasitized'




