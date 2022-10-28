# importing important packages
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


################################### DATASET PROCESSING ####################################

# Initializing the objects for training and validation
training_data_object = ImageDataGenerator(rescale=1.0/255)
validation_data_object = ImageDataGenerator(rescale=1.0/255)

# processing/augmenting traininig images
train_generator = training_data_object.flow_from_directory(
        'D:/IIT/Academics/5fth Semester/ECN-343/Project/Project ETE/train',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# processing/augmenting testing images
validation_generator = validation_data_object.flow_from_directory(
        'D:/IIT/Academics/5fth Semester/ECN-343/Project/Project ETE/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')


################################### DEFININIG THE NETWORK ####################################

# create sequential model architecture
network = Sequential()

# definition of convolution layers
network.add(Conv2D(32, kernel_size=(3, 3), kernel_initializer='random_normal', activation='relu', input_shape=(48, 48, 1)))
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.25))

network.add(Conv2D(128, kernel_size=(3, 3), kernel_initializer='random_normal', activation='relu'))
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Conv2D(128, kernel_size=(3, 3), kernel_initializer='random_normal', activation='relu'))
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.25))

# definition of hidden and output layers
network.add(Flatten())
network.add(Dense(1024, activation='relu'))
network.add(Dropout(0.5))
network.add(Dense(7, activation='softmax'))

# disabling the use of GPU
cv2.ocl.setUseOpenCL(False)

# Backpropagation parameters
network.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])


################################### TRAINING AND SAVING THE NETWORK ####################################

# Traning the model
network_info = network.fit_generator(
        train_generator, 
        steps_per_epoch=28709 // 64,
        epochs=200,
        validation_data=validation_generator,
        validation_steps=7178 // 64)

# saving the structure in json file
model_json = network.to_json()
with open("structure.json", "w") as json_file:
    json_file.write(model_json)

# saving the weights in .h5 file
network.save_weights('weights.h5')
#############################################33 END OF THE PROGRAM #############################################