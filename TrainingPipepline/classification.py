# Importing the necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class Classifier:
    def preprocess_dataset(train_dir, test_dir):
        # Preprocessing the training set
        train_datagen = ImageDataGenerator(rescale=1./255)
        training_set = train_datagen.flow_from_directory(
            train_dir,
            target_size=(512, 512),
            batch_size=32,
            class_mode='binary',
            seed=42
        )
        # Preprocessing the test set
        test_datagen = ImageDataGenerator(rescale=1./255)

        test_set = test_datagen.flow_from_directory(
            test_dir,
            target_size=(512, 512),
            batch_size=32,
            class_mode='binary',
            seed=42
        )
        return training_set, test_set

    def prepareModel():
        # Creating the CNN
        model = Sequential()
        # Convolutional input layer
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)))
        # Convolutional hidden layer
        model.add(Conv2D(32, (3, 3), activation='relu'))
        # Max Pooling layer
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
        # Convolutional hidden layers
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        # Max Pooling layer
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Flattening layer
        model.add(Flatten())
        # Fully connected output layer
        model.add(Dense(1, activation='sigmoid', dtype= tf.float32))
        # Compile the model
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self):
        # Path to the training and test set
        train_dir = 'classification/1_classification/Train'
        test_dir = 'classification/1_classification/Test'
        # Preprocessing the dataset
        training_set, test_set = self.preprocess_dataset(train_dir, test_dir)
        # Preparing the model
        model = self.prepareModel()
        # Training the model
        model = model.fit(training_set, epochs=10, validation_data=test_set)
        # Evaluating the model
        model.evaluate(test_set)
        # Saving the model
        model.save('classification_model.keras')
    
    def is_an_id_card(self, image_path):
        # Load the model classification_model.keras
        model = load_model('classification_model.keras')
        # Load the image
        image = load_img(image_path, target_size=(512, 512))

        image_array = img_to_array(image)
        image_array = image_array / 255.0
        image_array = tf.expand_dims(image_array, axis=0)
        # Predict the image
        prediction = model.predict(image_array)
        # Return True if the image is an ID card means prediction is closer to zero
        # as zero is the id card class
        return (prediction[0][0] < 0.5)
