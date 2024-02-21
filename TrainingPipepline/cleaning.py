import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout
from PIL import Image

class ImageCleaner:
    def __init__(self):
        self.target_size = (720, 720)

    # Define U-Net model
    def unet_model():
        inputs = Input((720, 720, 3))

        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(inputs)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2DTranspose(512,2,strides=(2,2),padding='same')(drop5)
        merge6 = concatenate([drop4,up6], axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv6)

        up7 = Conv2DTranspose(256,2,strides=(2,2),padding='same')(conv6)
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv7)

        up8 = Conv2DTranspose(128,2,strides=(2,2),padding='same')(conv7)
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv8)

        up9 = Conv2DTranspose(64,2,strides=(2,2),padding='same')(conv8)
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv9)

        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same')(conv9)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

        model = Model(inputs, conv10)
        return model

    # Data generator
    def data_generator(self, image_folder, mask_folder, batch_size=8):
        image_files = os.listdir(image_folder)
        while True:
            batch_images = []
            batch_masks = []
            selected_files = np.random.choice(image_files, size=batch_size, replace=False)
            for file in selected_files:
                # base_filename = file.split("__")[0]  # Extract base filename without the suffix
                base_filename = file.replace(".png", "")


                img_path = os.path.join(image_folder, f"{base_filename}")
                mask_path = os.path.join(mask_folder, f"{base_filename}")

                img_path = img_path + ".png"
                mask_path = mask_path + ".png"


                # Load and preprocess image
                img = load_img(img_path, target_size=self.target_size)
                img = img_to_array(img) / 255.0
                batch_images.append(img)

                # Load and preprocess mask
                mask = load_img(mask_path, target_size=self.target_size, color_mode='grayscale')
                mask = img_to_array(mask) / 255.0
                batch_masks.append(mask)

            yield np.array(batch_images), np.array(batch_masks)

    def train(self):
        # Define folder paths
        data_dir = '/cleaning/'
        train_image_folder = data_dir + 'Train/Ids'
        train_mask_folder = data_dir + 'Train/GroundTruth'
        test_image_folder = data_dir + 'Test/Ids'
        test_mask_folder = data_dir + 'Test/GroundTruth'

        # Create U-Net model
        model = self.unet_model()

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        batch_size = 2
        epochs = 10

        train_generator = self.data_generator(train_image_folder, train_mask_folder, batch_size=batch_size)
        validation_generator = self.data_generator(test_image_folder, test_mask_folder, batch_size=batch_size)

        model.fit(train_generator, epochs=epochs, steps_per_epoch=len(os.listdir(train_image_folder)) // batch_size,
                validation_data=validation_generator, validation_steps=len(os.listdir(test_image_folder)) // batch_size)
        # Save the model
        model.save('cleaning_model.keras')

    # Function to load and preprocess a single test image
    def load_test_image(self, image_path):
        img = load_img(image_path, target_size=self.target_size)
        img = img_to_array(img) / 255.0
        return np.expand_dims(img, axis=0)
    
    def clean_image(self, image_path):
        # Load the model
        model = load_model('cleaning_model.keras')

        output_image_path = 'cleaning/' + os.path.splitext(os.path.basename(image_path))[0] + ".png"
        test_image = self.load_test_image(image_path)
        cleaned_image = model.predict(test_image)

        Image.fromarray((np.squeeze(cleaned_image[0]) * 255).astype(np.uint8)).save(output_image_path)
        return output_image_path
