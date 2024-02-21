import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.optimizers import Adam
from PIL import Image
import cv2

class ImageSegmenter:
    def __init__(self):
        self.target_size = (720, 720)

    # Define U-Net architecture
    def unet(input_shape, num_classes):
        inputs = Input(input_shape)

        # Contracting path
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        # Bottom
        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)

        # Expansive path
        up6 = Conv2D(512, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5))
        merge6 = Concatenate(axis=3)([conv4, up6])
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
        merge7 = Concatenate(axis=3)([conv3, up7])
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
        merge8 = Concatenate(axis=3)([conv2, up8])
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
        merge9 = Concatenate(axis=3)([conv1, up9])
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

        outputs = Conv2D(num_classes, 1, activation='sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=outputs)
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
                mask_path = os.path.join(mask_folder, f"{base_filename}_seg")

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

    # Function to load and preprocess a single test image
    def load_test_image(self, image_path):
        img = load_img(image_path, target_size=self.target_size)
        img = img_to_array(img) / 255.0
        return np.expand_dims(img, axis=0)

    # Function to save predicted masks as images
    def save_predicted_masks(self, model, test_image_folder, output_folder):
        test_image_files = os.listdir(test_image_folder)

        for file in test_image_files:
            base_filename = file
            img_path = os.path.join(test_image_folder, f"{base_filename}")
            output_path = os.path.join(output_folder, f"{base_filename}")

            # Load and preprocess the test image
            test_img = self.load_test_image(img_path, target_size=self.target_size)

            # Generate predictions
            predicted_mask = model.predict(test_img)

            # Save the predicted mask as an image
            predicted_mask = (predicted_mask > 0.5).astype(np.uint8) * 255  # Assuming binary segmentation
            predicted_mask_img = Image.fromarray(predicted_mask[0, :, :, 0], mode='L')
            predicted_mask_img.save(output_path)

    # Function to overlay the predicted mask on the original image and save the result
    def overlay_masks_on_images(self, model, test_image_folder, output_folder):
        test_image_files = os.listdir(test_image_folder)

        for file in test_image_files:
            base_filename = file
            img_path = os.path.join(test_image_folder, f"{base_filename}")
            output_path = os.path.join(output_folder, f"{base_filename}__overlay.png")
            # Load and preprocess the test image
            test_img = self.load_test_image(img_path, target_size=self.target_size)
            # Generate predictions
            predicted_mask = model.predict(test_img)
            # Threshold the predicted mask
            threshold = 0.5
            predicted_mask_binary = (predicted_mask > threshold).astype(np.uint8)
            # Find contours from the binary mask
            contours, _ = cv2.findContours(predicted_mask_binary[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Assuming the largest contour corresponds to the area of interest
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(cnt)
                crop_bbox = (x, y, x+w, y+h)
            else:
                # Default to full image if no contours found
                crop_bbox = (0, 0, predicted_mask_binary.shape[2], predicted_mask_binary.shape[1])
            # Apply the binary mask to the original image
            overlaid_img = test_img[0] * predicted_mask_binary

            # Squeeze dimensions and convert to uint8
            overlaid_img = (np.squeeze(overlaid_img) * 255).astype(np.uint8)

            # Convert overlaid image to PIL image
            overlaid_pil_img = Image.fromarray(overlaid_img)
            origial_image = Image.fromarray((np.squeeze(test_img[0]) * 255).astype(np.uint8))
            # Crop the overlaid image if crop_bbox is provided
            if crop_bbox is not None:
                # overlaid_pil_img = overlaid_pil_img.crop(crop_bbox)
                origial_image = origial_image.crop(crop_bbox)
            origial_image.save(output_path)

    def train_and_save_model(self):
        # Load the data
        data_dir = 'segmentation'
        train_image_folder = data_dir+'/Train/Ids'
        train_mask_folder = data_dir+'/Train/GroundTruth'
        test_image_folder = data_dir+'/Test/Ids'
        test_mask_folder = data_dir+'/Test/GroundTruth'
        batch_size = 2
        epochs = 10
        
        train_generator = self.data_generator(train_image_folder, train_mask_folder, batch_size=batch_size)
        validation_generator = self.data_generator(test_image_folder, test_mask_folder, batch_size=batch_size)
        
        # Define the model
        model = self.unet((720, 720, 3), 1)
        model.compile(optimizer=Adam(), loss='binary_focal_crossentropy', metrics=['accuracy'])
        model.fit(train_generator, epochs=epochs, steps_per_epoch=len(os.listdir(train_image_folder)) // batch_size,
            validation_data=validation_generator, validation_steps=len(os.listdir(test_image_folder)) // batch_size)

        # Save the model
        model.save('segmentation_model.keras')
        return model
    
    def segment_id_card(self, image_path):
        model = load_model('segmentation_model.keras')
        segmentated_id_folder = "segmentation/segmented_id_cards/"
        os.makedirs(segmentated_id_folder, exist_ok=True)
        output_file_path = segmentated_id_folder + os.path.splitext(os.path.basename(image_path))[0] + ".png"
        
        # Load and preprocess the test image
        test_img = self.load_test_image(image_path)
        # Generate predictions
        predicted_mask = model.predict(test_img)
        # Threshold the predicted mask
        threshold = 0.5
        predicted_mask_binary = (predicted_mask > threshold).astype(np.uint8)
        # Find contours from the binary mask
        contours, _ = cv2.findContours(predicted_mask_binary[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Assuming the largest contour corresponds to the area of interest
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            crop_bbox = (x, y, x+w, y+h)
        else:
            # Default to full image if no contours found
            crop_bbox = (0, 0, predicted_mask_binary.shape[2], predicted_mask_binary.shape[1])
        
        origial_image = Image.fromarray((np.squeeze(test_img[0]) * 255).astype(np.uint8))
        # Crop the original image if crop_bbox is provided
        if crop_bbox is not None:
            origial_image = origial_image.crop(crop_bbox)
            origial_image.save(output_file_path)
            return output_file_path
        return None