import os
import json

from TrainingPipepline.classification import Classifier
from TrainingPipepline.segmentation import ImageSegmenter
from TrainingPipepline.descewing import Deskewer
from TrainingPipepline.cleaning import ImageCleaner
from TrainingPipepline.ocr import OCR

def perform_ocr(image_path):
    text_extractor = OCR()
    return text_extractor.extract_text(image_path)

def perform_cleaning(image_path):
    cleaner = ImageCleaner()
    return cleaner.clean_image(image_path)

def perform_deskewing(image_path):
    deskewer = Deskewer()
    return deskewer.deskew_image(image_path)

def perform_segmentation(image_path):
    segmenter = ImageSegmenter()
    return segmenter.segment_id_card(image_path)

def main():
    test_image_directory = "TestingPipeline/test_image_dir/"
    # Perform the pipeline on each image in the test_image_directory
    extracted_data = {}
    for filename in os.listdir(test_image_directory):
        image_path = test_image_directory + filename
        # Perform the segmentation to extract the id card from the image
        segmented_image_path = perform_segmentation(image_path)
        if segmented_image_path is not None:
            # Perform the deskewing to straighten the id card
            deskewed_image_path = perform_deskewing(segmented_image_path)
            if deskewed_image_path is not None:
                # Perform the cleaning to remove any noise from the id card
                cleaned_image_path = perform_cleaning(deskewed_image_path)
                # Perform the OCR to extract the text from the id card
                extracted_data[filename] = perform_ocr(cleaned_image_path).split('\n')
            else:
                print("Deskewing failed")
        else:
            print("Segmentation failed")
    
    output_path = "ocr/ocr_output.json"
    with open(output_path, 'w') as json_file:
        json.dump(extracted_data, json_file, indent=4)
    

if __name__ == "__main__":
    main()