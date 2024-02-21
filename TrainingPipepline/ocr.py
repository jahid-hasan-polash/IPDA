import os
import pytesseract
from PIL import Image
import re

class OCR:
    def ocr(self, image_path):
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    
    def look_for_name(self, lines):
        dob_pattern = re.compile(r'\b(?:0[1-9]|[12][0-9]|3[01])\.(?:0[1-9]|1[012])\.(?:19|20)\d\d\b')

        for index, line in enumerate(lines):
            match = re.search(dob_pattern, line)    
            if match:
                return " ".join(lines[:index]), index
        return None, None

    def extract_text(self, image_path):
        extracted_text = self.ocr(image_path)
        print(extracted_text)

        # output_folder = "ocr"
        # id_info = {}
        # lines = extracted_text.split('\n')
        # if self.is_front_page_data(extracted_text, os.path.basename(image_path)):
        #     id_info["Id No"] = lines[0].split(' ')[-1]
        #     name, next_item_index = self.look_for_name(lines[1:])
        #     if name is not None:
        #         id_info["Name"] = name
        #         id_info["Date of birth"] = lines[next_item_index].split(' ')[0]
        #         id_info["Nationality"] = lines[next_item_index].split(' ')[-1]
        #         id_info["Place of birth"] = lines[next_item_index + 1]
        #         id_info["Date of issue"] = lines[next_item_index + 2].split(' ')[0]
        # else:
        #     id_info["additional_info"] = extracted_text

        # output_path = os.path.join(output_folder, os.path.splitext(os.path.basename(image_path))[0] + ".txt")
        # with open(output_path, "w") as text_file:
        #     for field_name, value in id_info.items():
        #         text_file.write(f"{field_name}: {value}\n")
        
        # return output_path
        return extracted_text

    def is_front_page_data(self, text, file_name):
        # ideally if name is found, it is the front page
        # But as our OCR extraction is not perfect, we are relying on the file name
        # return re.search(r'Name', text)
        return "front" in file_name.lower()