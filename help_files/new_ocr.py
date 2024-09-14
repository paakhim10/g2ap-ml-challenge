from paddleocr import PaddleOCR
from PIL import Image
from urllib.request import Request, urlopen
from io import BytesIO
from pdf2image import convert_from_bytes
import os
import time
import io
import csv

# Caching the OCR model
def load_ocr_model():
    model = PaddleOCR(use_angle_cls=True, lang='en',use_gpu=0)
    return model

# Merge OCR results
def merge_data(values):
    data = []
    for idx in range(len(values)):
        data.append([values[idx][1][0]])
    return data

# OCR processing function
def invoke_ocr(doc, content_type):
    start_time = time.time()

    model = load_ocr_model()

    bytes_img = io.BytesIO()

    format_img = "JPEG"
    if content_type == "image/png":
        format_img = "PNG"

    doc.save(bytes_img, format=format_img)
    bytes_data = bytes_img.getvalue()
    bytes_img.close()

    result = model.ocr(bytes_data, cls=True)

    values = []
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            values.append(line)

    values = merge_data(values)

    end_time = time.time()
    processing_time = end_time - start_time

    return values, processing_time

# Function to process image from URL
def process_image_from_url(image_url):
    headers = {"User-Agent": "Mozilla/5.0"}  # to avoid 403 errors
    req = Request(image_url, headers=headers)

    with urlopen(req) as response:
        content_type = response.info().get_content_type()

        if content_type in ["image/jpeg", "image/jpg", "image/png"]:
            doc = Image.open(BytesIO(response.read()))
        elif content_type == "application/octet-stream":
            pdf_bytes = response.read()
            pages = convert_from_bytes(pdf_bytes, 300)
            doc = pages[0]
        else:
            raise ValueError("Invalid file type. Only JPG/PNG images and PDFs are allowed.")

        # Invoke the OCR on the document
        result, processing_time = invoke_ocr(doc, content_type)

        return result

# Example usage:
image_url = 'https://m.media-amazon.com/images/I/61I9XdN6OFL.jpg'
ocr_result = process_image_from_url(image_url)
print("OCR Result:", ocr_result)

def process_csv_file(csv_file_path):
    total_start_time = time.time()
    # Open the CSV file
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        
        # Read the header (if present)
        headers = next(reader, None)  # Skip the header row
        
        # Process the first 100 URLs in the first column
        for idx, row in enumerate(reader):
            if idx >= 80:  # Limit to the first 100 entries
                break
            image_url = row[0]  # Assuming the first column contains the URL
            
            try:
                # Call the OCR function for the given image URL
                ocr_result = process_image_from_url(image_url)
                print(f"OCR Result for {image_url}: {ocr_result}")
            except Exception as e:
                print(f"Error processing {image_url}: {e}")
    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time
    print(f"Total processing time: {total_processing_time:.2f} seconds")


# Example usage:
csv_file_path = 'train.csv'  # Path to your CSV file containing image URLs

process_csv_file(csv_file_path)