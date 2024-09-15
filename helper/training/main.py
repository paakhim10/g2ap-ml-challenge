import pytesseract
import csv
import requests
from PIL import Image
from io import BytesIO
from itertools import chain

def parse_csv_file(file_path):
    data = []
    
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        
        for row in reader:
            data.append({
                "url": row[0],
                "item_id": row[1],
                "attribute": row[2]
            })
    
    return data

csv_file_path = './resource/dataset/train.csv'
parsed_data = parse_csv_file(csv_file_path)

base_url = "http://127.0.0.1:8000/api/v1/sparrow-ocr/inference"


counter = 0

# Open the CSV file to append predictions
with open('training_ocr_out.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['index', 'prediction'])

    for item in parsed_data:
        counter += 1

        if counter >= (10_000):
            break

        url = item['url']

        try:
            # Fetch the image from the URL
            # response = requests.get(url)
            # response.raise_for_status()
            # image = Image.open(BytesIO(response.content))

            payload = {
                "image_url": url,
            }
            
            response = requests.request("POST", base_url, data=payload)
            test_values = list(chain.from_iterable(eval(response.text)))
        
            # print(test_values)

            # Write the result into predictions.csv
            writer.writerow([counter-1, str(test_values)])

        except Exception as e:
            print(f"Error processing image {counter-1} from URL {url}: {e}")
            writer.writerow([counter-1, "Error"])