from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer
import csv
from pathlib import Path
import time

# Model path
model_path = "openbmb/MiniCPM-Llama3-V-2_5"

# User and assistant names
U_NAME = "User"
A_NAME = "Assistant"


def load_model_and_tokenizer():
    print(f"Loading model and tokenizer from {model_path}")
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16).to(device="cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()
model.eval()
print("Model and tokenizer loaded successfully!")


unit_variations = {
    'cm': 'centimetre', 'centimeter': 'centimetre',
    'mm': 'millimetre', 'meter': 'metre', 'm': 'metre',
    'kg': 'kilogram', 'g': 'gram', 'mg': 'milligram',
    'lbs': 'pound', 'lb': 'pound', 'oz': 'ounce',
    'kv': 'kilovolt', 'mv': 'millivolt', 'v': 'volt',
    'kw': 'kilowatt', 'w': 'watt',
}

entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon',
                    'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
}



# Interaction loop
def main(image_path, entity_name):
    imagefile = None
    if image_path:
        try:
            imagefile = Image.open(image_path).convert('RGB')
        except Exception as e:
            pass

    
    allowed_units = entity_unit_map[entity_name]

    
    print(image_path, entity_name, allowed_units, imagefile)
    
    user_text = f'''
        Analyze the image provided and detect the specified metric related to the device in the image.
        The metric you need to detect is called {entity_name}.
        Return only the value and the unit in one of the allowed units: {allowed_units}. 
        Do not include any explanations, comments, or additional text.
        The response should follow this exact format: "<value> <unit>".
        If there are multiple units in the image, return the most appropriate one based on the allowed units list.
        IMPORTANT: Only return the value and unit, nothing else.
    '''


    # Prepare input for the model
    msgs = [{"role": "user", "content": user_text}]
    
    # Process and generate response with the model
    res = model.chat(
        image=imagefile,
        msgs=msgs,
        context=None,
        tokenizer=tokenizer,
        sampling=True,
        top_p=0.8,                  
        top_k=100,                   
        repetition_penalty=1.2,     
        temperature=0.8,             
        stream=False,
        # max_new_tokens=10
    )

    print(res)
    return res
    

def parse_csv_file(file_path):
    data = []
    
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        
        for row in reader:
            filename = Path(row[1]).name
            data.append({
                "index": int(row[0]),
                "image_name": filename,
                "image_id": row[2],
                "entity_value": row[3]
            })
    
    return data


csv_file_path = './dataset/test1.csv'
parsed_data = parse_csv_file(csv_file_path)


if __name__ == "__main__":

    file = open('predictions1.csv', mode='w', newline='')
    writer = csv.writer(file)
    writer.writerow(['index','prediction'])
    
    start_time = time.time()

    for item in parsed_data:
        index = item['index']
        image_name = item['image_name']
        entity_value = item['entity_value']

        result = main("./images/" + image_name, entity_value)

        writer.writerow([index, result])
    
    file.close()

    print("Time taken:", time.time() - start_time)