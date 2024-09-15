import csv
import requests
import re
from itertools import chain
import pandas as pd
import time

def parse_csv_file(file_path):
    data = []
    
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        
        for row in reader:
            if row[1] == 'Error':
                continue
            data.append({
                "index": int(row[0]),
                "prediction_list": eval(row[1]),
                "entity_name": row[2],
                "entity_value": row[3]
            })
    
    return data

csv_file_path = './training/train_ocr.csv'
parsed_data = parse_csv_file(csv_file_path)

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

def normalize_unit(unit):
    """Convert unit to its standardized form."""
    unit = unit.lower()
    return unit_variations.get(unit, unit)

def extract_values_and_units(key, value_list):
    """
    Extracts numeric values and units for a given key.
    :param key: The entity key (like 'voltage', 'width')
    :param value_list: A list of string values to parse (like ['175-250.6V'])
    :return: List of tuples (range of numbers, normalized unit)
    """
    extracted_data = []
    
    # Get the valid units for the entity key
    valid_units = entity_unit_map.get(key, set())
    
    # Pattern to capture number ranges and units
    range_pattern = r"([-+]?\d*\.?\d+)\s*-\s*([-+]?\d*\.?\d+)\s*([a-zA-Z]+)"
    single_value_pattern = r"([-+]?\d*\.?\d+)\s*([a-zA-Z]+)"
    
    for value in value_list:
        # Check for ranges first
        range_match = re.search(range_pattern, value)
        if range_match:
            start_value = float(range_match.group(1))
            end_value = float(range_match.group(2))
            unit = normalize_unit(range_match.group(3))
            if unit in valid_units:
                extracted_data.append(((start_value, end_value), unit))
        else:
            # Fallback to single values
            matches = re.findall(single_value_pattern, value)        

            for match in matches:
                number = float(match[0])
                unit = normalize_unit(match[1])

                if unit in valid_units:
                    extracted_data.append((number, unit))
                    return extracted_data
    
    return extracted_data


file = open('predictions.csv', mode='w', newline='')
writer = csv.writer(file)
writer.writerow(['index','entity_name', 'prediction', 'actual_entity_value', 'text'])

start_time = time.time()
print("Start time = ", start_time)

for item in parsed_data:
    index = item['index']
    prediction_list = item['prediction_list']
    entity_name = item['entity_name']
    actual_entity_value = item['entity_value']

    # test_values = list(chain.from_iterable(prediction_list)) # flatten 2d list

    x = extract_values_and_units(entity_name, prediction_list)
    
    out = ""

    if (len(x) == 0):
        out = ""
    else:
        x = x[0]
        # print(type(x[0])) 
        if(isinstance(x[0], float)):
            out = str(x[0]) + " " + x[1]
        else:
            out = '"[' + str(x[0][0]) + ", " + str(x[0][1]) + '] ' + x[1]+'"'
        
    writer.writerow([item['index'], entity_name, out, actual_entity_value, str(prediction_list)])

file.close()

end_time = time.time()
print("End time = ", end_time)
print("Time taken = ", end_time - start_time)
