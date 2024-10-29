import re
import pandas as pd

dataframes = {}

for i in range(1, 4):
    data1 = pd.read_csv(f"./beginning/predictions{i}3.csv")
    data2 = pd.read_csv(f"./data{i}.csv")
    
    df = pd.merge(data1, data2, on=['index'])
    
    df = df[['index', 'prediction', 'entity_name']]
    
    dataframes[i] = df

    # Unit variations and entity_unit_map as defined
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

    # Function to normalize units
    def normalize_unit(unit):
        """Convert unit to its standardized form."""
        unit = unit.lower()
        return unit_variations.get(unit, unit)  # Use unit_variations to map to standard form

    # Function to extract numeric values and their units
    def extract_value_with_unit(entity, text):
        possible_units = entity_unit_map.get(entity, set())
        
        all_units = set()
        for unit in possible_units:
            all_units.add(unit.lower())
            for variation, standard_unit in unit_variations.items():
                if standard_unit == unit:
                    all_units.add(variation.lower())
        
        # Regex pattern to find numeric values followed by a unit
        pattern = r'(\d+(?:\.\d+)?)\s*(' + '|'.join(re.escape(unit) for unit in all_units) + r')'
        
        match = re.search(pattern, text.lower())
        
        if match:
            value = match.group(1)
            unit = normalize_unit(match.group(2))
            return f"{value} {unit}"
        else:
            return ""  # No match found, returning an empty string

    # Function to handle ranges and normalize extracted values
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
        
        # Patterns to capture ranges and single values
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

    # Apply the extraction function to each row
    df['prediction'] = df.apply(lambda row: extract_value_with_unit(row['entity_name'], str(row['prediction'])), axis=1)

    # print(df['prediction'][1])

    # Filter DataFrame to only keep relevant columns and write to CSV
    dataframes[i] = df[['index', 'prediction']]
    dataframes[i].to_csv(f"./out/outputstest{i}.csv", index=False)