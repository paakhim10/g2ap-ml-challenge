def read_file(file_path):
    try:
        with open(file_path, 'r') as f:
            header = f.readline().strip().split(',')
            data = []
            for line in f:
                values = line.strip().split(',')
                if len(values) == len(header):
                    data.append(dict(zip(header, values)))
                else:
                    print(f"Skipping malformed line in {file_path}: {line}")
        print(f"Successfully read {len(data)} rows from {file_path}")
        return data
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return []

def write_file(file_path, data):
    try:
        with open(file_path, 'w') as f:
            header = ','.join(data[0].keys())
            f.write(header + '\n')
            for row in data:
                f.write(','.join(str(value) for value in row.values()) + '\n')
        print(f"Successfully wrote {len(data)} rows to {file_path}")
    except Exception as e:
        print(f"Error writing file {file_path}: {str(e)}")

def get_index_range(data):
    indices = [int(row['index']) for row in data if 'index' in row]
    return min(indices) if indices else 0, max(indices) if indices else 0

# Read all files
pred_files = [
    "./split/output/outputstest1.csv",
    "./split/output/outputstest2.csv",
    "./split/output/outputstest3.csv"
]
pred_data = [read_file(file) for file in pred_files]
pre_output = read_file("./../submissions/output_3.csv")

# Create a dictionary to store updates
updates = {}

# Process each prediction file
for i, pred in enumerate(pred_data, 1):
    start, end = get_index_range(pred)
    print(f"Processing prediction file {i}: index range {start} to {end}")
    for row in pred:
        if 'index' in row:
            index = int(row['index'])
            updates[index] = row
    print(f"Added {len(pred)} rows to updates from prediction file {i}")

# Update pre_output with the new values
updated_count = 0
for row in pre_output:
    if 'index' in row:
        index = int(row['index'])
        if index in updates:
            row.update(updates[index])
            updated_count += 1

print(f"Updated {updated_count} rows in pre_output")

# Write the updated data back to a new file
write_file('./../submissions/updated_output_3.csv', pre_output)

print("Dataset merging complete. Updated output saved as './../submissions/updated_output_3.csv'.")