import json

def print_sample_json(json_path, sample_size=10):
    try:
        # Open and load the JSON file
        with open(json_path, 'r') as file:
            data = json.load(file)
        
        # Check if the data is a list or a dictionary
        if isinstance(data, list):
            # If it's a list, take a sample
            sample_data = data[:sample_size]
        elif isinstance(data, dict):
            # If it's a dictionary, take a sample of its items
            sample_data = dict(list(data.items())[:sample_size])
        else:
            print("Data format is not recognized.")
            return

        # Convert the sample data to a pretty-printed JSON string
        pretty_data = json.dumps(sample_data, indent=4)
        print(pretty_data)
    
    except json.JSONDecodeError as e:
        print(f"Failed to load JSON: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

import json

def sort_json_file(filepath):
    # Read the JSON data from the file
    with open(filepath, 'r') as file:
        data = json.load(file)

    # Custom key function for sorting that ignores non-alphanumeric characters
    def sort_key(key):
        return ''.join(filter(str.isalnum, key)).lower()

    # Sort the dictionary by keys using the custom key function
    sorted_data = {key: data[key] for key in sorted(data, key=sort_key)}

    # Write the sorted JSON object back to the file
    with open(filepath, 'w') as file:
        json.dump(sorted_data, file, indent=4, ensure_ascii=False)

import json

def read_json_keys(input_filepath, output_filepath):
    # Read JSON data from a file
    with open(input_filepath, 'r') as file:
        data = json.load(file)

    # Get all distinct keys
    keys = list(data.keys())

    # Write the keys to another JSON file
    with open(output_filepath, 'w') as file:
        json.dump(keys, file, indent=4)

def extract_and_save_json_values(input_filepath, output_filepath):
    # Read JSON data from a file
    with open(input_filepath, 'r') as file:
        data = json.load(file)

    # Set to hold all unique values
    values_set = set()

    # Iterate through each value in the dictionary
    for value in data.values():
        if isinstance(value, list):
            # Extend the set with all items from the list
            values_set.update(value)
        else:
            # Add the single item to the set
            values_set.add(value)

    # Convert the set to a list for JSON serialization
    unique_values = list(values_set)

    # Write the unique values to another JSON file
    with open(output_filepath, 'w') as file:
        json.dump(unique_values, file, indent=4)

import json

def invert_json_values_and_keys(input_filepath, output_filepath):
    # Read JSON data from a file
    with open(input_filepath, 'r') as file:
        data = json.load(file)

    # Dictionary to hold the inverted mapping
    inverted_dict = {}

    # Iterate through each key and value in the dictionary
    for key, value in data.items():
        if isinstance(value, list):
            for item in value:
                if item in inverted_dict:
                    inverted_dict[item].append(key)
                else:
                    inverted_dict[item] = [key]
        else:
            if value in inverted_dict:
                inverted_dict[value].append(key)
            else:
                inverted_dict[value] = [key]

    # Write the inverted dictionary to another JSON file
    with open(output_filepath, 'w') as file:
        json.dump(inverted_dict, file, indent=4)


def count_prms_in_utt(prms_filepath, large_json_filepath, output_filepath):
    # Load the list of PRMs from prms.json
    with open(prms_filepath, 'r') as file:
        prms = json.load(file)
    
    # Initialize a dictionary to hold the counts
    prms_count = {prm: 0 for prm in prms}

    # Load the large JSON data
    with open(large_json_filepath, 'r') as file:
        data = json.load(file)

    # Iterate through each entry in the large JSON
    for entry in data:
        if 'utt' in entry:
            utt_content = entry['utt']
            # Check if the utterance is a list and handle accordingly
            if isinstance(utt_content, list):
                words = set(word.lower() for sentence in utt_content for word in sentence.split())
            else:
                words = set(utt_content.lower().split())
            # Check each PRM against words in the utterance
            for prm in prms:
                if prm.lower() in words:
                    prms_count[prm] += 1

    # Create a sparse dictionary containing only PRMs that have been found at least once
    sparse_prms_count = {key: value for key, value in prms_count.items() if value > 0}

    # Write the sparse dictionary to a JSON file
    with open(output_filepath, 'w') as file:
        json.dump(sparse_prms_count, file, indent=4)


# Replace 'path_to_your_json_file.json' with the path to your JSON file
if __name__ == "__main__":
    main()
