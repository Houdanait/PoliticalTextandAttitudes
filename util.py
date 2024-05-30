import h5py
from collections import defaultdict
import json
from collections import Counter

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
        
    print("Potential PRM Length:", len(prms))
    
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

def print_sorted_json(input_filepath):
    # Load the JSON data from the file
    with open(input_filepath, 'r') as file:
        data = json.load(file)

    # Sort the dictionary by values in descending order
    sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))

    # Print the sorted dictionary
    for key, value in sorted_data.items():
        print(f"{key}: {value}")

# Function to group entries by (transcript_id, statement)
def group_by_key(json_list):
    grouped = defaultdict(list)
    for item in json_list:
        key = (item['transcript_id'], item['statement'])
        grouped[key].append(item)
    return grouped


import numpy as np
from scipy.spatial.distance import cosine


def identify_similar_parts(vectors, threshold=0.7):
    num_vectors = len(vectors)
    num_dimensions = len(vectors[0])
    
    dimension_similarities = np.zeros(num_dimensions)
    
    for i in range(num_vectors):
        for j in range(i + 1, num_vectors):
            vec1, vec2 = vectors[i], vectors[j]
            for k in range(num_dimensions):
                dim_vec1 = np.zeros(num_dimensions)
                dim_vec2 = np.zeros(num_dimensions)
                dim_vec1[k] = vec1[k]
                dim_vec2[k] = vec2[k]
                dimension_similarities[k] += 1 - cosine(dim_vec1, dim_vec2)
    
    dimension_similarities /= (num_vectors * (num_vectors - 1) / 2)
    similar_dimensions = np.where(dimension_similarities > threshold)[0]
    
    return similar_dimensions, dimension_similarities

def extract_vectors_from_tuples(data):
    return [item[0] for item in data]

def get_category(h5_file, category):
    samples = []
    with h5py.File(h5_file, 'r') as f:
        for idx in f.keys():
            group = f[idx]
            matched_terms = eval(group['matched_terms'][()].decode('utf-8'))
            hedge_terms = {term: value for term, value in matched_terms.items() if value == category}
            if hedge_terms:
                hedged_item = {
                    'transcript_id': group['transcript_id'][()],
                    'statement_id': group['statement_id'][()],
                    'original_string': group['original_string'][()].decode('utf-8'),
                    'tokens': group['tokens'][()].astype(str).tolist(),
                    'embeddings': group['embeddings'][()],
                    'matched_terms': hedge_terms
                }
                samples.append(hedged_item)
    print(f"Category: {category} - {len(samples)} sample sentences found.")
    return samples

def get_dual_matches(h5_file):
    term_counts = {}

    with h5py.File(h5_file, 'r') as f:
        for idx in f.keys():
            group = f[idx]
            matched_terms = eval(group['matched_terms'][()].decode('utf-8'))
            for term, value in matched_terms.items():
                if term not in term_counts:
                    term_counts[term] = Counter()
                term_counts[term][value] += 1

    return term_counts

def find_phrase_indices(tokens, phrase, tokenizer):
    """
    Find the indices of the phrase in the token list.
    """
    phrase_tokens = tokenizer.tokenize(phrase)
    phrase_length = len(phrase_tokens)
    
    for i in range(len(tokens) - phrase_length + 1):
        if tokens[i:i + phrase_length] == phrase_tokens:
            return list(range(i, i + phrase_length))
    return []

def get_phrase_vector(phrase, tokens, output):
    """
    Get the vector for a multi-word phrase by averaging the embeddings of each word and its subwords in the phrase.
    """
    words = phrase.split()
    token_indices = []
    
    for word in words:
        word_indices = []
        for i, token in enumerate(tokens):
            # Check for exact match or subword match
            if token == word or token.lstrip("##") == word:
                word_indices.append(i)
        token_indices.extend(word_indices)
    
    if not token_indices:
        return None
    
    # Extract embeddings for specific word indices and average them
    token_embeddings = output.last_hidden_state[0, token_indices, :]
    phrase_vector = token_embeddings.mean(dim=0).detach()
    return phrase_vector
    
    
def get_relevant_segment(tokens, target_indices, max_length=512):
    """
    Get the relevant segment of tokens centered around the target indices within the max_length.
    """
    start = max(0, min(target_indices) - (max_length // 2))
    end = start + max_length
    if end > len(tokens):
        end = len(tokens)
        start = max(0, end - max_length)
    return tokens[start:end], start, end

def get_vectors(item, category, model, tokenizer, max_tokens) -> list:
    string = item["original_string"]
    string = string.replace("<", "").replace(">", "")
    
    # First tokenize without truncation to get the full token list
    full_encoded_input = tokenizer(string, return_tensors='pt', truncation=False)
    full_tokens = tokenizer.convert_ids_to_tokens(full_encoded_input['input_ids'][0])
    
    matched_terms = item['matched_terms']
    vectors = []
    
    for term, value in matched_terms.items():
        if value == category:
            # Find the indices of the target term or phrase in the full token list
            if " " in term:  # It's a phrase
                term_indices = find_phrase_indices(full_tokens, term, tokenizer)
            else:  # It's a single word or subword
                term_indices = [i for i, token in enumerate(full_tokens) if token == term or token.lstrip("##") == term]

            if term_indices:
                # Get the relevant segment of tokens
                relevant_tokens, start, end = get_relevant_segment(full_tokens, term_indices, max_length=max_tokens)
                
                # Retokenize the relevant segment
                relevant_string = tokenizer.convert_tokens_to_string(relevant_tokens)
                encoded_input = tokenizer(relevant_string, return_tensors='pt', truncation=True, max_length=max_tokens)
                
                tokens = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])
                output = model(**encoded_input)
                
                if " " in term:  # It's a phrase
                    phrase_vector = get_phrase_vector(term, tokens, output)
                    if phrase_vector is not None:
                        vectors.append((phrase_vector, term, category))
                else:  # It's a single word or subword
                    word_indices = [i for i, token in enumerate(tokens) if token == term or token.lstrip("##") == term]
                    if word_indices:
                        word_embeddings = output.last_hidden_state[0, word_indices, :]
                        word_vector = word_embeddings.mean(dim=0).detach()
                        vectors.append((word_vector, term, category))
    return vectors

def get_category_vectors(data, category, model, tokenizer, max_tokens) -> list:
    vectors = []
    for item in data:
        vectors.extend(get_vectors(item, category, model, tokenizer, max_tokens))
    return vectors

# Replace 'path_to_your_json_file.json' with the path to your JSON file
if __name__ == "__main__":
    main()



