import random
import json

file_path = 'news_dialogue.json'

with open(file_path, 'r') as file:
    data = json.load(file)  # Attempt to load the JSON
    
# # Randomly sample 200 items from the data
sampled_data = random.sample(data, 200)

# Output the sampled data as a JSON file
with open('news_dialogue_sample.json', 'w') as file:
    json.dump(sampled_data, file)
