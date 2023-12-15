# Sample data - replace this with your actual data
bufferData = [
    {'id': 1, 'text': 'Text for ID 1'},
    {'id': 2, 'text': 'Text for ID 2'},
    {'id': 3, 'text': 'Text for ID 3'}
]

# ID to search for - replace this with your actual ID
id_to_search = 2

# Initialize transcrib as an empty string
transcrib = ''

# Loop through bufferData to find the matching ID
for item in bufferData:
    if 'id' in item and item['id'] == id_to_search:
        transcrib = item['text']

# Display the result
if transcrib:
    print(f"Found text for ID {id_to_search}: {transcrib}")
else:
    print(f"No text found for ID {id_to_search}")
