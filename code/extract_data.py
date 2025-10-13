import json
from nuscenes.nuscenes import NuScenes

# --- Configuration ---
# Replace 'your_questions.json' with the actual path to your JSON file
questions_file = 'data/NuScenes_val_questions.json'
# Replace with your nuScenes data root if different
dataroot = 'nuscenes'
# Replace with the nuScenes version you are using (e.g., 'v1.0-mini')
version = 'v1.0-trainval'

# --- Load Questions ---
try:
    with open(questions_file, 'r') as f:
        questions_data = json.load(f)['questions']
    print(f"Successfully loaded questions from {questions_file}")
except FileNotFoundError:
    print(f"Error: Questions file not found at {questions_file}")
    # You might want to exit or handle this error appropriately
    questions_data = []
print(f"Total questions loaded: {len(questions_data)}")
# --- Initialize NuScenes ---
try:
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    print("NuScenes dataset initialized successfully.")
except Exception as e:
    print(f"Error initializing NuScenes dataset: {e}")
    nusc = None # Set nusc to None if initialization fails

# --- Process Questions and Extract Annotations ---
if nusc and questions_data:
    # Extract all sample tokens from questions
    question_sample_tokens = set()
    for question_info in questions_data:
        sample_token = question_info.get('sample_token')
        if sample_token:
            question_sample_tokens.add(sample_token)
    
    print(f"Unique sample tokens in questions: {len(question_sample_tokens)}")
    
    # Get all available sample tokens from nuScenes dataset
    available_sample_tokens = set()
    for sample in nusc.sample:
        available_sample_tokens.add(sample['token'])
    
    print(f"Available sample tokens in nuScenes dataset: {len(available_sample_tokens)}")
    
    # Find missing tokens
    missing_tokens = question_sample_tokens - available_sample_tokens
    found_tokens = question_sample_tokens & available_sample_tokens
    
    print(f"\n--- Token Analysis ---")
    print(f"Tokens found in dataset: {len(found_tokens)}")
    print(f"Tokens missing from dataset: {len(missing_tokens)}")
    
    
    # Save missing tokens to file
    if missing_tokens:
        missing_tokens_file = 'missing_tokens.json'
        with open(missing_tokens_file, 'w') as f:
            json.dump(list(missing_tokens), f, indent=4)
        print(f"\nMissing tokens saved to {missing_tokens_file}")

# --- Process Questions and Extract Annotations ---
extracted_data = {}

if nusc and questions_data:
    processed_sample_tokens = set() # Keep track of processed tokens to avoid redundancy
    successful_extractions = 0
    failed_extractions = 0

    for question_info in questions_data:
        sample_token = question_info.get('sample_token')

        if sample_token and sample_token not in processed_sample_tokens:
            #print(f"\nProcessing sample token: {sample_token}")
            processed_sample_tokens.add(sample_token)

            try:
                # Get the sample record
                sample_record = nusc.get('sample', sample_token)

                # Get all annotation tokens for the sample
                annotation_tokens = sample_record.get('anns', [])

                sample_annotations_data = []

                # Iterate through each annotation and extract details
                for ann_token in annotation_tokens:
                    try:
                        annotation_metadata = nusc.get('sample_annotation', ann_token)

                        # Extract relevant information from the annotation
                        annotation_details = {
                            'annotation_token': ann_token,
                            'category_name': annotation_metadata.get('category_name'),
                            'translation': annotation_metadata.get('translation'),
                            'size': annotation_metadata.get('size'),
                            'rotation': annotation_metadata.get('rotation'),
                            'num_lidar_pts': annotation_metadata.get('num_lidar_pts'),
                            'num_radar_pts': annotation_metadata.get('num_radar_pts'),
                            'visibility_level': nusc.get('visibility', annotation_metadata.get('visibility_token', {})).get('level') if annotation_metadata.get('visibility_token') else None,
                            'attribute_names': [nusc.get('attribute', attr_token).get('name') for attr_token in annotation_metadata.get('attribute_tokens', [])]
                            # Add other fields if needed
                        }
                        sample_annotations_data.append(annotation_details)

                    except Exception as e:
                        print(f"Error processing annotation {ann_token} for sample {sample_token}: {e}")

                # Store the extracted data for this sample token
                extracted_data[sample_token] = sample_annotations_data
                successful_extractions += 1

            except Exception as e:
                print(f"Error processing sample token {sample_token}: {e}")
                failed_extractions += 1

    print(f"\n--- Processing Summary ---")
    print(f"Successful extractions: {successful_extractions}")
    print(f"Failed extractions: {failed_extractions}")


# --- Output Results ---
print("\n--- Extracted Annotation Data ---")
# You can print the dictionary or save it to a file
#print(json.dumps(extracted_data, indent=4))
print(f"Total samples processed: {len(extracted_data)}")
# Example of saving to a JSON file
output_file = 'data/extracted_annotations.json'
with open(output_file, 'w') as outfile:
    for key in extracted_data:
        json.dump({key: extracted_data[key]}, outfile)
        outfile.write('\n')
print(f"\nExtracted data saved to {output_file}")