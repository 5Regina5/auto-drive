import json
import os

# --- Configuration ---
questions_file = 'data/NuScenes_val_questions.json'
raw_annotations_dir = 'raw_questions'

# --- Define target categories ---
target_categories = ['existence', 'counting', 'query_object', 'query_status', 'comparison']

def load_questions():
    """Load questions from JSON file"""
    try:
        with open(questions_file, 'r') as f:
            questions_data = json.load(f)['questions']
        print(f"Successfully loaded questions from {questions_file}")
        print(f"Total questions loaded: {len(questions_data)}")
        return questions_data
    except FileNotFoundError:
        print(f"Error: Questions file not found at {questions_file}")
        return []

def create_output_directory():
    """Create raw_annotations directory if it doesn't exist"""
    if not os.path.exists(raw_annotations_dir):
        os.makedirs(raw_annotations_dir)
        print(f"Created directory: {raw_annotations_dir}")

def categorize_questions_by_type_and_height(questions_data):
    """Categorize questions by type and height level (h0/h1)"""
    # Initialize data structures for each category and height level
    categorized_data = {}
    for category in target_categories:
        categorized_data[category] = {'h0': [], 'h1': []}
    
    processed_count = 0
    categorization_stats = {cat: {'h0': 0, 'h1': 0} for cat in target_categories}
    
    # Template type mapping to our categories
    template_mapping = {
        'exist': 'existence',
        'count': 'counting', 
        'object': 'query_object',
        'status': 'query_status',
        'comparison': 'comparison'
    }
    
    for question_info in questions_data:
        try:
            # Extract template type from the question data
            template_type = question_info.get('template_type', '').lower()
            
            # Map template type to our target categories
            question_type = template_mapping.get(template_type)
            
            if question_type and question_type in target_categories:
                # Determine height level based on num_hop
                # h0 for num_hop = 0, h1 for num_hop = 1 (or > 0)
                num_hop = question_info.get('num_hop', 0)
                height_level = 'h0' if num_hop == 0 else 'h1'
                
                # Add to appropriate category and height level
                categorized_data[question_type][height_level].append(question_info)
                categorization_stats[question_type][height_level] += 1
                processed_count += 1
                
        except Exception as e:
            print(f"Error processing question: {e}")
            continue
    
    print(f"\n--- Categorization Summary ---")
    print(f"Total questions processed: {processed_count}")
    
    for category in target_categories:
        h0_count = categorization_stats[category]['h0']
        h1_count = categorization_stats[category]['h1']
        total_count = h0_count + h1_count
        print(f"{category}: {total_count} total (h0: {h0_count}, h1: {h1_count})")
    
    return categorized_data

def save_categorized_data(categorized_data):
    """Save categorized data to separate JSON files"""
    saved_files = []
    
    for category in target_categories:
        for height_level in ['h0', 'h1']:
            output_file = os.path.join(raw_annotations_dir, f'{category}_{height_level}.json')
            data_to_save = categorized_data[category][height_level]
            
            with open(output_file, 'w', encoding='utf-8') as outfile:
                json.dump(data_to_save, outfile, indent=2, ensure_ascii=False)
            
            print(f"Saved {len(data_to_save)} questions for {category}_{height_level} to {output_file}")
            saved_files.append(output_file)
    
    return saved_files

def main():
    """Main execution function"""
    print("=== Question Categorization Script ===")
    
    # Load questions data
    questions_data = load_questions()
    if not questions_data:
        print("No questions data loaded. Exiting.")
        return
    
    # Create output directory
    create_output_directory()
    
    # Categorize questions by type and height level
    categorized_data = categorize_questions_by_type_and_height(questions_data)
    
    # Save categorized data to files
    saved_files = save_categorized_data(categorized_data)
    
    print(f"\n=== Processing Complete ===")
    print(f"Generated {len(saved_files)} files in {raw_annotations_dir} directory:")
    for file_path in saved_files:
        print(f"  - {file_path}")

if __name__ == "__main__":
    main()