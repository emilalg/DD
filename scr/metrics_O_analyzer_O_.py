import argparse
import os

def extract_number(line, key):
    """
    Extracts a numeric value for the given key from a line of text.
    """
    try:
        start_index = line.find(f'"{key}"') + len(f'"{key}": ')
        end_index = line.find(',', start_index)
        if end_index == -1:  # If there's no trailing comma, find the end of the line
            end_index = None
        return float(line[start_index:end_index])
    except ValueError:
        return None

def parse_trial_data(filename):
    """
    Parses the file, looking directly for 'trial_number', 'fscore', and capturing 'parameters'.
    """
    best_fscore = float('-inf')
    best_trial_number = None
    best_parameters_str = ""  # Store the best trial's parameters as a string
    temp_parameters_str = ""  # Temporarily store parameters for each trial
    capture_parameters = False  # Flag to start capturing parameter lines
    
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if '"trial_number":' in line:
                # When a new trial starts, check if the last one had the best fscore
                trial_number = extract_number(line, 'trial_number')
                temp_parameters_str = ""  # Reset parameters string for the new trial
                capture_parameters = False  # Reset flag
            elif '"fscore":' in line:
                fscore = extract_number(line, 'fscore')
                if fscore > best_fscore:
                    best_fscore = fscore
                    best_trial_number = trial_number
                    best_parameters_str = temp_parameters_str  # Save parameters for the best trial
            elif line.startswith('"parameters":'):
                capture_parameters = True  # Start capturing parameters
               
            elif capture_parameters:
                if line.startswith('}') or line.endswith('}'):
                    temp_parameters_str += line + "\n"  # Capture the closing of parameters
                    capture_parameters = False  # Stop capturing after the parameters block ends
                else:
                    temp_parameters_str += line + "\n"  # Continue capturing parameters
                
    return best_trial_number, best_fscore, best_parameters_str.strip()

def main():
    parser = argparse.ArgumentParser(description="Analyze trial data from a file.")
    parser.add_argument('--filename', required=True, help="The filename to analyze.")
    
    args = parser.parse_args()
    
    # Prepend the fixed path to the filename provided
    full_path = f"test_output/hypertuner/Default/{args.filename}"
    
    best_trial_number, best_fscore, best_parameters_str = parse_trial_data(full_path)
    
    # Generate output filename
    base_filename, _ = os.path.splitext(args.filename)
    output_filename = f"test_output/hypertuner/Default/{base_filename}_analyzed.txt"
    
    # Write the result to a file
    with open(output_filename, 'w') as output_file:
        if best_trial_number is not None:
            output_content = f"The best fscore is {best_fscore} found in trial number {best_trial_number}.\nParameters:\n{best_parameters_str}"
            print(output_content)  # Also print to console
            output_file.write(output_content + "\n")
        else:
            print("No valid trials found.")
            output_file.write("No valid trials found.\n")

if __name__ == "__main__":
    main()
