import pandas as pd
import os

def create_intent_dataset_from_folder(folder_path, output_csv_filename):
    """
    Reads all .txt files from a specified folder, uses their filenames as
    intent labels, and compiles the contents into a single CSV file.
    """
    all_data = []

    print(f"Starting to process files from folder: '{folder_path}'...")

    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found. Please make sure the folder exists.")
        return

    # Loop through each file in the specified directory
    for filename in os.listdir(folder_path):
        # Process only text files
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            try:
                # Extract the intent from the filename
                # os.path.splitext splits the filename and extension, we take the first part
                intent_label = os.path.splitext(filename)[0]
                print(f"Processing file: '{filename}' -> Intent: '{intent_label}'")

                # Read each line from the file
                with open(file_path, 'r', encoding='utf-8') as f:
                    utterances = f.readlines()

                    # Process each line (utterance) in the file
                    for utterance in utterances:
                        # Clean up the line by removing leading/trailing whitespace
                        cleaned_utterance = utterance.strip()

                        # Ensure the line is not empty before adding it
                        if cleaned_utterance:
                            # Append the data as a dictionary to list
                            all_data.append({
                                'utterance': cleaned_utterance,
                                'intent': intent_label
                            })
            except Exception as e:
                print(f"An error occurred while processing '{filename}': {e}")

    # Convert the list of dictionaries to a pandas DataFrame
    if not all_data:
        print("No data was processed. The output file will not be created.")
        return

    df = pd.DataFrame(all_data)

    # Save the DataFrame to a CSV file
    try:
        df.to_csv(output_csv_filename, index=False)
        print(f"\nSuccessfully created dataset with {len(df)} records.")
        print(f"Data saved to '{output_csv_filename}'")
    except Exception as e:
        print(f"An error occurred while saving the CSV file: {e}")


if __name__ == "__main__":
    intents_folder = "Intents"
    output_filename = 'real_estate_intent_dataset.csv'
    create_intent_dataset_from_folder(intents_folder, output_filename)
