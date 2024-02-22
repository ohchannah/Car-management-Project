import os

def add_zero_to_all_lines(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        with open(file_path, 'w') as file:
            for line in lines:
                modified_line = '0' + line
                file.write(modified_line)

        print(f"'0' added to all lines in {file_path}")

    except Exception as e:
        print(f"Error adding '0' to {file_path}: {e}")

def add_zero_to_all_lines_in_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            add_zero_to_all_lines(file_path)

# Replace 'your_directory_path' with the actual path to your directory containing the text files
directory_path = 'C:/Users/VCL/Desktop/d0ng1nah/datasets/new/labels'
add_zero_to_all_lines_in_directory(directory_path)
