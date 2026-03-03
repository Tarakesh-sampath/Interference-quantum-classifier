import re
import os

def cleanup_file(filepath):
    print(f"Cleaning up {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Remove carriage returns
    content = content.replace('\r', '')

    # 2. Process line by line
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Strip leading/trailing whitespace
        stripped = line.strip()
        if stripped:
            # Replace multiple spaces with a single space
            collapsed = re.sub(r' +', ' ', stripped)
            cleaned_lines.append(collapsed)

    # 3. Join without adding extra empty lines
    final_content = '\n'.join(cleaned_lines)
    
    # Trim leading/trailing newlines from the whole file
    final_content = final_content.strip()

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(final_content)
    print("Done.")

if __name__ == "__main__":
    path = '/home/tarakesh/Work/Repo/measurement-free-quantum-classifier/interference_quantum_classifier.txt'
    cleanup_file(path)
