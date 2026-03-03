
import os

def split_file_by_words(filepath, words_per_chunk=1200):
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by any whitespace to get words
    words = content.split()
    
    filename, ext = os.path.splitext(filepath)
    
    for i in range(0, len(words), words_per_chunk):
        chunk_words = words[i : i + words_per_chunk]
        chunk_content = ' '.join(chunk_words)
        part_num = (i // words_per_chunk) + 1
        chunk_filepath = f"{filename}_word_part{part_num}{ext}"
        with open(chunk_filepath, 'w', encoding='utf-8') as f_chunk:
            f_chunk.write(chunk_content)
        print(f"Created {chunk_filepath} ({len(chunk_words)} words)")

if __name__ == "__main__":
    # Path relative to the script's likely execution location or absolute
    path = '/home/tarakesh/Work/Repo/measurement-free-quantum-classifier/interference_quantum_classifier.txt'
    split_file_by_words(path)
