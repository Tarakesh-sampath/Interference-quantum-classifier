import os

EXCLUDE_DIRS = {
    'dataset', 
    'Documents', 
    'measurement_free_quantum_classifier.egg-info', 
    'status_reports', 
    #'results',
    '.git',
    '.venv',
    '__pycache__',
    'Archive_src',
    'research_docs',
}

EXCLUDE_EXTS = {'.pdf', '.png', '.jpg', '.jpeg', '.gif', '.pyc', '.o', '.a', '.so', '.exe', '.bin', '.egg-info', '.lock'}

def get_directory_structure(root_dir):
    lines = []
    for root, dirs, files in os.walk(root_dir):
        # Filter directories in-place
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        level = root.replace(root_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        lines.append(f"{indent}{os.path.basename(root)}/")
        
        # Skip files in root directory
        if root == root_dir:
            continue

        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            if not any(f.endswith(ext) for ext in EXCLUDE_EXTS) and not f.startswith('.'):
                lines.append(f"{sub_indent}{f}")
    return "\n".join(lines)

def get_file_contents(root_dir):
    content_blocks = []
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        # Skip files in root directory
        if root == root_dir:
            continue

        for f in files:
            file_path = os.path.join(root, f)
            rel_path = os.path.relpath(file_path, root_dir)
            
            if any(f.endswith(ext) for ext in EXCLUDE_EXTS) or f.startswith('.'):
                if f != '.python-version' and f != '.gitignore': # Allow some config files
                    continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    ext = os.path.splitext(f)[1].lstrip('.')
                    if not ext: ext = 'text'
                    content_blocks.append(f"## File: {rel_path}\n\n```{ext}\n{content}\n```\n")
            except Exception as e:
                # content_blocks.append(f"## File: {rel_path}\n\nError reading file: {e}\n")
                pass
                
    return "\n".join(content_blocks)

if __name__ == "__main__":
    root = "/home/tarakesh/Work/Repo/measurement-free-quantum-classifier"
    output_file = "/home/tarakesh/Work/Repo/measurement-free-quantum-classifier/project_summary.md"
    
    structure = get_directory_structure(root)
    contents = get_file_contents(root)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Project Summary\n\n")
        f.write("## Directory Structure\n\n```\n")
        f.write(structure)
        f.write("\n```\n\n")
        f.write(contents)
    
    print(f"Generated {output_file}")
