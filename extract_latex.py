import re
import os

def extract_text(tex_path, output_path):
    with open(tex_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Remove comments
    content = re.sub(r'(?<!\\)%.*$', '', content, flags=re.MULTILINE)

    # 2. Remove math environments
    # Display math: \begin{equation}...\end{equation}, \begin{align}...\end{align}, etc.
    math_envs = ['equation', 'align', 'displaymath', 'gather', 'alignat', 'flalign', 'multline']
    for env in math_envs:
        # Using [^]*? for non-greedy multiline match
        pattern = rf'\\begin\{{{env}\*?\}}[\s\S]*?\\end\{{{env}\*?\}}'
        content = re.sub(pattern, '', content)
    
    # \[ ... \] and $$ ... $$
    content = re.sub(r'\\\[[\s\S]*?\\\]', '', content)
    content = re.sub(r'\$\$[\s\S]*?\$\$', '', content)

    # 3. Remove Figures and Tables EXCEPT the caption text might need to be removed specifically
    # but the user said "leave all the figure explanation and the equations in the tex file"
    # Wait, the request says: "extract all text/paragraph lines from the .tex file and put it in the txt file leaveall the figure explaantion and the equations inthe tex file"
    # This means the .txt file should NOT have equations and figure explanations.
    
    # Figure and Table captions
    content = re.sub(r'\\caption\{([\s\S]*?)\}', '', content)
    
    # Remove figure/table environments entirely if they only contain non-text/explanation
    content = re.sub(r'\\begin\{(figure|table).?\}([\s\S]*?)\\end\{\1.?\}', '', content)

    # 4. Remove inline math $ ... $ and \( ... \)
    content = re.sub(r'\$[\s\S]*?\$', '', content)
    content = re.sub(r'\\\([\s\S]*?\\\)', '', content)

    # 5. Clean up other LaTeX commands but keep their content if applicable
    # \section{...}, \subsection{...} -> ...
    content = re.sub(r'\\(section|subsection|subsubsection|paragraph|subparagraph)\*?\{([\s\S]*?)\}', r'\2\n', content)
    
    # \textbf{...}, \textit{...}, \emph{...} -> ...
    content = re.sub(r'\\(textbf|textit|emph|texttt|textsc)\{([\s\S]*?)\}', r'\2', content)

    # Handle citations \cite{...} -> [Ref] or just remove
    content = re.sub(r'\\cite\{.*?\}', '', content)
    
    # Handle labels \label{...} and references \ref{...}
    content = re.sub(r'\\label\{.*?\}', '', content)
    content = re.sub(r'\\ref\{.*?\}', '', content)
    
    # Remove preamble \documentclass... \begin{document}
    content = re.sub(r'[\s\S]*?\\begin\{document\}', '', content)
    # Remove end document
    content = re.sub(r'\\end\{document\}[\s\S]*$', '', content)

    # Remove itemize/enumerate tags
    content = re.sub(r'\\begin\{(itemize|enumerate)\}', '', content)
    content = re.sub(r'\\end\{(itemize|enumerate)\}', '', content)
    content = re.sub(r'\\item', '\n- ', content)

    # Final cleanup of remaining commands \command...
    content = re.sub(r'\\[a-zA-Z]+\*?(\{.*?\})?', '', content)
    
    # Fix whitespace
    content = re.sub(r'\n{3,}', '\n\n', content)
    content = content.strip()

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    tex_file = "/home/tarakesh/Work/Repo/measurement-free-quantum-classifier/paper_writeup/Isdo_theory_5/interference_quantum_classifier.tex"
    txt_file = "/home/tarakesh/Work/Repo/measurement-free-quantum-classifier/interference_quantum_classifier.txt"
    extract_text(tex_file, txt_file)
