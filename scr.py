import json

def extract_code_from_notebook(notebook_path, output_file):
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as nb_file:
            notebook_content = json.load(nb_file)
        
        # Check if it's a valid notebook
        if 'cells' not in notebook_content:
            print("Invalid Jupyter Notebook file.")
            return
        
        # Extract code cells
        code_cells = [
            cell['source']
            for cell in notebook_content['cells']
            if cell['cell_type'] == 'code'
        ]
        
        # Flatten the list and join with newline
        all_code = '\n\n'.join(''.join(cell) for cell in code_cells)
        
        # Write to the output Python file
        with open(output_file, 'w', encoding='utf-8') as py_file:
            py_file.write(all_code)
        
        print(f"Code successfully extracted to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Specify the input notebook and output script file
notebook_path = "C:/Users/Morsi Store DZ/sentinent-analysis-using-gemini-api/src/amazon_f_hanson.ipynb"  # Replace with your notebook path
output_file = "output_script.py"       # Replace with desired output filename

# Extract code
extract_code_from_notebook(notebook_path, output_file)