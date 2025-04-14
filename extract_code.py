# extract_code.py
import os
import re
from pathlib import Path

docs_dir = Path("docs")
src_dir = Path("src")
src_dir.mkdir(exist_ok=True)

# Regex to find fenced code blocks (rust or python)
# ```(rust|python)
# (code content)
# ```
code_block_regex = re.compile(r"```(rust|python)\s*([\s\S]*?)\s*```")

def process_markdown_file(md_file_path):
    """
    Processes a single markdown file:
    - Finds rust/python code blocks.
    - Extracts code to new files in src/.
    - Replaces code blocks with links in the original md file.
    """
    print(f"Processing: {md_file_path}")
    try:
        content = md_file_path.read_text(encoding='utf-8')
        new_content = content
        matches = list(code_block_regex.finditer(content))
        base_filename = md_file_path.stem # Filename without extension

        for i, match in enumerate(matches):
            lang = match.group(1)
            code = match.group(2).strip()
            block_to_replace = match.group(0)

            # Determine file extension
            ext = ".rs" if lang == "rust" else ".py"

            # Create a unique filename for the source file
            src_filename = f"{base_filename}_{i+1}{ext}"
            src_file_path = src_dir / src_filename

            # Write code to the new source file
            try:
                src_file_path.write_text(code, encoding='utf-8')
                print(f"  -> Extracted code to: {src_file_path}")

                # Create the relative link from docs/file.md to src/file_n.ext
                relative_link = f"../src/{src_filename}"
                link_markdown = f"[Link to `{src_filename}`]({relative_link})"

                # Replace the original code block with the link
                # Use a simple string replacement; assumes blocks are unique enough
                # For more robustness, might need index-based replacement if identical blocks exist
                if block_to_replace in new_content:
                     new_content = new_content.replace(block_to_replace, link_markdown, 1)
                else:
                     print(f"  [Warning] Could not find exact block to replace for {src_filename} in {md_file_path}. Manual check needed.")


            except IOError as e:
                print(f"  [Error] Failed to write {src_file_path}: {e}")
            except Exception as e:
                 print(f"  [Error] An unexpected error occurred while processing {src_filename}: {e}")


        # Write the modified content back to the markdown file if changes were made
        if new_content != content:
            try:
                md_file_path.write_text(new_content, encoding='utf-8')
                print(f"  -> Updated links in: {md_file_path}")
            except IOError as e:
                print(f"  [Error] Failed to update {md_file_path}: {e}")
        else:
            print(f"  -> No code blocks found or no changes needed for: {md_file_path}")


    except FileNotFoundError:
        print(f"[Error] File not found: {md_file_path}")
    except Exception as e:
        print(f"[Error] Failed to process {md_file_path}: {e}")

# Iterate over all markdown files in the docs directory
for md_file in docs_dir.glob("*.md"):
    process_markdown_file(md_file)

print("\nScript finished.")