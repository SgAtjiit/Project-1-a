import os
import json
import re
from .pdf_parser import PDFParser
from .heading_classifier import HeadingClassifier

def process_pdf(pdf_path, output_dir):
    """
    Processes a single PDF file to extract its title and hierarchical outline.
    """
    parser = PDFParser(pdf_path)
    classifier = HeadingClassifier() # Loads the trained NLP model

    title = parser.get_title_candidate()
    text_blocks = parser.extract_text_blocks()

    outline = []
    current_h1_index = -1
    current_h2_index = -1

    for block in text_blocks:
        # Skip blocks that are too short or likely just page numbers/footers
        if len(block["text"]) < 5 or re.match(r'^\d+$', block["text"].strip()):
            continue

        predicted_level = classifier.classify_heading(block)

        # Enforce hierarchical order and ensure consecutive same-level headings are distinct
        if predicted_level == "Title" and not title: # Prioritize the first detected title
            title = block["text"]
            continue # Title is handled separately

        if predicted_level == "H1":
            # Check for redundancy: Don't add if the last H1 is identical
            if not outline or not (outline[-1]["level"] == "H1" and outline[-1]["text"] == block["text"]):
                outline.append({"level": "H1", "text": block["text"], "page": block["page"]})
                current_h1_index = len(outline) - 1
                current_h2_index = -1 # Reset H2 for new H1
            continue

        if predicted_level == "H2":
            # Ensure an H2 appears after an H1
            if current_h1_index != -1:
                if not outline or not (outline[-1]["level"] == "H2" and outline[-1]["text"] == block["text"]):
                    outline.append({"level": "H2", "text": block["text"], "page": block["page"]})
                    current_h2_index = len(outline) - 1
            else: # If no H1 yet, treat as H1 if it's prominent
                if classifier._heuristic_classify(block) == "H1": # Check if it could be an H1 by heuristics
                     outline.append({"level": "H1", "text": block["text"], "page": block["page"]})
                     current_h1_index = len(outline) - 1
                     current_h2_index = -1

            continue

        if predicted_level == "H3":
            # Ensure an H3 appears after an H2 (and implicitly after an H1)
            if current_h2_index != -1:
                if not outline or not (outline[-1]["level"] == "H3" and outline[-1]["text"] == block["text"]):
                    outline.append({"level": "H3", "text": block["text"], "page": block["page"]})
            else: # If no H2 yet, treat as H2 if it's prominent
                if classifier._heuristic_classify(block) == "H2":
                    if current_h1_index != -1: # only if there is a H1
                        outline.append({"level": "H2", "text": block["text"], "page": block["page"]})
                        current_h2_index = len(outline) - 1
                    else: # if no H1, treat as H1 if prominent
                        if classifier._heuristic_classify(block) == "H1":
                            outline.append({"level": "H1", "text": block["text"], "page": block["page"]})
                            current_h1_index = len(outline) - 1
                            current_h2_index = -1
            continue
        
        # Consider the possibility that the title might be classified as H1 by the model
        # if the title detection heuristic fails or if the first H1 is indeed the title.
        if not title and (predicted_level == "H1" or predicted_level == "Title") and block["page"] == 1:
            title = block["text"]


    # Final check: If no title was found, use the first H1 candidate as the title if it makes sense.
    if not title and outline and outline[0]["level"] == "H1" and outline[0]["page"] == 1:
        title = outline.pop(0)["text"] # Remove it from outline if used as title

    output_data = {
        "title": title if title else "Untitled Document",
        "outline": outline
    }

    # Generate output filename
    base_filename = os.path.basename(pdf_path)
    name_without_ext = os.path.splitext(base_filename)[0]
    output_filename = os.path.join(output_dir, f"{name_without_ext}.json")

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Processed '{pdf_path}' and saved output to '{output_filename}'")


if __name__ == "__main__":
    input_dir = "/app/input"
    output_dir = "/app/output"

    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

    print(f"Starting PDF processing in {input_dir}")
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in {input_dir}. Please ensure PDFs are mounted to this directory.")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        print(f"Processing PDF: {pdf_path}")
        try:
            process_pdf(pdf_path, output_dir)
        except Exception as e:
            print(f"Failed to process {pdf_file}: {e}")