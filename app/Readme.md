# PDF Outline Extractor: Connecting the Dots Through Docs

This solution addresses the "Round 1A: Understand Your Document" challenge by extracting a structured outline (Title, H1, H2, H3) from PDF documents and outputting it in a clean, hierarchical JSON format.

## 🚀 Approach

The core of this solution lies in a multi-pronged approach to accurately identify headings, moving beyond simple font size heuristics:

1.  **Robust PDF Parsing**: We utilize `PyMuPDF` (fitz) for efficient and detailed extraction of text blocks, including their content, font size, font name, and bounding box (position) on the page. This granular information is crucial for accurate classification.

2.  **Hybrid Heading Classification (NLP + Heuristics)**:
    * **NLP-driven Classification**: The primary method for identifying heading levels is a pre-trained **Logistic Regression model** powered by `TfidfVectorizer`. This model is trained on diverse text samples (simulated for this challenge) to understand patterns indicative of headings (e.g., specific phrases, common heading structures). This allows the solution to generalize better across different PDF layouts where font sizes alone might be misleading.
    * **Heuristic Fallback and Enhancement**: While NLP is central, heuristics (like font size, bolding, and relative position) are used as a fallback mechanism if the NLP model is not available or encounters issues. More importantly, heuristics are also used to *refine* the NLP model's predictions, especially for edge cases or to provide initial strong signals (e.g., the largest text on the first page is highly likely the title).
    * **Hierarchical Enforcement**: A post-processing step ensures the logical hierarchy of headings (e.g., an H2 must follow an H1, an H3 must follow an H2). This step also handles redundancy, preventing duplicate consecutive headings.

3.  **Title Detection**: A specific heuristic is applied to identify the document title, typically looking for the largest and most prominent text block on the first page.

4.  **JSON Output**: The extracted title and hierarchical headings are formatted into the specified JSON structure.

## 📦 Models and Libraries Used

* **`PyMuPDF` (fitz)**: For efficient PDF parsing and text extraction.
* **`scikit-learn`**: For building and utilizing the text classification model (Logistic Regression and TfidfVectorizer).
* **`NLTK`**: For basic text preprocessing (tokenization, stopword removal) as part of the NLP pipeline.
* **`joblib`**: For serializing (saving/loading) the trained NLP model.

**Model Size Compliance**: The chosen `TfidfVectorizer` with `max_features` and `LogisticRegression` are inherently lightweight models, ensuring the total model size remains well within the 200MB constraint.

## ⚙️ How to Build and Run Your Solution

**Note**: For competition submission, your solution will be built and run using the commands specified in the "Expected Execution" section of the challenge brief. The instructions below are for local development and testing.

### 1. **Clone the Repository (or create the file structure)**

First, ensure you have the correct folder structure as described in the challenge.