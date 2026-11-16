import os
from pypdf import PdfReader
from docx import Document


def extract_text_from_file(file_obj, filename: str) -> str:
    """
    Extracts text from PDF, DOCX, or TXT files.
    `file_obj` is the Django UploadedFile object.
    """
    ext = os.path.splitext(filename)[1].lower()
    text = ""

    try:
        if ext == ".txt":
            # Decode text file
            text = file_obj.read().decode("utf-8", errors="ignore")

        elif ext == ".pdf":
            reader = PdfReader(file_obj)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        elif ext == ".docx":
            doc = Document(file_obj)
            for para in doc.paragraphs:
                text += para.text + "\n"
            # Also extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text += cell.text + " "
                    text += "\n"

        else:
            raise ValueError(
                f"Unsupported file format: {ext}. Only .pdf, .docx, and .txt are allowed.")

    except Exception as e:
        raise ValueError(
            f"Failed to extract text from {filename}. Error: {str(e)}")

    if not text.strip():
        raise ValueError(
            f"The document {filename} appears to be empty or contains no readable text.")

    return text.strip()
