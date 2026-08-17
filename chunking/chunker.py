import os
import re
from typing import List, Dict, Any

class DocumentChunker:
    """
    Structure-aware document chunker that extracts section headers 
    and preserves metadata (doc_name, section, chunk_id).
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def clean_text(self, text: str) -> str:
        """Removes excessive whitespace and standardizes newlines."""
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Reads a markdown document, parses sections, and splits into structured chunks."""
        doc_name = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        cleaned_content = self.clean_text(content)
        
        # Split by Markdown H2 Headers (## )
        sections = re.split(r'(?=\n##\s+)', cleaned_content)
        chunks = []
        chunk_idx = 0

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # Extract Section Title
            section_match = re.match(r'##\s+(.*)', section)
            section_title = section_match.group(1).strip() if section_match else "Overview"

            # Clean header text from chunk body if desired, or keep intact for context
            chunk_id = f"{doc_name}#chunk_{chunk_idx}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": section,
                "metadata": {
                    "document_name": doc_name,
                    "section_title": section_title,
                    "chunk_id": chunk_id,
                    "char_count": len(section)
                }
            })
            chunk_idx += 1

        return chunks

    def process_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """Loads and processes all markdown files in a directory."""
        all_chunks = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith((".md", ".txt")):
                    file_path = os.path.join(root, file)
                    all_chunks.extend(self.process_file(file_path))
        return all_chunks