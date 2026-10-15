"""Canonical paper fetching and auditable public author-code retrieval."""
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent
os.environ['RESEARCH_PAPERS_CACHE'] = str(ROOT / 'fulltext')
HELPER = Path('C:/Users/Administrator/.codex/skills/research-papers/scripts/fetch_and_parse.py')
spec = importlib.util.spec_from_file_location('research_fetcher_deep', HELPER)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
original = module.parse_pdf_to_markdown

def parser(pdf_path, out_path):
    result = original(pdf_path, out_path)
    if result[0]:
        return result
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        text = ''.join(f'\n\n<!-- PDF PAGE {i+1} -->\n\n' +
                       (p.extract_text(extraction_mode='layout') or '')
                       for i,p in enumerate(reader.pages))
        if len(text.strip()) < 100:
            return result
        Path(out_path).write_text(text,encoding='utf-8')
        return True,'pypdf-layout fallback',[],False
    except Exception:
        return result

module.parse_pdf_to_markdown = parser
if __name__ == '__main__':
    module.main()
