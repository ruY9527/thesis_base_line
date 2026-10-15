"""Invoke the canonical research fetcher with the bundled pypdf fallback.

Retrieval, cache, read plan, and sentinel remain owned by the skill helper.
The built-in runtime lacks the helper's optional PDF parsers.
"""
import importlib.util
from pathlib import Path
import sys
from pypdf import PdfReader

HELPER = Path(r'C:\Users\Administrator\.codex\skills\research-papers\scripts\fetch_and_parse.py')
spec = importlib.util.spec_from_file_location('research_fetcher', HELPER)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

original_parser = module.parse_pdf_to_markdown

def parser(pdf_path, out_path):
    result = original_parser(pdf_path, out_path)
    if result[0]:
        return result
    try:
        reader = PdfReader(str(pdf_path))
        pages = [f'\n\n<!-- PDF PAGE {i + 1} -->\n\n' +
                 (page.extract_text(extraction_mode='layout') or '')
                 for i, page in enumerate(reader.pages)]
        text = ''.join(pages)
        if len(text.strip()) < 100:
            return result
        Path(out_path).write_text(text, encoding='utf-8')
        return True, 'pypdf-layout (bundled fallback)', [], False
    except Exception as error:
        print(f'pypdf fallback failed: {error}', file=sys.stderr)
        return result

module.parse_pdf_to_markdown = parser
module.main()
