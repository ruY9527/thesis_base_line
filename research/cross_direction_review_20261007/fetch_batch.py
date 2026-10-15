"""Fetch official full texts through the research-papers canonical helper.

Only a PDF parser fallback is added. Retrieval and reading sentinels remain
owned by the helper. All writes stay in this research pack.
"""
from pathlib import Path
import concurrent.futures
from dataclasses import asdict
import importlib.util
import json
import os
import sys

ROOT = Path(__file__).resolve().parent
os.environ['RESEARCH_PAPERS_CACHE'] = str(ROOT / 'fulltext')
HELPER = Path('C:/Users/Administrator/.codex/skills/research-papers/scripts/fetch_and_parse.py')
spec = importlib.util.spec_from_file_location('cross_direction_fetcher', HELPER)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
original = module.parse_pdf_to_markdown

def parser(pdf_path, out_path):
    result = original(pdf_path, out_path)
    if result[0]:
        return result
    from pypdf import PdfReader
    try:
        reader = PdfReader(str(pdf_path))
        content = '\n\n'.join(f'<!-- PDF PAGE {i+1} -->\n\n' + (page.extract_text() or '') for i, page in enumerate(reader.pages))
        if len(content.strip()) < 100:
            return result
        Path(out_path).write_text(content, encoding='utf-8')
        return True, 'pypdf fallback; page markers retained', [], False
    except Exception as error:
        print(str(error), file=sys.stderr)
        return result

module.parse_pdf_to_markdown = parser

SOURCES = {
 'carl': 'https://proceedings.iclr.cc/paper_files/paper/2026/file/282967f8abaae52a452a97ee961410f3-Paper-Conference.pdf',
 'pino': 'https://proceedings.neurips.cc/paper_files/paper/2025/file/89af8e4eb696738f2c9e589522968a09-Paper-Conference.pdf',
 'diff_spectra': 'https://proceedings.neurips.cc/paper_files/paper/2025/file/c34ddd05eb089991f06f3c5dc36836e0-Paper-Conference.pdf',
 'hipandas': 'https://openaccess.thecvf.com/content/ICCV2025/papers/Xu_Hipandas_Hyperspectral_Image_Joint_Denoising_and_Super-Resolution_by_Image_Fusion_ICCV_2025_paper.pdf',
 'uafl': 'https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Enhancing_Unregistered_Hyperspectral_Image_Super-Resolution_via_Unmixing-based_Abundance_Fusion_Learning_CVPR_2026_paper.pdf',
 'umnet': 'https://ojs.aaai.org/index.php/AAAI/article/download/37988/41950',
 'unipan': 'https://openaccess.thecvf.com/content/ICCV2025/papers/Cui_Enpowering_Your_Pansharpening_Models_with_Generalizability_Unified_Distribution_is_All_ICCV_2025_paper.pdf',
 'srlf': 'https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_A_Selective_Re-learning_Mechanism_for_Hyperspectral_Fusion_Imaging_CVPR_2025_paper.pdf',
 'copgnet': '10.1016/j.inffus.2025.103997',
 'bugpan': '10.1016/j.inffus.2025.102938',
 'difiv': 'https://www.cedric-richard.fr/Articles/wang2023deep.pdf',
 'lightweight_unaligned': 'https://arxiv.org/pdf/2609.01060',
 'gsnr': 'https://openaccess.thecvf.com/content/CVPR2026/papers/Gualdron-Hurtado_GSNR_Graph_Smooth_Null-Space_Representation_for_Inverse_Problems_CVPR_2026_paper.pdf',
 'ambiguity_separation': 'https://arxiv.org/pdf/2605.15050',
 'uq_nullspace': 'https://arxiv.org/pdf/2304.06955',
 'specaware': 'https://arxiv.org/pdf/2510.27219',
 'task_decoupled': 'https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Task_Decoupled_Framework_for_Reference-Based_Super-Resolution_CVPR_2022_paper.pdf',
}

def fetch(entry):
    name, identifier = entry
    result = asdict(module.fetch_and_parse(identifier))
    record = {'name': name, 'identifier': identifier, **result}
    (ROOT / f'fetch-{name}.json').write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'name':name, **{k:result.get(k) for k in ['status','source','parsed_lines','parsed_chars','sentinel','tried']}}, ensure_ascii=False), flush=True)
    return record

if __name__ == '__main__':
    chosen = {k:v for k,v in SOURCES.items() if not sys.argv[1:] or k in sys.argv[1:]}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(fetch, chosen.items()))
    records = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(ROOT.glob('fetch-*.json'))]
    (ROOT / 'retrieval-manifest.json').write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
