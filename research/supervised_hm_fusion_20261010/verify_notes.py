"""Verify saved research notes and unchanged baseline files without executing models."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

reports = []
for key in ("emrdiff", "srlf"):
    run = ROOT / "deeppapernote_runs" / key
    saved = read(run / f"{key}_save.json")
    lint = read(run / f"{key}_final_lint.json")
    grounding = read(run / f"{key}_grounding_lint.json")
    quality = read(run / f"{key}_final_quality_review.json")
    readability = read(run / f"{key}_final_readability_review.json")
    decisions = read(run / f"{key}_figure_table_decisions.json")
    plan = read(run / f"{key}_note_plan.json")
    bundle = read(run / f"{key}_bundle.json")
    manifest = read(run / f"{key}_source_manifest.json")
    note = Path(saved["note_path"])
    sha = digest(note)
    checks = {
        "all_lint_gates": all(v for k, v in lint.items() if k.startswith("passes_")),
        "grounding_gate": grounding["passes_grounding"],
        "quality_review": quality["passes_quality_review"],
        "readability_review": readability["passes_readability_review"],
        "language_chain": all(x["output_language"] == "zh-CN" for x in [saved, lint, grounding, quality, readability, decisions, plan, bundle]) and bundle["writing_contract"]["language"] == "zh-CN",
        "note_and_review_sha_match": all(x["note_sha256"] == sha for x in [lint, quality, readability]),
        "formal_save_success": saved["status"] == "ok" and saved["output_mode"] == "workspace",
        "no_pending_figure_decisions": not any(x["decision"] == "review_pending" for x in decisions["decisions"]),
        "no_source_truncation": not manifest["pdf"]["text_truncated"],
        "materialized_asset_sha_match": all(digest(x["dest_image_path"]) == x["reviewed_asset_sha256"] for x in saved["materialized_figures"]),
    }
    text = note.read_text(encoding="utf-8")
    embeds = re.findall(r"!\[[^\]]*\]\((images/[^)]+)\)", text)
    checks["all_image_links_exist"] = all((note.parent / x).is_file() for x in embeds)
    checks["image_count_matches_save"] = len(embeds) == len(saved["materialized_figures"])
    retained = sum(x["decision"] in {"placeholder", "visual_defect"} for x in decisions["decisions"])
    checks["placeholder_count_matches"] = text.count("> [!figure]") == retained
    reports.append({
        "id": key,
        "title": saved["title"],
        "identifier": bundle["metadata"].get("doi") or bundle["metadata"]["source_url"],
        "note_path": str(note),
        "domain": note.parent.parent.name,
        "save_mode": saved["output_mode"],
        "note_sha256": sha,
        "warning_count": sum(len(x.get("warnings", [])) for x in [lint, grounding, quality, readability]),
        "materialized_count": len(embeds),
        "retained_placeholder_count": retained,
        "checks": checks,
        "passed": all(checks.values()),
    })

integrity = read(ROOT / "baseline-integrity.json")
for f in integrity["files"]:
    f["after_sha256"] = digest(f["file"]).upper()
    f["unchanged"] = f["after_sha256"] == f["before_sha256"]
integrity["all_unchanged"] = all(x["unchanged"] for x in integrity["files"])
integrity["last_verified"] = "2026-10-11"
(ROOT / "baseline-integrity.json").write_text(json.dumps(integrity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

missing_links = []
for md in [*ROOT.glob("*.md"), *(ROOT / "notes").rglob("*.md")]:
    for target in re.findall(r"\]\(([^)]+)\)", md.read_text(encoding="utf-8")):
        if re.match(r"^(https?://|#|[A-Za-z]:/)", target):
            continue
        if not (md.parent / target).exists():
            missing_links.append({"document": str(md), "target": target})

code = ROOT / "deeppapernote_runs" / "srlf" / "author_core_snapshot.txt"
code_sha = digest(code)
expected = "bcea9265778b8b86f22df1c41d89571bceac6ad1e1e798715393a8c77561ea7d"
result = {
    "verified_on": "2026-10-11",
    "literature_search_cutoff": "2026-10-10",
    "notes": reports,
    "baseline_all_unchanged": integrity["all_unchanged"],
    "missing_local_links": missing_links,
    "additional_author_code": {"url": "https://raw.githubusercontent.com/YuanyeLiu/SRLF-Net/main/deep_select_mamba.py", "snapshot_date": "2026-10-11", "path": str(code), "sha256": code_sha, "sha_match": code_sha == expected},
    "model_training_or_numerical_replication_performed": False,
    "scope": "Saved-note byte integrity, required skill gates, language, copied image identities, links and baseline integrity. Model reviews are model judgments; lint does not establish scientific novelty.",
}
result["passed"] = all(x["passed"] for x in reports) and integrity["all_unchanged"] and not missing_links and code_sha == expected
(ROOT / "notes-verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"passed": result["passed"], "notes": [{k:x[k] for k in ["id", "domain", "warning_count", "materialized_count", "retained_placeholder_count", "passed"]} for x in reports], "baseline_all_unchanged": result["baseline_all_unchanged"], "missing_links": missing_links}, ensure_ascii=False))
if not result["passed"]:
    raise SystemExit(1)
