"""Apply explicit model visual decisions; never infer a visual verdict."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter

p = argparse.ArgumentParser()
p.add_argument("decisions")
p.add_argument("reviews")
a = p.parse_args()
path = Path(a.decisions)
d = json.loads(path.read_text(encoding="utf-8"))
reviews = json.loads(Path(a.reviews).read_text(encoding="utf-8"))
assert d["output_language"] == reviews["output_language"] == "zh-CN"
for item in d["decisions"]:
    if item["source_id"] not in reviews["reviews"]:
        continue
    r = reviews["reviews"][item["source_id"]]
    sha = hashlib.sha256(Path(item["source_image_path"]).read_bytes()).hexdigest()
    prior = item.get("visual_review", {})
    item["visual_review"] = {
        "status": r["status"],
        "reviewed_asset_sha256": sha,
        "preserved_scientific_elements": r.get("preserved", []),
        "omitted_scientific_elements": r.get("omitted", []),
        "notes": r["notes"],
        "failure_reason": r.get("failure_reason", ""),
        "repair_attempts": prior.get("repair_attempts", 0),
        "revised_bbox": r.get("revised_bbox", prior.get("revised_bbox", [])),
    }
    item["source_image_sha256"] = sha
    if r["status"] == "pass":
        item["decision"] = "insert"
        item["skip_reason"] = ""
        item["materialization_status"] = "pending"
    elif r["status"] == "fail":
        item["decision"] = "visual_defect"
        item["skip_reason"] = r["failure_reason"]
    elif r["status"] == "repair_requested":
        item["decision"] = "review_pending"
    else:
        raise ValueError("Unsupported model review status")
    if "target_section" in r:
        item["target_section"] = r["target_section"]
d["summary"] = {"total_items": len(d["decisions"]), "by_decision": dict(Counter(x["decision"] for x in d["decisions"]))}
path.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(d["summary"], ensure_ascii=False))
