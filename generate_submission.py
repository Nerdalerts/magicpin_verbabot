import os
import sys
import time
import json
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from bot import compose

def main():
    base_dir = Path("dataset/expanded")
    with open(base_dir / "test_pairs.json", "r", encoding="utf-8") as f:
        pairs = json.load(f)["pairs"]

    submissions = []
    print(f"Generating compositions for {len(pairs)} test pairs...")
    for p in pairs:
        tid = p["test_id"]
        trg_file = next(base_dir.glob(f"triggers/{p['trigger_id']}.json"))
        trg = json.load(open(trg_file, encoding="utf-8"))
        
        m_file = next(base_dir.glob(f"merchants/{p['merchant_id']}.json"))
        merchant = json.load(open(m_file, encoding="utf-8"))
        
        cat_slug = merchant["category_slug"]
        cat_file = next(base_dir.glob(f"categories/{cat_slug}.json"))
        category = json.load(open(cat_file, encoding="utf-8"))
        
        customer = None
        if p.get("customer_id"):
            c_file = next(base_dir.glob(f"customers/{p['customer_id']}.json"))
            customer = json.load(open(c_file, encoding="utf-8"))
            
        result = compose(category, merchant, trg, customer)
        row = {
            "test_id": tid,
            "body": result["body"],
            "cta": result["cta"],
            "send_as": result["send_as"],
            "suppression_key": result["suppression_key"],
            "rationale": result["rationale"]
        }
        submissions.append(row)
        print(f"  [{tid}] -> {result['body'][:60]}...")
        # Brief sleep to stay within free-tier rate limits
        time.sleep(1.0)

    with open("submission.jsonl", "w", encoding="utf-8") as f:
        for s in submissions:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nDone! Successfully written {len(submissions)} rows to submission.jsonl.")

if __name__ == "__main__":
    main()
