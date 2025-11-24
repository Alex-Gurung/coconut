"""
Edit FILES with the eval_outputs.json paths you want to concatenate, then run:
    python combine_eval_outputs.py
Writes a single dictionary with all outputs merged into one list.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union

# Update this list with your eval_outputs.json files
FILES = [
    # "/path/to/run1/eval_outputs.json",
    # "/path/to/run2/eval_outputs.json",
    "check14_test4_eval_outputs.json",
    "check14_test3_eval_outputs.json",
    "check14_test2_eval_outputs.json",
    "check14_test1_eval_outputs.json",
    "check14_test0_eval_outputs.json",
]

# Where to write the combined outputs
# OUTPUT = "combined_eval_outputs.json"
OUTPUT = "combined_check14_test_eval_outputs.json"


def read_runs(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        data: Union[Dict[str, Any], List[Dict[str, Any]]] = json.load(f)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported JSON structure in {path}")


combined: Dict[str, Any] = {"combined_from": [], "outputs": []}

for input_path in FILES:
    path = Path(input_path)
    runs = read_runs(path)

    for run in runs:
        outputs = run.get("outputs")
        if not isinstance(outputs, list):
            raise ValueError(f"'outputs' missing or not a list in {path}")

        # Remember where this chunk came from
        combined["combined_from"].append(
            {
                "file": str(path),
                "checkpoint": run.get("checkpoint"),
                "name": run.get("config", {}).get("name")
                if isinstance(run.get("config"), dict)
                else None,
            }
        )

        combined["outputs"].extend(outputs)

out_path = Path(OUTPUT)
out_path.write_text(json.dumps(combined, indent=2))
print(f"Wrote {len(combined['outputs'])} entries to {out_path}")
