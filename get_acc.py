import json
import re

BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}", re.IGNORECASE)

fname = '/mnt/disk/coconut/checkpoints/qwenmusr-coconut-v3/eval_outputs.json'


def extract_last_boxed(text: str) -> str:
    matches = list(BOXED_RE.finditer(text or ""))
    if matches:
        return matches[-1].group(1).strip()
    return (text or "").strip()


def parse_answer_letter(raw_text: str) -> str:
    content = extract_last_boxed(raw_text)
    content = (content or "").strip().upper()
    if len(content) == 1 and content in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return content
    toks = [t for t in re.split(r"\s+|[,;]", content) if t]
    if toks and len(toks[-1]) == 1 and toks[-1] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return toks[-1]
    return ""

with open(fname, 'r') as f:
    d = f.read()

data = json.loads(d)

outputs = data['outputs']

scores = []
for output in outputs:
    ans = output['ground_truth_answer']
    cot = output['extracted_cot']
    pred = parse_answer_letter(cot)
    ans = parse_answer_letter(ans)
    print(pred, ans, pred == ans)
    scores.append(pred == ans)
print(f"len(scores): {len(scores)}")
print(f"acc: {sum(scores)/len(scores)}")
