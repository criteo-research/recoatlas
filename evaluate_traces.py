import argparse
import json
from pathlib import Path
from artifacts import resolve, load_dataset, CATEGORIES
from evaluation.metrics import summarize


def main():
    p=argparse.ArgumentParser(description='Evaluate fractional SetHit@K; missing reports score zero.')
    p.add_argument('--category',choices=CATEGORIES,required=True)
    p.add_argument('--artifacts',type=Path)
    p.add_argument('--partition',default='agent_hidden',choices=['shared_train','agent_dev','agent_hidden'])
    p.add_argument('--traces',type=Path,required=True)
    a=p.parse_args()
    root,_=resolve(a.category,a.artifacts,required=['data/product_texts.json',f'queries/{a.partition}.jsonl'])
    rows=[json.loads(s) for s in a.traces.read_text().splitlines() if s.strip()]
    catalog=json.loads((root/'data/product_texts.json').read_text())
    print(json.dumps(summarize(load_dataset(root,a.partition),rows,catalog),indent=2))


if __name__=='__main__': main()
