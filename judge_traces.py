"""Blind whole-set direct-relevance judging with the released fixed rubric."""
import argparse
import hashlib
import json
import random
from pathlib import Path
from artifacts import CATEGORIES, resolve, load_dataset, sha256, SOURCES
from evaluation.judge_validation import validate

CONFIG = Path(__file__).parent/'prompts/direct_relevance.json'


def make_request(task, predictions, catalog, config):
    if len(predictions)!=len(task['positive_items']) or len(set(predictions))!=len(predictions):
        raise ValueError('Judge requires a complete unique report')
    if any(i not in catalog for i in predictions): raise ValueError('Unknown product')
    ids=list(predictions)
    seed=hashlib.sha256(f"{config['seed']}|{task['candidate_id']}".encode()).hexdigest()
    random.Random(int(seed,16)).shuffle(ids)
    texts={f'P{i+1}':catalog[x][:297]+'...' if len(catalog[x])>300 else catalog[x] for i,x in enumerate(ids)}
    data={'request':task['query'],'requested_item_count':len(task['positive_items']),'catalog_evidence':texts}
    prompt=config['common']+'\n'+config['rubric']+'\n\nINPUT DATA:\n'+json.dumps(data,ensure_ascii=False)
    return [{'role':'system','content':config['system']},{'role':'user','content':prompt}],list(texts)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--category',choices=CATEGORIES,required=True)
    p.add_argument('--artifacts',type=Path)
    p.add_argument('--partition',default='agent_hidden',choices=['agent_dev','agent_hidden'])
    p.add_argument('--traces',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--model',help='Your accessible judge model/deployment; see documented paper deployment')
    p.add_argument('--base-url')
    p.add_argument('--dry-run',action='store_true',help='Validate requests and print counts without calling any API')
    a=p.parse_args(argv)
    if not a.dry_run and (not a.output or not a.model): p.error('--model and --output are required unless using --dry-run')
    if a.output and (a.output.exists() or a.output.with_suffix('.summary.json').exists()): p.error('Use a new output path; existing judgments are never overwritten')
    root,_=resolve(a.category,a.artifacts,required=['data/product_texts.json',f'queries/{a.partition}.jsonl'])
    tasks={r['candidate_id']:r for r in load_dataset(root,a.partition)}
    catalog=json.loads((root/'data/product_texts.json').read_text())
    config=json.loads(CONFIG.read_text())
    rows=[json.loads(s) for s in a.traces.read_text().splitlines() if s.strip()]
    if len({r['candidate_id'] for r in rows})!=len(rows): p.error('Duplicate prediction IDs')
    # Validate all inputs before any paid request. Agent explanations and tool scores never enter the request.
    requests=[]
    for row in rows:
        if row['candidate_id'] not in tasks: p.error('Traces contain a query outside the selected partition')
        if row['status'] not in ('complete','error'): p.error('Unknown trace status')
        if row['status']=='complete':
            messages,labels=make_request(tasks[row['candidate_id']],row['predicted_item_ids'],catalog,config)
            requests.append((row['candidate_id'],messages,labels))
    if a.dry_run:
        print(json.dumps({'requests':len(requests),'skipped_error_reports':len(rows)-len(requests),
                          'expected_partition_queries':len(tasks),'api_calls':0},indent=2))
        return 0
    from llm_client import get_sync_client
    client=get_sync_client(base_url=a.base_url)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    good=[]
    with a.output.open('x') as stream:
        for cid,messages,labels in requests:
            for attempt in range(3):
                result={'candidate_id':cid,'attempt':attempt+1,'status':'error'}
                try:
                    response=client.chat.completions.create(model=a.model,messages=messages,
                        temperature=config['temperature'],max_completion_tokens=config['max_completion_tokens'],
                        response_format={'type':'json_object'})
                    if response.choices[0].finish_reason!='stop': raise ValueError('Incomplete judgment')
                    payload=json.loads(response.choices[0].message.content)
                    validate(payload,labels)
                    result.update(status='complete',payload=payload,quality=payload['score']/4)
                    good.append(result['quality'])
                except Exception as error:
                    result['error_type']=type(error).__name__
                stream.write(json.dumps(result)+'\n'); stream.flush()
                if result['status']=='complete': break
    summary={'expected_reports':len(requests),'valid_judgments':len(good),'unresolved':len(requests)-len(good),
             'category':a.category,'partition':a.partition,'expected_partition_queries':len(tasks),
             'input_reports':len(rows),'skipped_error_reports':len(rows)-len(requests),
             'traces_sha256':sha256(a.traces),'artifact_sources_sha256':sha256(SOURCES),
             'mean_direct_relevance':sum(good)/len(good) if good else None,
             'requested_model':a.model,'paper_deployment':config['model'],
             'same_deployment_name':a.model==config['model'],
             'prompt_sha256':hashlib.sha256(CONFIG.read_bytes()).hexdigest()}
    a.output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    return 0 if len(good)==len(requests) else 1


if __name__=='__main__': raise SystemExit(main())
