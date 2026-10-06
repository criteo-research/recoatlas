"""Run a fixed bundle agent or deterministic baseline on released queries."""
import argparse
import json
import os
import sys
from pathlib import Path
from artifacts import CATEGORIES, resolve, load_dataset, sha256, SOURCES
from evaluation.metrics import summarize, sethit

PAPER_CONFIG = Path(__file__).parent / 'configs/paper_agent.json'


def main(argv=None):
    settings = json.loads(PAPER_CONFIG.read_text())
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--category', choices=CATEGORIES, default='amazon-musical-instruments')
    p.add_argument('--artifacts', type=Path)
    p.add_argument('--partition', choices=['shared_train','agent_dev','agent_hidden'], default='agent_hidden')
    p.add_argument('--tools', choices=['utility','semantic'], default='utility')
    p.add_argument('--agent', choices=['llm','non-llm'], default='llm')
    p.add_argument('--model', default=settings['model'])
    p.add_argument('--base-url', default=os.getenv('OPENAI_BASE_URL'))
    p.add_argument('--thinking', choices=['on','off'], default='off')
    p.add_argument('--max-output-tokens', type=int)
    p.add_argument('--thinking-token-budget', type=int)
    p.add_argument('--grammar', action=argparse.BooleanOptionalAction, default=True,
                   help='Paper Qwen output contract; requires a vLLM-compatible grammar endpoint and xgrammar. Use --no-grammar for other endpoints.')
    p.add_argument('--device', default='cpu')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    if a.limit < 0: p.error('--limit must be nonnegative')
    if a.output.exists() or a.output.with_suffix('.summary.json').exists(): p.error('Output exists; use a new output path')
    if a.max_output_tokens is not None and a.max_output_tokens <= 0: p.error('Token limit must be positive')
    if a.thinking_token_budget is not None and a.thinking_token_budget <= 0: p.error('Thinking budget must be positive')
    if a.agent=='llm' and a.thinking=='on' and not a.base_url: p.error('Thinking requires --base-url for a Qwen-compatible server')
    if a.agent=='llm' and a.grammar:
        if not a.base_url: p.error('Grammar requires --base-url; use --no-grammar for other endpoints')
        try:
            import xgrammar
        except ImportError:
            p.error('Install requirements-grammar.txt or explicitly use --no-grammar')
    required = ['data/product_texts.json', 'data/subcategories.json', 'models/recsys_model_qwen.pt',
                'embeddings/item_embeddings_qwen.pt', f'queries/{a.partition}.jsonl']
    if a.tools == 'utility': required.append('embeddings/comp_embeddings_qwen.pt')
    try:
        category, meta = resolve(a.category,a.artifacts,required=required)
    except ValueError as exc:
        p.error(str(exc))
    tasks = load_dataset(category,a.partition)
    if a.limit: tasks = tasks[:a.limit]
    from tools.load import build_tools
    retrieval, comp, diversity = build_tools(category,a.tools,a.device)
    sub = json.loads((category/'data/subcategories.json').read_text())['item_to_subcategory']
    strategy = (Path(__file__).parent/'prompts/strategy.txt').read_text().strip()
    if a.agent == 'llm':
        from llm_client import get_sync_client
        from backend import ComparableCountingBackend
        from agents.bundle import build_optimized_agent_class
        from agents.diversity import extend_agent
        backend = ComparableCountingBackend(get_sync_client(base_url=a.base_url), 'anonymous',
            max_output_tokens=a.max_output_tokens or settings[f'thinking_{a.thinking}_max_output_tokens'],
            temperature=settings['temperature'], local=bool(a.base_url), thinking=a.thinking,
            thinking_token_budget=a.thinking_token_budget or (settings['thinking_token_budget'] if a.thinking=='on' else None),
            capture_model_traces=True, strict_bundle_json=a.grammar)
        Agent = extend_agent(build_optimized_agent_class(),diversity)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    with a.output.open('x') as stream:
        for index, task in enumerate(tasks, 1):
            k=len(task['positive_items'])
            row={'candidate_id':task['candidate_id'], 'status':'error', 'predicted_item_ids':[]}
            try:
                if a.agent=='llm':
                    backend.reset(); backend.set_bundle_size(k)
                    agent=Agent(retrieval,comp,None, model=a.model,top_k=k,max_tool_rounds=settings['max_tool_rounds'],
                        backend=backend,use_harness_actions=False,planning=False,
                        query_timeout_seconds=settings['query_timeout_seconds'],temperature=settings['temperature'],optimized_strategy=strategy,
                        ledger_limit=settings['ledger_limit'],min_distinct_searches=settings['min_distinct_searches'],min_comp_calls=settings['min_comp_calls'])
                    report=agent.chat_report(task['query'])
                    row.update(report=report,protocol=agent.protocol_stats(report.get('trace',[])),api=backend.stats())
                    ids=[r['product_id'] for r in report['results']]
                elif a.agent=='non-llm':
                    from baselines import three_tool_baseline
                    ids=three_tool_baseline(task['query'],k,retrieval,comp,diversity,sub)
                row.update(status='complete',predicted_item_ids=ids)
            except Exception as error:
                # Avoid embedding endpoint addresses, local paths, or credentials in traces.
                row['error_type']=type(error).__name__
            row['feedback']={'sethit_at_k':sethit(row['predicted_item_ids'],task['positive_items'],retrieval.product_reps)
                             if row['status']=='complete' else 0.0}
            stream.write(json.dumps(row)+'\n'); stream.flush(); rows.append(row)
            print(f"{index}/{len(tasks)}: {row['status']}", file=sys.stderr, flush=True)
    summary=summarize(tasks,rows,retrieval.product_reps)
    summary.update(category=a.category, partition=a.partition, tools=a.tools, agent=a.agent,
                   model=a.model if a.agent=='llm' else None, thinking=a.thinking,
                   max_tool_rounds=settings['max_tool_rounds'], strategy_sha256=sha256(Path(__file__).parent/'prompts/strategy.txt'),
                   paper_config_sha256=sha256(PAPER_CONFIG), artifact_sources_sha256=sha256(SOURCES),
                   dataset_sha256=sha256(category/'queries'/f'{a.partition}.jsonl'),
                   limit=a.limit, grammar=a.grammar if a.agent=='llm' else None,
                   max_output_tokens=backend.max_output_tokens if a.agent=='llm' else None,
                   thinking_token_budget=backend.thinking_token_budget if a.agent=='llm' else None,
                   temperature=settings['temperature'] if a.agent=='llm' else None)
    a.output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    return 0 if summary['missing_or_failed_queries']==0 else 1


if __name__=='__main__': raise SystemExit(main())
