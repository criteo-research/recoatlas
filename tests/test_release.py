import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from evaluation.metrics import sethit, summarize, paired_bootstrap
from evaluation.judge_validation import validate
from judge_traces import make_request, CONFIG
from bundle_output_contract import validate_contract


def test_fractional_metric_and_positions():
    assert sethit(['a'],['a','b','c','d'])==.25
    assert sethit(['a','a','a','a','b'],['a','b','c','d'])==.25
    assert sethit(['bad','a','b'],['a','b'],{'a','b'})==.5
    tasks=[{'candidate_id':'one','positive_items':['a','b','c','d']},
           {'candidate_id':'two','positive_items':['a','b','c','d']}]
    result=summarize(tasks,[{'candidate_id':'one','status':'complete','predicted_item_ids':['a']}])
    assert result['sethit_all_expected']==.125
    assert result['sethit_completed_only']==.25
    assert result['missing_or_failed_queries']==1


def test_judge_is_blind_and_deterministic():
    config=json.loads(CONFIG.read_text())
    task={'candidate_id':'test','query':'Request for instruments','positive_items':['hidden1','hidden2','hidden3'],
          'purchase_history':['secret'], 'tool':'utility', 'explanation':'not judge evidence'}
    catalog={x:x*400 for x in ['a','b','c']}
    messages,labels=make_request(task,['a','b','c'],catalog,config)
    assert make_request(task,['a','b','c'],catalog,config)==(messages,labels)
    raw=json.dumps(messages)
    assert all(s not in raw for s in ['hidden1','secret','utility','not judge evidence'])
    data=json.loads(messages[-1]['content'].split('INPUT DATA:\n')[1])
    assert set(data)=={'request','requested_item_count','catalog_evidence'}
    assert all(len(s)==300 for s in data['catalog_evidence'].values())


@pytest.mark.parametrize('score',[True,4.1,5,-1,'4'])
def test_bad_judgment(score):
    with pytest.raises(ValueError):validate({'score':score,'reason':'fit','evidence_items':['P1']},['P1'])


def test_schema_diversity():
    d={'thought':'check','step_goal':'duplicates','action':'get_substitute_products',
       'action_input':{'item_ids':['a','b'],'similarity_threshold':.95}}
    assert validate_contract(json.dumps(d),3)==d
    d['action_input']['similarity_threshold']=.5
    with pytest.raises(ValueError):validate_contract(json.dumps(d),3)


class Retrieval:
    product_reps={x:'Title: '+x for x in ['a','b','c','d']}
    def search(self,query,top_k):
        return json.dumps([{'item_id':x,'score':1.,'description':x} for x in self.product_reps])


class Comp:
    def get_complementary(self,ids,top_k):
        return json.dumps([{'item_id':'d','score':1.,'description':'d'}])


class Diversity:
    def get_substitutes(self,ids,similarity_threshold):
        return json.dumps({'kept':[{'item_id':x} for x in ids], 'removed':[]})


class Backend:
    def __init__(self):self.calls=[]
    def complete(self,**kwargs):
        self.calls.append(kwargs)
        responses=[
            {'thought':'first','step_goal':'search','action':'search_products','action_input':{'query':'guitar','top_k':10}},
            {'thought':'second','step_goal':'search','action':'search_products','action_input':{'query':'microphone','top_k':10}},
            {'thought':'expand','step_goal':'comp','action':'get_complementary_products','action_input':{'item_ids':['a'],'top_k':10}},
            {'thought':'check','step_goal':'duplicates','action':'get_substitute_products','action_input':{'item_ids':['a','b','c'],'similarity_threshold':.95}},
            {'report_explanation':'a useful set','results':[{'product_id':x,'reasoning':'fits'} for x in ['a','b','c']]}]
        return json.dumps(responses[min(len(self.calls)-1,4)])


def test_full_agent_fake_endpoint():
    from agents.bundle import build_optimized_agent_class
    from agents.diversity import extend_agent
    backend=Backend()
    agent=extend_agent(build_optimized_agent_class(),Diversity())(Retrieval(),Comp(),None,
        optimized_strategy=Path('prompts/strategy.txt').read_text(),backend=backend,top_k=3,
        max_tool_rounds=6,use_harness_actions=False,planning=False,min_distinct_searches=2,min_comp_calls=1)
    report=agent.chat_report('I want to record a guitar performance.')
    assert [r['product_id'] for r in report['results']]==['a','b','c']
    assert agent.protocol_stats(report['trace'])['diversity_check_performed']
    assert len(report['trace'])<=6


def test_bootstrap_pairing():
    with pytest.raises(ValueError):paired_bootstrap({'a':1},{'b':1})
    assert paired_bootstrap({'a':1,'b':.5},{'a':1,'b':.5},resamples=100)['ci95']==[0,0]


def test_tools_tiny_catalog(tmp_path):
    import torch
    from tools.complementarity import ComplementarityTool
    from tools.substitutability import SubstitutabilityTool
    texts={x:'Title: '+x for x in ['a','b','c','d']}
    sub={'item_to_subcategory':{'a':'x','b':'x','c':'y','d':'z'},'subcategory_to_items':{'x':['a','b'],'y':['c'],'z':['d']}}
    (tmp_path/'texts.json').write_text(json.dumps(texts))
    (tmp_path/'sub.json').write_text(json.dumps(sub))
    torch.save({'item_ids':list(texts),'item_embs':torch.tensor([[1.,0],[1.,0],[.8,.6],[0.,1.]])},tmp_path/'emb.pt')
    tool=ComplementarityTool(str(tmp_path/'texts.json'),str(tmp_path/'emb.pt'),str(tmp_path/'sub.json'),trained=False,device='cpu')
    assert 'b' not in [r['item_id'] for r in json.loads(tool.get_complementary(['a'],3))]
    dup=SubstitutabilityTool(str(tmp_path/'texts.json'),str(tmp_path/'sub.json'),str(tmp_path/'emb.pt'),device='cpu')
    result=json.loads(dup.get_substitutes(['a','b','c'],similarity_threshold=.95))
    assert [r['item_id'] for r in result['kept']]==['a','c']


def test_query_encoder_import():
    from tools.query_encoder import QueryRetrievalModel
    assert callable(QueryRetrievalModel)
