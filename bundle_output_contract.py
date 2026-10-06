"""Versioned, gold-blind output contract for the two-tool bundle agent."""
from functools import lru_cache
import hashlib
import json

VERSION = 'bundle-json-compact-v1'


def two_tool_schema(top_k, *, final_only=False):
    if type(top_k) is not int or not 1 <= top_k <= 7:
        raise ValueError('Bundle output contract supports target sizes 1 through 7')

    def string(limit):
        return {'type': 'string', 'minLength': 1, 'maxLength': limit}

    def obj(properties):
        return {'type': 'object', 'properties': properties,
                'required': list(properties), 'additionalProperties': False}

    product_id = {'type': 'string', 'pattern': '^[A-Za-z0-9_-]{1,64}$'}
    final = obj({'report_explanation': string(512), 'results': {
        'type': 'array', 'minItems': top_k, 'maxItems': top_k,
        'items': obj({'product_id': product_id, 'reasoning': string(256)})}})
    if final_only:
        return final
    branches = []
    for name, inputs in (
        ('search_products', {'query': string(512), 'top_k': {'type': 'integer', 'minimum': 1, 'maximum': 100}}),
        ('get_complementary_products', {'item_ids': {'type': 'array', 'minItems': 1, 'maxItems': 7, 'items': product_id},
                                        'top_k': {'type': 'integer', 'minimum': 1, 'maximum': 100}}),
    ):
        branches.append(obj({'thought': string(512), 'step_goal': string(256),
                             'action': {'const': name}, 'action_input': obj(inputs)}))
    return {'anyOf': branches + [final]}


@lru_cache(maxsize=14)
def compiled_contract(top_k, final_only=False):
    # Send explicit EBNF, since this vLLM version reads JSON whitespace options
    # from server config, not the similarly named request-level option.
    import xgrammar
    schema = bundle_schema(top_k, final_only=final_only)
    grammar = str(xgrammar.Grammar.from_json_schema(
        schema, any_whitespace=False, separators=(',', ':'), strict_mode=True))
    grammar = grammar.replace('"0.94999999999999996"', '"0.95"')
    return grammar, hashlib.sha256(grammar.encode()).hexdigest()


def validate_contract(content, top_k, *, final_only=False):
    """Defense in depth if a serving backend ignores the output grammar."""
    try:
        data = json.loads(content)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError('INVALID_BUNDLE_SCHEMA: response is not JSON') from error

    def matches(value, schema):
        import re
        if 'anyOf' in schema:
            return any(matches(value, branch) for branch in schema['anyOf'])
        if 'const' in schema:
            return value == schema['const']
        kind = schema['type']
        if kind == 'object':
            return (isinstance(value, dict) and set(value) == set(schema['required'])
                    and all(matches(value[k], s) for k, s in schema['properties'].items()))
        if kind == 'array':
            return (isinstance(value, list) and schema['minItems'] <= len(value) <= schema['maxItems']
                    and all(matches(v, schema['items']) for v in value))
        if kind == 'string':
            return (isinstance(value, str) and schema.get('minLength', 0) <= len(value) <= schema.get('maxLength', 64)
                    and ('pattern' not in schema or re.fullmatch(schema['pattern'], value) is not None))
        return type(value) is int and schema['minimum'] <= value <= schema['maximum']

    if not matches(data, bundle_schema(top_k, final_only=final_only)):
        raise ValueError('INVALID_BUNDLE_SCHEMA: response violates the bundle output contract')
    return data

import copy

def three_tool_schema(top_k, *, final_only=False):
    schema = copy.deepcopy(two_tool_schema(top_k, final_only=final_only))
    if not final_only:
        branch = copy.deepcopy(schema['anyOf'][1])
        branch['properties']['action'] = {'const': 'get_substitute_products'}
        inputs = branch['properties']['action_input']
        inputs['properties']['item_ids']['maxItems'] = 60
        inputs['properties'].pop('top_k')
        inputs['properties']['similarity_threshold'] = {'const': 0.95}
        inputs['required'] = ['item_ids', 'similarity_threshold']
        schema['anyOf'].insert(-1, branch)
    return schema

bundle_schema = three_tool_schema
