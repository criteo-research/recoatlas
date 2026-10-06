"""Advisory near-duplicate tool extension."""
import json

INSTRUCTIONS = """Diversity / near-duplicate check:
Before final selection, use get_substitute_products on your observed candidate shortlist,
ordered from highest to lowest priority, with similarity_threshold=0.95. Reserve a tool
round for this check within the existing budget. It compares only products in the same
subcategory and returns kept items and removed near-duplicates. Treat this as advisory
evidence: prefer kept alternatives for redundant roles, but do not sacrifice explicit
user requirements merely to increase variety. It does not establish compatibility or
prove that products are interchangeable. If pruning leaves an uncovered role, select
another observed suitable candidate or search again if budget remains. Return exactly
the requested number of items. Never invent product IDs. The check is not forced by
the harness; actual tool use is recorded separately."""


def extend_agent(base_class, diversity_tool):
    class DiversityBundleSpecificAgent(base_class):
        def __init__(self, retrieval, comp, _unused=None, **kwargs):
            super().__init__(retrieval, comp, diversity_tool, **kwargs)
            self.diversity_checks = []

        def _system_prompt(self):
            return super()._system_prompt() + '\n\n' + INSTRUCTIONS

        def _execute_action(self, action, args, candidate_ids):
            if action != 'get_substitute_products':
                return super()._execute_action(action, args, candidate_ids)
            ids = args.get('item_ids') if isinstance(args, dict) else None
            if (not isinstance(ids, list) or not 1 <= len(ids) <= 60
                    or any(not isinstance(i, str) or i not in candidate_ids for i in ids)
                    or len(set(ids)) != len(ids)):
                return {'error': 'Use 1–60 unique, previously observed candidate IDs.'}, candidate_ids
            if args.get('similarity_threshold', 0.95) != 0.95:
                return {'error': 'This evaluation fixes similarity_threshold at 0.95.'}, candidate_ids
            result = json.loads(self.substitutability.get_substitutes(ids, similarity_threshold=0.95))
            self.diversity_checks.append({
                'input_ids': ids,
                'kept_ids': [r['item_id'] for r in result['kept']],
                'removed': [{k: r[k] for k in ('item_id', 'duplicate_of', 'similarity')}
                            for r in result['removed']],
                'similarity_threshold': 0.95,
            })
            # Do not silently erase unchecked candidates or override the LLM's selection.
            return result, candidate_ids

        def _state_block(self, **kwargs):
            state = json.loads(super()._state_block(**kwargs))
            state['diversity_checks'] = self.diversity_checks
            return json.dumps(state)

        def _finalize_report(self, *args, **kwargs):
            # The original forced finalizer sees only the candidate ledger; retain
            # deduplication evidence even when the final tool round was exhausted.
            strategy = self.optimized_strategy
            self.optimized_strategy += ('\n\n' + INSTRUCTIONS + '\nCompleted diversity checks:\n'
                                        + json.dumps(self.diversity_checks))
            try:
                return super()._finalize_report(*args, **kwargs)
            finally:
                self.optimized_strategy = strategy

        def protocol_stats(self, trace):
            return {**super().protocol_stats(trace),
                    'diversity_calls': len(self.diversity_checks),
                    'diversity_check_performed': bool(self.diversity_checks)}

    return DiversityBundleSpecificAgent
