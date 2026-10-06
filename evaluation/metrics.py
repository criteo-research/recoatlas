"""Fractional reference recall, with explicit missing-report accounting."""
import math


def sethit(predictions, references, catalog=None):
    gold = set(references)
    if not gold:
        raise ValueError('Empty reference set')
    # Limit positions BEFORE removing duplicate/invalid IDs.
    positions = predictions[:len(gold)]
    valid = {x for x in positions if isinstance(x, str) and (catalog is None or x in catalog)}
    return len(valid & gold)/len(gold)


def summarize(tasks, predictions, catalog=None):
    by_id = {r['candidate_id']: r for r in predictions}
    if len(by_id) != len(predictions):
        raise ValueError('Duplicate predictions; select one attempt per query explicitly')
    if set(by_id)-{t['candidate_id'] for t in tasks}:
        raise ValueError('Predictions contain unknown queries')
    values, completed = [], []
    for task in tasks:
        row = by_id.get(task['candidate_id'], {})
        score = sethit(row.get('predicted_item_ids', []), task['positive_items'], catalog) if row.get('status')=='complete' else 0.0
        values.append(score)
        if row.get('status')=='complete': completed.append(score)
    return {'expected_queries': len(tasks), 'complete_queries': len(completed),
            'missing_or_failed_queries': len(tasks)-len(completed),
            'sethit_all_expected': sum(values)/len(values) if values else None,
            'sethit_completed_only': sum(completed)/len(completed) if completed else None}


def paired_bootstrap(left, right, seed=20260925, resamples=10000):
    import numpy as np
    if set(left) != set(right) or not left:
        raise ValueError('Pairing requires the same nonempty query IDs')
    delta = np.array([left[i]-right[i] for i in sorted(left)], dtype=float)
    if not np.isfinite(delta).all(): raise ValueError('Non-finite score')
    rng = np.random.default_rng(seed)
    draws = []
    for offset in range(0, resamples, 250):
        draws.extend(rng.choice(delta, (min(250,resamples-offset),len(delta))).mean(axis=1))
    return {'n':len(delta), 'mean_difference':float(delta.mean()),
            'ci95':np.quantile(draws,[.025,.975]).tolist(), 'resamples':resamples, 'seed':seed}
