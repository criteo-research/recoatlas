def validate(payload, labels):
    if not isinstance(payload, dict) or set(payload) != {'score', 'reason', 'evidence_items'}:
        raise ValueError('schema')
    if type(payload['score']) is not int or not 0 <= payload['score'] <= 4:
        raise ValueError('score')
    if not isinstance(payload['reason'], str) or not payload['reason'].strip():
        raise ValueError('reason')
    if not isinstance(payload['evidence_items'], list) or any(
            not isinstance(x, str) or x not in labels for x in payload['evidence_items']):
        raise ValueError('evidence_items')
