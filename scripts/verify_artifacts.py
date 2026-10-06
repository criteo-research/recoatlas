"""Validate one category's checksums, catalog alignment, and reference membership."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artifacts import CATEGORIES, resolve, load_dataset, source_config


def verify(category, artifacts=None, metadata_only=False):
    metadata = source_config()['categories'][category]
    required = [rel for rel in metadata['files'] if not (metadata_only and rel.endswith('.pt'))]
    root, metadata = resolve(category, artifacts, required=required)
    catalog = json.loads((root / 'data/product_texts.json').read_text())
    sub = json.loads((root / 'data/subcategories.json').read_text())
    if len(catalog) != metadata['catalog_items'] or any(not isinstance(v, str) or not v for v in catalog.values()):
        raise ValueError('Catalog size or text mismatch')
    if not set(sub['item_to_subcategory']) <= set(catalog):
        raise ValueError('Subcategories contain out-of-catalog items')
    inverse = {key: set(ids) for key, ids in sub['subcategory_to_items'].items()}
    for item, category_name in sub['item_to_subcategory'].items():
        if item not in inverse.get(category_name, set()):
            raise ValueError('Subcategory maps disagree')
    for partition, count in metadata['query_counts'].items():
        rows = load_dataset(root, partition)
        if len(rows) != count or any(not set(r['positive_items']) <= catalog.keys() for r in rows):
            raise ValueError('Query count or reference membership mismatch')
    if category == 'hm-clothing' and any(len(i) != 10 or not i.isdigit() for i in catalog):
        raise ValueError('H&M IDs must be zero-padded ten-digit strings')
    if not metadata_only:
        import torch
        base = torch.load(root / 'embeddings/item_embeddings_qwen.pt', map_location='cpu', weights_only=True)
        comp = torch.load(root / 'embeddings/comp_embeddings_qwen.pt', map_location='cpu', weights_only=True)
        if list(base['item_ids']) != list(comp['item_ids']) or list(base['item_ids']) != list(catalog):
            raise ValueError('Embedding item order mismatch')
        for key, data in [('item_embs', base), ('anchor_embs', comp), ('complement_embs', comp)]:
            tensor = data[key]
            if tensor.ndim != 2 or tensor.shape[0] != len(catalog) or not torch.isfinite(tensor).all():
                raise ValueError('Invalid embedding array: ' + key)
            if not torch.allclose(tensor.float().norm(dim=1), torch.ones(len(catalog)), atol=1e-4):
                raise ValueError('Unnormalized embedding array: ' + key)
        checkpoint = torch.load(root / 'models/recsys_model_qwen.pt', map_location='cpu', weights_only=True)
        if not checkpoint.get('adapter_state_dict') or not checkpoint.get('model_config'):
            raise ValueError('Missing retrieval adapter/configuration')
    return {'category': category, 'status': 'verified', 'catalog_items': len(catalog),
            'query_counts': metadata['query_counts'], 'tensors_checked': not metadata_only,
            'items_without_subcategory': len(catalog.keys() - sub['item_to_subcategory'].keys())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--category', choices=CATEGORIES, required=True)
    parser.add_argument('--artifacts', type=Path)
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.category, args.artifacts, args.metadata_only), indent=2))


if __name__ == '__main__':
    main()
