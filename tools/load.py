from pathlib import Path


def build_tools(root, mode='utility', device='cpu'):
    from .trained_retrieval import TrainedRetrievalTool, LockedComplementarityTool
    from .complementarity import ComplementarityTool
    from .substitutability import SubstitutabilityTool
    root = Path(root)
    if mode not in ('semantic', 'utility'):
        raise ValueError('Unknown tool mode')
    retrieval = TrainedRetrievalTool(root, device, root/'models/recsys_model_qwen.pt')
    if mode == 'semantic':
        retrieval.model.query_encoder.base_model.disable_adapter_layers()
        layers = [m for m in retrieval.model.query_encoder.modules() if hasattr(m, 'lora_A')]
        if not layers or not all(m.disable_adapters for m in layers):
            raise ValueError('Could not disable all query adapters')
    comp = ComplementarityTool(
        product_texts_path=str(root/'data/product_texts.json'),
        complement_embeddings_path=str(root/('embeddings/comp_embeddings_qwen.pt' if mode=='utility' else 'embeddings/item_embeddings_qwen.pt')),
        subcategories_path=str(root/'data/subcategories.json'), trained=mode=='utility', device=device)
    diversity = SubstitutabilityTool(
        product_texts_path=str(root/'data/product_texts.json'),
        embeddings_path=str(root/'embeddings/item_embeddings_qwen.pt'),
        subcategories_path=str(root/'data/subcategories.json'), device=device)
    if not (retrieval.all_items == comp.all_items == diversity.all_items):
        raise ValueError('Tool catalog order mismatch')
    return retrieval, LockedComplementarityTool(comp), diversity
