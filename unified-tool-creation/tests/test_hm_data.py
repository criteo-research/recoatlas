import json

import pandas as pd

from tools_builder.benchmark_split import build_user_split_manifest
from tools_builder.hm_data import canonical_catalog, prepare_hm_inputs


def articles():
    return pd.DataFrame(
        [
            {
                "article_id": 123,
                "prod_name": "Shirt",
                "detail_desc": "A sufficiently detailed cotton shirt.",
                "product_type_name": "Shirt",
                "colour_group_name": "Blue",
                "graphical_appearance_name": "Solid",
                "index_name": "Ladieswear",
                "section_name": "Womens Everyday",
            },
            {
                "article_id": 456,
                "prod_name": "",
                "detail_desc": "This description is long enough.",
                "product_type_name": "Trousers",
                "colour_group_name": "Black",
                "graphical_appearance_name": "Solid",
                "index_name": "Ladieswear",
                "section_name": "Womens Everyday",
            },
        ]
    )


def test_catalog_filters_and_zero_padding():
    frame, texts, categories = canonical_catalog(articles())
    assert list(frame.item_id) == ["0000000123"]
    assert list(texts) == ["0000000123"]
    assert "Type: Shirt" in texts["0000000123"]
    assert categories == {"0000000123": "Shirt"}


def test_prepare_uses_shared_train_only(tmp_path):
    articles_path = tmp_path / "articles.parquet"
    articles().to_parquet(articles_path, index=False)
    transactions = pd.DataFrame(
        {
            "customer_id": ["train", "hidden"],
            "article_id": [123, 123],
            "t_dat": ["2020-01-01", "2020-01-02"],
            "price": [0.1, 0.2],
        }
    )
    transactions_path = tmp_path / "transactions.parquet"
    transactions.to_parquet(transactions_path, index=False)
    for index in range(100):
        manifest = build_user_split_manifest(
            ["train", "hidden"], salt=f"fixture-{index}", category="H_and_M"
        )
        if manifest["splits"]["shared_train"]:
            break
    else:
        raise AssertionError("Could not construct a shared_train fixture")
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(manifest))
    summary = prepare_hm_inputs(
        articles_path, transactions_path, split_path, tmp_path / "run"
    )
    output = pd.read_parquet(tmp_path / "run/data/interactions.parquet")
    assert set(output.user_id) == set(manifest["splits"]["shared_train"])
    assert set(output.parent_asin) == {"0000000123"}
    assert set(output.rating) == {5.0}
    assert summary["shared_train_purchases"] == len(output)
