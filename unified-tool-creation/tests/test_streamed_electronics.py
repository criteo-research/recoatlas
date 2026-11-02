import json

from tools_builder import streamed_electronics as adapter


def test_existing_date_semantics_and_review_membership(tmp_path, monkeypatch):
    def product(asin, date):
        return {
            "parent_asin": asin,
            "details": json.dumps({"Date First Available": date}),
            "price": "12.5",
            "description": ["A sufficiently detailed product description"],
            "title": "Test product",
            "main_category": "Electronics",
            "average_rating": 4.5,
            "rating_number": 3,
            "categories": ["Electronics", "Accessories"],
        }

    metadata = [
        product("old", "January 1, 2019"),
        product("keep", "January 1, 2020"),
        product("missing", "unknown"),
    ]
    reviews = [
        {
            "parent_asin": key,
            "user_id": "u",
            "timestamp": 1600000000000,
            "rating": 5,
            "title": "Title",
            "text": "A long enough English review",
        }
        for key in ("old", "keep", "missing")
    ]
    monkeypatch.setenv("RECO_SCRATCH_ROOT", str(tmp_path))
    monkeypatch.setattr(adapter, "check_ephemeral_budget", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        adapter,
        "_stream",
        lambda filename: iter(metadata if "meta_" in filename else reviews),
    )
    review_frame, metadata_frame = adapter.load_filtered_data()
    assert list(metadata_frame.parent_asin) == ["keep"]
    assert list(review_frame.parent_asin) == ["keep"]
    assert int(review_frame.timestamp.iloc[0]) == 1600000000000

    def fail_if_streamed(_filename):
        raise AssertionError("Verified caches must be reused")

    monkeypatch.setattr(adapter, "_stream", fail_if_streamed)
    review_frame_2, metadata_frame_2 = adapter.load_filtered_data()
    assert review_frame_2.equals(review_frame)
    assert metadata_frame_2.equals(metadata_frame)
