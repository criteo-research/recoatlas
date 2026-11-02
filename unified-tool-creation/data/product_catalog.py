import pandas as pd
from typing import Dict


class ProductCatalog:
    """Generates lean product text representations (Title + Description)."""

    def __init__(self, df_reviews: pd.DataFrame):
        cols = ['parent_asin', 'product_title', 'product_description']
        self.products = df_reviews[cols].drop_duplicates(subset='parent_asin').set_index('parent_asin')

    def _truncate_title(self, parent_asin: str) -> str:
        title = self.products.loc[parent_asin, 'product_title']
        return title[:200] + "..." if len(title) > 200 else title

    def generate_one(self, parent_asin: str) -> str:
        """Lean representation: Title + Description."""
        p = self.products.loc[parent_asin]
        title = self._truncate_title(parent_asin)
        return f"Title: {title} | Description: {p['product_description']}"

    def generate_all(self) -> Dict[str, str]:
        """Generate lean representations for all products."""
        return {asin: self.generate_one(asin) for asin in self.products.index}
