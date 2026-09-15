from __future__ import annotations

import unittest

from prepare_triage_data import dual_tokenizer_filter, historical_prefix_hashes, parse_wikitext_articles


class FakeTokenizer:
    def __call__(self, texts, *, add_special_tokens: bool, truncation: bool, max_length: int, padding: bool):
        values = texts if isinstance(texts, list) else [texts]
        encoded = [
            (text.split() + (["special"] if add_special_tokens else []))[:max_length]
            for text in values
        ]
        return {"input_ids": encoded}


class PrepareDataTests(unittest.TestCase):
    def test_article_parser_matches_shared_semantics(self) -> None:
        rows = ["= One =", " alpha   beta ", "= = sub = =", "gamma", "= Two =", "short"]
        self.assertEqual(parse_wikitext_articles(rows, min_chars=5), ["One alpha beta = = sub = = gamma", "Two short"])

    def test_historical_prefix_is_deterministic(self) -> None:
        docs = [f"document {index}" for index in range(20)]
        self.assertEqual(
            historical_prefix_hashes(docs, seed=3, count=10),
            historical_prefix_hashes(docs, seed=3, count=10),
        )

    def test_dual_tokenizer_filter_requires_both(self) -> None:
        accepted, lengths = dual_tokenizer_filter(
            ["one two", "one two three four"],
            {"a": FakeTokenizer(), "b": FakeTokenizer()},
            required_tokens=4,
        )
        self.assertEqual(accepted, ["one two three four"])
        self.assertEqual(len(lengths), 1)


if __name__ == "__main__":
    unittest.main()
