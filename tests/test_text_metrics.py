from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from rouge_score import rouge_scorer
from sacrebleu.metrics.bleu import BLEU as SacreBLEU
from sacrebleu.metrics.chrf import CHRF

from paperlab_space.config import MetricSpec
from paperlab_space.metrics import References, get_metric

HYPOTHESES: list[str] = [
    "the cat sat on the mat",
    "a quick brown fox jumps over the lazy dog",
    "it is raining heavily today",
    "paris is the capital of france",
]
REFERENCES: References = [
    ["the cat is sitting on the mat"],
    ["the quick brown fox jumped over the lazy dog"],
    ["it rains a lot today", "heavy rain today"],
    ["the capital of france is paris"],
]


def streams(references: References) -> Any:
    """sacreBLEU's reference streams, padded with None."""
    return [
        [r[0] for r in references if r],
        [r[1] if r and len(r) > 1 else None for r in references],
    ]


def test_bleu_matches_sacrebleu() -> None:
    result = get_metric("bleu").compute(list(HYPOTHESES), REFERENCES)
    expected = SacreBLEU().corpus_score(HYPOTHESES, streams(REFERENCES))
    assert result.value == pytest.approx(expected.score / 100)
    assert result.info["signature"].startswith("nrefs:var|case:mixed|eff:no|tok:13a")
    assert result.extra["brevity_penalty"] == pytest.approx(expected.bp)
    assert result.per_row is not None and all(0 <= v <= 1 for v in result.per_row if v is not None)
    assert result.n_scored == 4


def test_chrf_matches_sacrebleu() -> None:
    result = get_metric("chrf").compute(list(HYPOTHESES), REFERENCES)
    expected = CHRF().corpus_score(HYPOTHESES, streams(REFERENCES))
    assert result.value == pytest.approx(expected.score / 100)
    plus = get_metric(MetricSpec(name="chrf", params={"word_order": 2}))
    assert plus.compute(list(HYPOTHESES), REFERENCES).info["signature"].count("nw:2") == 1


@pytest.mark.parametrize("name", ["rouge1", "rouge2", "rougeL"])
def test_rouge_matches_rouge_score(name: str) -> None:
    reference = rouge_scorer.RougeScorer([name])
    expected = [
        max(reference.score(ref, hyp)[name].fmeasure for ref in refs)
        for hyp, refs in zip(HYPOTHESES, REFERENCES, strict=True)
        if refs
    ]
    result = get_metric(name).compute(list(HYPOTHESES), REFERENCES)
    assert result.per_row == pytest.approx(expected)
    assert result.value == pytest.approx(sum(expected) / len(expected))


def test_rouge_non_latin_scripts() -> None:
    # rouge-score drops non-Latin text entirely; ours splits it like token_f1.
    result = get_metric("rouge1").compute(["我爱你"], [["我爱他"]])
    assert result.value == pytest.approx(2 / 3)
    assert get_metric("rougeL").compute(["ကောင်းပါတယ်"], [["ကောင်းပါတယ်"]]).value == 1.0


def test_bleu_auto_tokenizes_scripts_without_spaces() -> None:
    result = get_metric("bleu").compute(["我爱北京天安门"], [["我爱北京天安门"]])
    assert result.value == pytest.approx(1.0)
    assert "tok:none" in result.info["signature"] and "tokenize" in result.info
    latin = get_metric("bleu").compute(["hello there"], [["hello there"]])
    assert "tokenize" not in latin.info
    char = get_metric(MetricSpec(name="bleu", params={"tokenize": "char"}))
    assert "tok:char" in char.compute(["你好"], [["你好"]]).info["signature"]


def test_failed_and_unreferenced_rows() -> None:
    bleu = get_metric("bleu")
    result = bleu.compute(["the cat", None, "x"], [["the cat"], ["a dog"], None])
    assert result.n_scored == 2 and result.extra["failed"] == 1
    assert result.per_row is not None and result.per_row[2] is None
    lenient = get_metric(MetricSpec(name="bleu", params={"count_failed": False}))
    long = "the cat sat on the mat all day"
    assert lenient.compute([long, None], [[long], ["a dog"]]).value == 1.0
    rouge = get_metric("rougeL").compute([None, "a b"], [["x"], ["a b"]])
    assert rouge.per_row == [0.0, 1.0]
    assert get_metric("bleu").compute([None], [None]).value is None


@pytest.mark.parametrize("name", ["bleu", "chrf"])
def test_corpus_scorer_matches_compute(name: str) -> None:
    metric = get_metric(name)
    predictions: list[str | None] = [*HYPOTHESES, None]
    references: list[list[str] | None] = [*REFERENCES, ["missing"]]
    score = metric.scorer(predictions, references)
    full = metric.compute(predictions, references).value
    assert score(np.arange(5)) == pytest.approx(full)
    subset = [0, 2, 2]
    expected = metric.compute(
        [predictions[i] for i in subset], [references[i] for i in subset]
    ).value
    assert score(np.array(subset)) == pytest.approx(expected)


def test_invalid_options() -> None:
    from paperlab_space.errors import ConfigError

    with pytest.raises(ConfigError, match="word_order"):
        get_metric(MetricSpec(name="chrf", params={"word_order": -1}))
    with pytest.raises(ConfigError, match="no option"):
        get_metric(MetricSpec(name="rouge1", params={"stem": True}))
