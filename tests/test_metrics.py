from __future__ import annotations

import pytest

from paperlab_core.config import MetricSpec, RunConfig
from paperlab_core.data import ColumnMapping, Dataset
from paperlab_core.errors import ConfigError
from paperlab_core.metrics import (
    Normalizer,
    available_metrics,
    check_metrics,
    extract_label,
    get_metric,
    tokenize,
)
from paperlab_core.runner import run

from .fakes import FakeBackend


def test_normalizer() -> None:
    norm = Normalizer()
    assert norm('  "Positive."  ') == "positive"
    assert norm("ＡＢＣ") == "abc"  # full-width -> ASCII via NFKC
    assert norm("答案。") == "答案"
    assert norm("-5%") == "-5%"
    assert Normalizer(ignore_punctuation=True)("don't stop!") == "don t stop"
    assert Normalizer(remove_articles=True)("The cat") == "cat"


def test_tokenize_scripts() -> None:
    assert tokenize("hello world") == ["hello", "world"]
    assert tokenize("我爱你") == ["我", "爱", "你"]
    assert tokenize("Python很好") == ["Python", "很", "好"]
    assert tokenize("안녕 하세요") == ["안녕", "하세요"]
    assert tokenize("กิน")[0] == "กิ"  # combining vowel stays with its consonant


def test_extract_label() -> None:
    labels = ["positive", "negative", "very positive"]
    norm = Normalizer()
    assert extract_label("The sentiment is Negative.", labels, norm) == "negative"
    assert extract_label("very positive!", labels, norm) == "very positive"
    assert extract_label("positive or negative", labels, norm) == "positive or negative"
    assert extract_label("unpositive", labels, norm) == "unpositive"  # whole words only
    assert extract_label("答案是正面", ["正面", "负面"], norm) == "正面"


def test_exact_match_rows() -> None:
    metric = get_metric("exact_match")
    result = metric.compute(
        ["Paris.", "berlin", None, "x"], [["paris"], ["Bonn", "Berlin"], ["a"], None]
    )
    assert result.per_row == [1.0, 1.0, 0.0, None]
    assert result.value == pytest.approx(2 / 3)
    assert result.n_scored == 3 and result.extra["failed"] == 1
    lenient = get_metric(MetricSpec(name="exact_match", params={"count_failed": False}))
    assert lenient.compute(["a", None], [["a"], ["b"]]).value == 1.0
    strict = get_metric(MetricSpec(name="exact_match", params={"ignore_case": False}))
    assert strict.compute(["Paris"], [["paris"]]).value == 0.0


def test_accuracy_with_labels() -> None:
    metric = get_metric(MetricSpec(name="accuracy", params={"labels": ["pos", "neg"]}))
    result = metric.compute(["It is pos.", "neg", "pos and neg"], [["pos"], ["pos"], ["neg"]])
    assert result.per_row == [1.0, 0.0, 0.0]


def test_f1_matches_hand_computation() -> None:
    gold = ["a", "a", "a", "b", "b", "c"]
    pred = ["a", "a", "b", "b", "c", "c"]
    result = get_metric("f1").compute(list(pred), [[g] for g in gold])
    # a: P=1, R=2/3, F1=0.8; b: P=1/2, R=1/2, F1=0.5; c: P=1/2, R=1, F1=2/3
    assert result.extra["f1[a]"] == pytest.approx(0.8)
    assert result.extra["f1[b]"] == pytest.approx(0.5)
    assert result.extra["f1[c]"] == pytest.approx(2 / 3)
    assert result.value == pytest.approx((0.8 + 0.5 + 2 / 3) / 3)
    assert result.extra["f1_micro"] == pytest.approx(4 / 6)
    assert result.extra["f1_weighted"] == pytest.approx((0.8 * 3 + 0.5 * 2 + 2 / 3) / 6)
    micro = get_metric(MetricSpec(name="f1", params={"average": "micro"}))
    assert micro.compute(list(pred), [[g] for g in gold]).value == pytest.approx(4 / 6)


def test_f1_multi_answer_and_failures() -> None:
    result = get_metric("f1").compute(["b", None], [["a", "b"], ["a"]])
    # row 1 is correct ("b" is a valid answer); row 2 failed -> wrong
    assert result.extra["f1[b]"] == 1.0 and result.extra["f1[a]"] == 0.0
    assert get_metric("f1").compute([None], [None]).value is None


def test_token_f1() -> None:
    metric = get_metric("token_f1")
    result = metric.compute(
        ["the cat sat", "我爱你", "", "x"], [["a cat sat down"], ["我爱他"], ["word"], ["y", "x"]]
    )
    assert result.per_row is not None
    assert result.per_row[0] == pytest.approx(2 * (2 / 3) * (2 / 4) / (2 / 3 + 2 / 4))
    assert result.per_row[1] == pytest.approx(2 / 3)
    assert result.per_row[2] == 0.0  # empty output scores zero
    assert result.per_row[3] == 1.0


def test_registry_errors() -> None:
    assert {"exact_match", "accuracy", "f1", "token_f1"} <= set(available_metrics())
    with pytest.raises(ConfigError, match="Unknown metric"):
        get_metric("bleu_typo")
    with pytest.raises(ConfigError, match="no option"):
        get_metric(MetricSpec(name="f1", params={"avg": "macro"}))
    with pytest.raises(ConfigError, match="average must be"):
        get_metric(MetricSpec(name="f1", params={"average": "median"}))


def test_metrics_need_answer_key() -> None:
    config = RunConfig.from_dict(
        {
            "mapping": {"input": "q"},
            "prompt": "{q}",
            "models": ["openai/m"],
            "metrics": ["exact_match"],
        }
    )
    data = Dataset.from_records([{"q": "x"}]).map(ColumnMapping(input="q"))
    with pytest.raises(ConfigError, match="need an answer key"):
        check_metrics(config, data)
    backend = FakeBackend()
    with pytest.raises(ConfigError, match="need an answer key"):
        run(config, data, backend=backend)
    assert backend.calls == []  # failed before any request


def test_run_scores_with_labels() -> None:
    data = Dataset.from_records([{"q": "a", "gold": "A"}, {"q": "b", "gold": "C"}])
    config = RunConfig.from_dict(
        {
            "mapping": {"input": "q", "expected": "gold"},
            "prompt": "{q}",
            "models": ["openai/m"],
            "metrics": [
                "exact_match",
                {"name": "exact_match", "label": "em_strict", "ignore_case": False},
            ],
        }
    )
    results = run(config, data, backend=FakeBackend(lambda m, p, a: p))
    assert results.scores["m"]["exact_match"].value == 0.5
    assert results.scores["m"]["em_strict"].value == 0.0
    assert results.scores["m"]["em_strict"].name == "em_strict"
