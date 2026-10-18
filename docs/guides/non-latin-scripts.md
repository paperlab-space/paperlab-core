# Non-Latin scripts

How text in Chinese, Japanese, Thai, Burmese, and other scripts is read, scored, and estimated.

## Reading the data

- Text is normalised to Unicode NFC when it is loaded, so the same character typed two ways (precomposed or with combining marks) compares equal.
- CSV files are read as UTF-8, or UTF-16 with a byte-order mark. Other encodings are detected. If text still looks garbled (`Ã©`, `�`), validation warns; re-save the file as UTF-8.
- Validation also warns about zero-width spaces and non-breaking spaces. These are common in Thai and Burmese text copied from the web, and they make exact matches fail even when the text looks identical.

## Tokenisation for token F1

`token_f1` compares outputs and references word by word. For scripts written without spaces between words, each character is a token:

| Script | Tokens |
| --- | --- |
| Chinese, Japanese (kanji, hiragana, katakana) | Characters |
| Thai, Lao, Khmer, Burmese, Tibetan | Characters, with combining marks (such as Thai vowel signs) kept on their base character |
| Korean, Latin, Cyrillic, and other spaced scripts | Words, split on whitespace |

Mixed text is split both ways: in `GPT-4 は便利`, `GPT-4` is one token and each Japanese character is another.

Character-level F1 is a coarse measure for these languages. A proper word segmenter would be better for some tasks; it is not built in.

## Normalisation in metrics

Metrics apply NFKC by default (`nfkc: true`). This treats full-width Latin letters and digits (`ＡＢＣ１２３`) as their ASCII forms and unifies other compatibility characters. See [Normalisation](metrics.md#normalisation).

With `labels`, a label in a script without spaces is found anywhere in the answer, while an English label must appear as a whole word. See [Label extraction](metrics.md#label-extraction).

## Cost estimates

Token estimates count about one token per character for Chinese, Japanese, Korean, Thai, Burmese, Lao, Khmer, Tibetan, and several Indic scripts, and about four characters per token for Latin-like text. Real tokenizers vary a lot across these scripts, so treat estimates as rough.

## Burmese and Zawgyi

Burmese text is expected in Unicode. Text in the older Zawgyi encoding looks like Burmese but uses different code points, so it never matches Unicode references. Convert it to Unicode first. Automatic Zawgyi detection is planned.
