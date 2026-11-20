"""Corpus-weighted error rates, with explicit Korean/English normalization."""
import unicodedata


def distance(left, right):
    previous = list(range(len(right) + 1))
    for index, item in enumerate(left, 1):
        current = [index]
        for column, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (item != other)))
        previous = current
    return previous[-1]


def normalize(text):
    text = unicodedata.normalize("NFC", text).casefold()
    return " ".join("".join(c for c in text if not unicodedata.category(c).startswith("P")).split())


def score(references, predictions):
    if len(references) != len(predictions) or not references:
        raise ValueError("같은 개수의 정답과 예측 문장이 필요합니다.")
    totals = {key: [0, 0] for key in ("cer_raw", "cer", "wer")}
    for reference, prediction in zip(references, predictions):
        raw = (unicodedata.normalize("NFC", reference), unicodedata.normalize("NFC", prediction))
        normalized = (normalize(reference), normalize(prediction))
        units = {"cer_raw": raw,
                 "cer": tuple("".join(text.split()) for text in normalized),
                 "wer": tuple(text.split() for text in normalized)}
        for key, (expected, actual) in units.items():
            totals[key][0] += distance(expected, actual)
            totals[key][1] += len(expected)
    return {key: {"percent": 100 * errors / max(1, length), "errors": errors, "reference_units": length}
            for key, (errors, length) in totals.items()}
