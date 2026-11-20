"""Compare the baseline and trained adapter on the same held-out recordings."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

from data import load_audio, read_manifest, validate_audio
from metrics import score
from runtime import adapter_source, load_model, recognize


def evaluate(rows, adapter=None, device="auto", revision=None):
    import torch

    processor, model, device = load_model(adapter=adapter, device=device, revision=revision)
    predictions, elapsed = [], 0.0
    for row in rows:
        samples = load_audio(row["audio"])
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        prediction = recognize(samples, processor, model, device, row["language"])
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - start
        predictions.append(prediction)
    references = [row["text"] for row in rows]
    result = {"metrics": score(references, predictions), "inference_seconds": elapsed,
              "device": device, "per_language": {}, "predictions": predictions}
    for language in sorted({row["language"] for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["language"] == language]
        result["per_language"][language] = score([references[i] for i in indices], [predictions[i] for i in indices])
    del model, processor
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="손대지 않은 test.jsonl")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True, help="새 평가 보고서 JSON")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    try:
        output = Path(args.output)
        if output.exists():
            raise FileExistsError("보고서가 이미 있습니다. 새 --output 경로를 지정하세요.")
        rows = read_manifest(args.data)
        _, revision, metadata = adapter_source(args.adapter)
        excluded_sessions = set(metadata.get("selection_sessions", []))
        excluded_audio = set(metadata.get("selection_audio_sha256", []))
        seen = set()
        for row in rows:
            _, fingerprint = validate_audio(row["audio"])
            if row["session"] in excluded_sessions or fingerprint in excluded_audio:
                raise ValueError("학습/체크포인트 선택에 사용한 자료는 최종 평가에 사용할 수 없습니다.")
            if fingerprint in seen:
                raise ValueError("평가 자료에 중복 음성이 있습니다.")
            seen.add(fingerprint)
        baseline = evaluate(rows, device=args.device, revision=revision)
        adapted = evaluate(rows, args.adapter, args.device, revision)
        report = {"normalization": "cer_raw=NFC; cer=NFC/casefold/ignore punctuation and whitespace; wer=normalized whitespace words",
                  "adapter": str(Path(args.adapter).resolve()), "training_info": metadata,
                  "baseline": baseline, "adapted": adapted,
                  "references": [{"audio": row["audio"], "text": row["text"], "language": row["language"], "session": row["session"]} for row in rows]}
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as destination:
            json.dump(report, destination, ensure_ascii=False, indent=2)
            destination.write("\n")
        print(f"CER: {baseline['metrics']['cer']['percent']:.2f}% → {adapted['metrics']['cer']['percent']:.2f}%")
        print(f"보고서: {output}")
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"평가 실패: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
