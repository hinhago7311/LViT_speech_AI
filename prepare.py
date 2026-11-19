"""Validate corrected recordings and split them by recording session."""
import argparse
import json
from pathlib import Path

from data import read_manifest, split_by_session, validate_audio


def _heldout_identity(row):
    fingerprint = row.get("audio_sha256") or validate_audio(row["audio"])[1]
    return fingerprint, row["session"]


def prepare_dataset(manifest, output, seed=42, previous_data=None):
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"출력 폴더가 이미 있습니다. 새 경로를 지정하세요: {output}")
    rows = read_manifest(manifest)
    seen = {}
    for row in rows:
        duration, fingerprint = validate_audio(row["audio"])
        if fingerprint in seen:
            raise ValueError(f"중복 음성입니다: {row['audio']} / {seen[fingerprint]}")
        seen[fingerprint] = row["audio"]
        row.update(duration=duration, audio_sha256=fingerprint)
    previous_assignment = None
    corrections = []
    if previous_data is not None:
        previous_data = Path(previous_data).expanduser().resolve()
        previous_assignment = {}
        current_by_identity = {_heldout_identity(row): row for row in rows}
        for name in ("train", "validation", "test"):
            previous_rows = read_manifest(previous_data / f"{name}.jsonl")
            for row in previous_rows:
                session = row["session"]
                if previous_assignment.setdefault(session, name) != name:
                    raise ValueError(f"이전 자료에서 한 세션이 여러 분할에 있습니다: {session}")
                if name != "train":
                    current = current_by_identity.get(_heldout_identity(row))
                    if current is None:
                        source = row.get("source_name") or Path(row["audio"]).name
                        raise ValueError(
                            "이전 validation/test 음성 또는 세션이 빠졌습니다. "
                            "기존 항목의 검수 완료·저장을 확인하세요: "
                            f"{source} / 세션 {session}"
                        )
                    before = {key: row[key] for key in ("text", "language")}
                    after = {key: current[key] for key in ("text", "language")}
                    if before != after:
                        corrections.append({
                            "split": name, "audio_sha256": current["audio_sha256"],
                            "audio": current["audio"], "session": session,
                            "before": before, "after": after,
                        })
    splits = split_by_session(rows, seed, previous_assignment=previous_assignment)
    output.mkdir(parents=True, exist_ok=False)
    for name, records in splits.items():
        with (output / f"{name}.jsonl").open("x", encoding="utf-8") as destination:
            for row in records:
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    if corrections:
        with (output / "label_corrections.json").open("x", encoding="utf-8") as destination:
            json.dump({"previous_data": str(previous_data), "corrections": corrections},
                      destination, ensure_ascii=False, indent=2)
            destination.write("\n")
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="사용자가 정답을 확인한 JSONL 파일")
    parser.add_argument("--output", required=True, help="새로 만들 출력 폴더")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--previous-data", help="이전 prepared 폴더: 기존 분할을 유지하고 새 세션은 train에 추가")
    args = parser.parse_args()
    try:
        splits = prepare_dataset(args.manifest, args.output, args.seed, args.previous_data)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    for name, rows in splits.items():
        print(f"{name}: {len(rows)}개, {sum(row['duration'] for row in rows):.1f}초")
    print(f"저장 완료: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
