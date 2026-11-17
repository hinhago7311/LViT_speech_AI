"""Merge a trained LoRA adapter and export a self-contained int8 CT2 model."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

from runtime import BASE_MODEL


def export_adapter(adapter, output, model_name=BASE_MODEL):
    from ctranslate2.converters import TransformersConverter
    from transformers import WhisperTokenizerFast
    from runtime import adapter_source, load_model

    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"출력 경로가 이미 있습니다. 새 경로를 사용하세요: {output}")
    base, revision, info = adapter_source(adapter, model_name)
    processor, model, _ = load_model(model_name, adapter, device="cpu")
    model = model.merge_and_unload()
    output.parent.mkdir(parents=True, exist_ok=True)
    created_output = False
    try:
        with tempfile.TemporaryDirectory(prefix="whisper-export-", dir=output.parent) as temporary:
            root = Path(temporary)
            merged, converted = root / "merged", root / "converted"
            model.save_pretrained(merged, safe_serialization=True)
            processor.save_pretrained(merged)
            if not (merged / "tokenizer.json").is_file():
                WhisperTokenizerFast.from_pretrained(merged, from_slow=True).save_pretrained(merged)
            converter = TransformersConverter(
                str(merged), copy_files=["tokenizer.json", "preprocessor_config.json"]
            )
            converter.convert(str(converted), quantization="int8")
            required = ("model.bin", "config.json", "tokenizer.json", "preprocessor_config.json")
            for name in required:
                path = converted / name
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError(f"내보낸 모델에 필수 파일이 없습니다: {name}")
                if name.endswith(".json"):
                    json.loads(path.read_text(encoding="utf-8"))
            provenance = dict(info)
            provenance.update(base_model=base, base_revision=revision, export_format="ctranslate2", quantization="int8")
            (converted / "training_info.json").write_text(
                json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            shutil.copyfile(Path(__file__).with_name("WHISPER_LICENSE.txt"), converted / "WHISPER_LICENSE.txt")
            # Reserve a new destination only after conversion succeeds. Never replace
            # an existing model, even if another export created it in the meantime.
            output.mkdir()
            created_output = True
            for path in converted.iterdir():
                path.rename(output / path.name)
    except BaseException:
        if created_output:
            shutil.rmtree(output)
        raise
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="LoRA 어댑터를 int8 faster-whisper 모델로 내보내기")
    parser.add_argument("--adapter", required=True, help="학습 결과 어댑터 폴더")
    parser.add_argument("--output", required=True, help="새 모델 출력 폴더")
    parser.add_argument("--model", default=BASE_MODEL)
    args = parser.parse_args(argv)
    try:
        output = export_adapter(args.adapter, args.output, args.model)
        print(f"내보내기 완료: {output}")
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"내보내기 실패: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
