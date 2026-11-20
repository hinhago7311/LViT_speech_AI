"""Fine-tune Whisper small with LoRA using reviewed audio/text pairs."""
import argparse
import json
import math
from pathlib import Path
import random
import sys

from data import load_audio, read_manifest, validate_audio
from metrics import score
from runtime import BASE_MODEL, adapter_source, load_model, recognize


def checked_splits(directory):
    splits, seen_audio, seen_sessions = {}, {}, {}
    for name in ("train", "validation", "test"):
        splits[name] = read_manifest(Path(directory) / f"{name}.jsonl")
        for row in splits[name]:
            _, fingerprint = validate_audio(row["audio"])
            if fingerprint in seen_audio:
                raise ValueError(f"중복 음성입니다: {row['audio']} / {seen_audio[fingerprint]}")
            seen_audio[fingerprint] = row["audio"]
            row["audio_sha256"] = fingerprint
            previous = seen_sessions.setdefault(row["session"], name)
            if previous != name:
                raise ValueError(f"녹음 세션이 {previous}/{name}에 겹칩니다: {row['session']}")
    return splits


def data_history(splits, previous):
    """Keep held-out recordings held out across repeated fine-tuning runs."""
    current = {}
    for name, rows in splits.items():
        current[name] = {"sessions": {row["session"] for row in rows},
                         "audio_sha256": {row["audio_sha256"] for row in rows}}
    result = {}
    for unit in ("sessions", "audio_sha256"):
        past_training = set(previous.get(f"training_{unit}", []))
        past_selection = set(previous.get(f"selection_{unit}", []))
        past_held_out = set(previous.get(f"held_out_{unit}", []))
        selection = current["train"][unit] | current["validation"][unit]
        if current["validation"][unit] & past_training:
            raise ValueError("이전 모델의 학습 자료가 현재 validation에 포함되어 있습니다.")
        if current["test"][unit] & past_selection or selection & past_held_out:
            raise ValueError("이어 학습에서도 기존 test 자료를 학습/선택 자료와 분리하세요.")
        result[f"training_{unit}"] = sorted(past_training | current["train"][unit])
        result[f"selection_{unit}"] = sorted(past_selection | selection)
        result[f"held_out_{unit}"] = sorted(past_held_out | current["test"][unit])
    return result


class Recordings:
    """Read audio on demand; do not retain the entire dataset in RAM."""
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {**row, "samples": load_audio(row["audio"])}


class Collator:
    def __init__(self, processor, decoder_start_token_id, max_target_positions):
        self.processor = processor
        self.start = decoder_start_token_id
        self.max_target_positions = max_target_positions

    def __call__(self, rows):
        import torch

        features = self.processor.feature_extractor(
            [row["samples"] for row in rows], sampling_rate=16000,
            return_tensors="pt", return_attention_mask=True,
        )
        labels = []
        for row in rows:
            self.processor.tokenizer.set_prefix_tokens(language=row["language"], task="transcribe")
            ids = self.processor.tokenizer(row["text"]).input_ids
            if ids and ids[0] == self.start:
                ids = ids[1:]
            if len(ids) > self.max_target_positions:
                raise ValueError("정답이 모델의 토큰 길이를 초과합니다. 음성과 정답을 함께 나누세요.")
            labels.append({"input_ids": ids})
        padded = self.processor.tokenizer.pad(labels, return_tensors="pt")
        return {"input_features": features.input_features,
                "attention_mask": features.attention_mask,
                "labels": padded.input_ids.masked_fill(padded.attention_mask.ne(1), -100).to(torch.long)}


def precision_for(device, requested):
    import torch

    if device == "cpu":
        if requested not in ("auto", "fp32"):
            raise ValueError("CPU 학습에는 --precision fp32를 사용하세요.")
        return "fp32"
    if requested != "auto":
        if requested == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("이 GPU는 BF16 학습을 지원하지 않습니다.")
        return requested
    if torch.cuda.is_bf16_supported():
        return "bf16"
    return "fp16" if torch.cuda.get_device_capability()[0] >= 7 else "fp32"


def validate(rows, processor, model, device):
    model.eval()
    predictions = [recognize(load_audio(row["audio"]), processor, model, device, row["language"])
                   for row in rows]
    return score([row["text"] for row in rows], predictions)


def train(args):
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    output = Path(args.output)
    if output.exists():
        raise FileExistsError("결과 폴더가 이미 있습니다. --output에 새 폴더를 지정하세요.")
    if min(args.epochs, args.batch_size, args.accumulation) < 1 or args.learning_rate <= 0:
        raise ValueError("epochs, batch-size, accumulation, learning-rate는 양수여야 합니다.")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max-steps는 양수여야 합니다.")
    splits = checked_splits(args.data)
    previous = adapter_source(args.adapter)[2] if args.adapter else {}
    if args.adapter and not all(key in previous for key in ("training_sessions", "selection_sessions", "held_out_sessions")):
        raise ValueError("이어 학습에는 이 도구의 데이터 이력이 포함된 training_info.json이 필요합니다.")
    history_info = data_history(splits, previous)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    processor, model, device = load_model(adapter=args.adapter, device=args.device, trainable=True)
    if not args.adapter:
        model = get_peft_model(model, LoraConfig(
            r=8, lora_alpha=16, lora_dropout=.05, bias="none", target_modules=["q_proj", "v_proj"],
        ))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    precision = precision_for(device, args.precision)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("학습 가능한 어댑터 파라미터가 없습니다.")
    loader = DataLoader(Recordings(splits["train"]), batch_size=args.batch_size, shuffle=True,
                        num_workers=0, collate_fn=Collator(processor, model.config.decoder_start_token_id,
                                                         model.config.max_target_positions))
    steps = args.epochs * math.ceil(len(loader) / args.accumulation)
    if args.max_steps:
        steps = min(steps, args.max_steps)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * .1), steps)
    output.mkdir(parents=True)
    info = {"base_model": BASE_MODEL, "base_revision": getattr(model.config, "_commit_hash", None),
            "method": "lora", "precision": precision, "seed": args.seed,
            "warm_start_adapter": args.adapter, "train_count": len(splits["train"]),
            "validation_count": len(splits["validation"]), "test_count": len(splits["test"]),
            "trainable_parameters": sum(parameter.numel() for parameter in parameters),
            "test_used_for_training": False, **history_info}

    def save_best(metrics, epoch, step):
        model.save_pretrained(output / "best", safe_serialization=True)
        processor.save_pretrained(output / "best")
        metadata = {**info, "epoch": epoch, "step": step, "validation": metrics}
        (output / "best" / "training_info.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    initial = validate(splits["validation"], processor, model, device)
    best = initial["cer"]["percent"]
    save_best(initial, 0, 0)
    history = [{"epoch": 0, "step": 0, "validation": initial}]
    step = 0
    print(f"device={device}, precision={precision}, trainable={info['trainable_parameters']:,}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for index, batch in enumerate(loader):
            window = min(args.accumulation, len(loader) - index // args.accumulation * args.accumulation)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device, dtype=dtype, enabled=precision != "fp32"):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise ValueError("학습 손실이 NaN/무한대입니다. 데이터와 정밀도 설정을 확인하세요.")
            losses.append(float(loss.detach()))
            scaler.scale(loss / window).backward()
            if (index + 1) % args.accumulation == 0 or index + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                print(f"epoch={epoch}, step={step}/{steps}, loss={losses[-1]:.4f}", flush=True)
                if step >= steps:
                    break
        metrics = validate(splits["validation"], processor, model, device)
        history.append({"epoch": epoch, "step": step, "loss": sum(losses) / len(losses), "validation": metrics})
        if metrics["cer"]["percent"] < best:
            best = metrics["cer"]["percent"]
            save_best(metrics, epoch, step)
        (output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if step >= steps:
            break
    print(f"완료: {output / 'best'} · validation CER={best:.2f}% (test는 별도 평가)", flush=True)
    return output / "best"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="prepare.py가 만든 폴더")
    parser.add_argument("--output", required=True, help="새 학습 결과 폴더")
    parser.add_argument("--adapter", help="기존 어댑터에서 새 학습 시작 (optimizer 복원 아님)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    args = parser.parse_args()
    try:
        train(args)
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"학습 실패: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
