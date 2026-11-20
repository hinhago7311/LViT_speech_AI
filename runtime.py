"""Shared local inference for the baseline and saved Whisper LoRA adapters."""
import json
from pathlib import Path


BASE_MODEL = "openai/whisper-small"


def adapter_source(adapter, model_name=BASE_MODEL, revision=None):
    """Read the actual base checkpoint, rejecting mismatched model selections."""
    from peft import PeftConfig

    adapter = Path(adapter)
    config = PeftConfig.from_pretrained(str(adapter))
    base = config.base_model_name_or_path
    if not base or base != model_name:
        raise ValueError(
            f"어댑터의 기본 모델은 {base!r}입니다. --model을 같은 모델로 지정하세요."
        )
    info_path = adapter / "training_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    if info.get("base_model", base) != base:
        raise ValueError("어댑터 설정과 training_info.json의 기본 모델이 다릅니다.")
    recorded_revision = info.get("base_revision") or config.revision
    if revision and recorded_revision and revision != recorded_revision:
        raise ValueError("학습 때 사용한 기본 모델 revision과 요청한 revision이 다릅니다.")
    return base, recorded_revision or revision, info


def load_model(model_name=BASE_MODEL, adapter=None, device="auto", revision=None, trainable=False):
    """Return (processor, model, device), with LoRA applied when requested."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError("device는 auto, cpu 또는 cuda여야 합니다.")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA GPU를 사용할 수 없습니다. 설치를 확인하거나 --device cpu를 쓰세요.")
    if adapter:
        model_name, revision, _ = adapter_source(adapter, model_name, revision)
    kwargs = {"revision": revision} if revision else {}
    model = WhisperForConditionalGeneration.from_pretrained(model_name, **kwargs)
    resolved_revision = getattr(model.config, "_commit_hash", None) or revision
    processor_kwargs = {"revision": resolved_revision} if resolved_revision and not adapter else {}
    processor = WhisperProcessor.from_pretrained(str(adapter) if adapter else model_name, **processor_kwargs)
    if adapter:
        # Generic PeftModel preserves Whisper's input_features argument.
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.to(device)
    model.train(trainable)
    return processor, model, device


def recognize(audio, processor, model, device, language="ko"):
    """Recognize one checked <=30-second, mono 16 kHz float32 recording."""
    import torch

    if language not in ("ko", "en"):
        raise ValueError("language는 ko 또는 en이어야 합니다.")
    inputs = processor(
        audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
    )
    features = inputs.input_features.to(device)
    attention_mask = inputs.attention_mask.to(device)
    max_tokens = max(1, model.config.max_target_positions - 4)
    with torch.inference_mode():
        tokens = model.generate(
            features,
            attention_mask=attention_mask,
            language=language,
            task="transcribe",
            return_timestamps=False,
            max_new_tokens=max_tokens,
            num_beams=3,
            do_sample=False,
        )
    return processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
