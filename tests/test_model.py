"""CPU integration checks using local random weights, never downloaded models."""
import argparse
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import (
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizerFast,
)
from transformers.models.whisper.tokenization_whisper import LANGUAGES

from runtime import load_model, recognize


def tiny_checkpoint(path):
    """One-layer Whisper plus byte tokenizer, small enough for an offline CPU check."""
    special = ["<|endoftext|>", "<|startoftranscript|>"]
    special += [f"<|{language}|>" for language in LANGUAGES]
    special += ["<|translate|>", "<|transcribe|>", "<|startoflm|>",
                "<|startofprev|>", "<|nocaptions|>", "<|notimestamps|>"]
    vocab = {text: index for index, text in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()) + special)}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<|endoftext|>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = WhisperTokenizerFast(
        tokenizer_object=backend, bos_token="<|endoftext|>", eos_token="<|endoftext|>",
        unk_token="<|endoftext|>", pad_token="<|endoftext|>", additional_special_tokens=special,
        language="ko", task="transcribe",
    )
    tokenizer.set_prefix_tokens(language="ko", task="transcribe")
    processor = WhisperProcessor(WhisperFeatureExtractor(chunk_length=1), tokenizer)
    config = WhisperConfig(
        vocab_size=len(tokenizer), encoder_layers=1, decoder_layers=1, d_model=16,
        encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=32, decoder_ffn_dim=32,
        max_source_positions=50, max_target_positions=24,
        pad_token_id=vocab["<|endoftext|>"], bos_token_id=vocab["<|endoftext|>"],
        eos_token_id=vocab["<|endoftext|>"], decoder_start_token_id=vocab["<|startoftranscript|>"],
        suppress_tokens=None, begin_suppress_tokens=None,
    )
    model = WhisperForConditionalGeneration(config)
    generation = model.generation_config
    generation._from_model_config = False
    generation.lang_to_id = {f"<|{language}|>": vocab[f"<|{language}|>"] for language in LANGUAGES}
    generation.task_to_id = {task: vocab[f"<|{task}|>"] for task in ("transcribe", "translate")}
    generation.no_timestamps_token_id = vocab["<|notimestamps|>"]
    generation.is_multilingual = True
    generation.suppress_tokens = None
    generation.begin_suppress_tokens = None
    model.save_pretrained(path)
    processor.save_pretrained(path)


class ModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(7)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.base = cls.root / "base"
        tiny_checkpoint(cls.base)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        torch.set_num_threads(cls.previous_threads)

    def test_lora_update_reload_recognize_and_export(self):
        from export import export_adapter
        from train import Collator

        processor, model, device = load_model(str(self.base), device="cpu", trainable=True)
        model = get_peft_model(model, LoraConfig(
            r=2, lora_alpha=4, lora_dropout=0, bias="none", target_modules=["q_proj", "v_proj"]
        ))
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        samples = (np.sin(np.arange(16000, dtype=np.float32) * .02) * .1).astype(np.float32)
        batch = Collator(processor, model.config.decoder_start_token_id, model.config.max_target_positions)([
            {"samples": samples, "text": "안녕", "language": "ko"},
            {"samples": samples, "text": "hello", "language": "en"},
        ])
        self.assertEqual(batch["labels"][0, 0].item(), processor.tokenizer.convert_tokens_to_ids("<|ko|>"))
        self.assertEqual(batch["labels"][1, 0].item(), processor.tokenizer.convert_tokens_to_ids("<|en|>"))
        self.assertTrue(torch.any(batch["labels"] == -100))
        trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        before = {name: parameter.detach().clone() for name, parameter in trainable.items()}
        optimizer = torch.optim.AdamW(trainable.values(), lr=.01)
        model.train()
        loss = model(**batch).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None and torch.any(parameter.grad != 0) for parameter in trainable.values()))
        optimizer.step()
        self.assertTrue(any(not torch.equal(before[name], parameter) for name, parameter in trainable.items()))
        model.eval()
        with torch.inference_mode():
            expected_logits = model(**batch).logits

        adapter = self.root / "updated-adapter"
        model.save_pretrained(adapter)
        processor.save_pretrained(adapter)
        (adapter / "training_info.json").write_text(json.dumps({
            "base_model": str(self.base), "base_revision": None,
        }), encoding="utf-8")
        reloaded_processor, reloaded, device = load_model(str(self.base), adapter, device="cpu")
        with torch.inference_mode():
            torch.testing.assert_close(reloaded(**batch).logits, expected_logits, atol=1e-6, rtol=1e-5)
        for language in ("ko", "en"):
            # Random weights have no recognition accuracy; this exercises generation.
            self.assertIsInstance(recognize(samples, reloaded_processor, reloaded, device, language), str)
        with self.assertRaisesRegex(ValueError, "기본 모델"):
            load_model(adapter=adapter, device="cpu")

        exported = export_adapter(adapter, self.root / "exported", model_name=str(self.base))
        for filename in ("model.bin", "config.json", "tokenizer.json", "preprocessor_config.json", "training_info.json", "WHISPER_LICENSE.txt"):
            self.assertGreater((exported / filename).stat().st_size, 0)
        import ctranslate2

        engine = ctranslate2.models.Whisper(str(exported), device="cpu", compute_type="int8")
        self.assertEqual(engine.device, "cpu")
        with self.assertRaises(FileExistsError):
            export_adapter(adapter, exported, model_name=str(self.base))

    def test_actual_training_loop_one_cpu_step(self):
        import train

        prepared = self.root / "prepared"
        prepared.mkdir()
        for index, split in enumerate(("train", "validation", "test")):
            audio = prepared / f"{split}.wav"
            samples = np.sin(np.arange(16000, dtype=np.float32) * (.02 + index * .01)) * .1
            sf.write(audio, samples, 16000, subtype="PCM_16")
            row = {"audio": str(audio), "text": "안녕", "session": split, "language": "ko", "reviewed": True}
            (prepared / f"{split}.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        args = argparse.Namespace(
            output=str(self.root / "training-run"), data=str(prepared), adapter=None,
            epochs=1, batch_size=1, accumulation=2, learning_rate=.001,
            max_steps=1, seed=42, device="cpu", precision="fp32",
        )

        def local_model(**kwargs):
            return load_model(str(self.base), **kwargs)

        with patch.object(train, "BASE_MODEL", str(self.base)), patch.object(train, "load_model", local_model), redirect_stdout(io.StringIO()):
            best = train.train(args)
        history = json.loads((Path(args.output) / "history.json").read_text(encoding="utf-8"))
        self.assertEqual(history[-1]["step"], 1)
        self.assertTrue(math.isfinite(history[-1]["loss"]))
        info = json.loads((best / "training_info.json").read_text(encoding="utf-8"))
        self.assertFalse(info["test_used_for_training"])
        self.assertEqual(info["base_model"], str(self.base))
        self.assertEqual(info["train_count"], 1)
        _, model, device = load_model(str(self.base), best, device="cpu", trainable=True)
        self.assertTrue(any(parameter.requires_grad for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
