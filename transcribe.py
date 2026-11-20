"""Recognize one audio file with baseline Whisper small or a trained adapter."""
import argparse
import sys

from runtime import BASE_MODEL


def main(argv=None):
    parser = argparse.ArgumentParser(description="Whisper small 음성 파일 인식")
    parser.add_argument("audio", help="30초 이하 음성 파일 경로")
    parser.add_argument("--adapter", help="학습 결과 어댑터 폴더; 생략하면 기본 모델")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--language", choices=("ko", "en"), default="ko")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    from data import load_audio
    from runtime import load_model, recognize

    try:
        audio = load_audio(args.audio)
        processor, model, device = load_model(args.model, args.adapter, args.device)
        print(recognize(audio, processor, model, device, args.language))
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"인식 실패: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
