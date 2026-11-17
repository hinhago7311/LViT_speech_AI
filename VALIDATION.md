# 초기 구현 검증

검증일: 2026-11-17 (Asia/Seoul). 실행 환경은 Linux, Python 3.12.14, CPU입니다.

## 실행한 검사

```sh
python -m unittest discover -s tests -v
git diff --check
```

**자동 테스트 14개 통과.** `prepare.py`, `train.py`, `evaluate.py`, `transcribe.py`, `export.py`의 `--help` 실행과 전체 Python 파일 구문 검사를 통과했습니다.

검증한 항목:

- 검수 표시·정답·언어·파일 경로 검사
- stereo/다른 샘플레이트의 mono 16kHz 변환
- 빈 음성·30초 초과·NaN/무한대·같은 디코딩 음성 중복 거부
- 세션 단위 train/validation/test 분리와 이어 학습 때 기존 분할 유지
- 이전 평가 자료 누락·세션 배정 충돌·기존 출력 덮어쓰기 거부
- 누적 학습·선택·test 이력의 경계 유지
- 한국어·영어 정답 prefix, 시작 토큰 제거, padding -100
- 작은 무작위 Whisper 모델에서 실제 LoRA 역전파·파라미터 변경
- 실제 학습 루프 CPU 1단계, validation 생성, best 저장, 이어 학습용 파라미터 활성화
- 어댑터 재로딩 전후 logits 일치, 한국어·영어 생성 코드 경로
- 실제 어댑터 병합·CTranslate2 int8 변환·CPU 엔진 로딩·필수 파일 확인
- 원문/정규화 CER 및 corpus 가중 집계

주요 의존성: torch 2.6.0+cpu, transformers 4.57.1, peft 0.17.1, soundfile 0.13.1, faster-whisper 1.2.1, ctranslate2 4.9.0.

## 아직 수행하지 않은 검증

- 실제 사용자 녹음 데이터와 정답을 사용한 추가 학습
- 원본 Whisper small 전체 가중치를 로드한 학습·평가·내보내기 실행
- 인식 정확도 개선이나 다양한 발음·말투에 대한 일반화
- GTX 1050 Ti 4GB / RTX 5060 Ti 16GB의 CUDA 설치, VRAM, 처리 속도
- Windows 또는 실제 LViT·OBS 연결 실행

통합 테스트의 작은 모델은 무작위 가중치이며, 생성된 문장은 음성 인식 정확도의 근거로 사용하지 않습니다. 실제 자료를 준비한 뒤 test 집합의 원본 모델 대비 CER·WER·처리 시간을 평가해야 합니다.
