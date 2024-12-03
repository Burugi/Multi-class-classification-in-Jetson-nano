# 임베디드 AI 보고서 README

## 프로젝트 개요

### 제목: QuickDraw 모델의 다중 경량화 기법 적용 및 성능 분석

본 프로젝트는 Knowledge Distillation, Pruning, Quantization 등 다양한 모델 경량화 기법을 활용하여 Jetson Nano와 같은 임베디드 환경에서 효과적으로 동작할 수 있는 딥러닝 모델을 개발하고 그 성능을 분석하는 것을 목표로 합니다.

---

## 연구 내용

### 연구 배경
- 딥러닝 모델의 성능 향상과 함께 모델 크기와 연산량이 증가.
- 제한된 컴퓨팅 리소스 환경에서 실시간 추론을 위해 경량화 기법이 필요.
- Jetson Nano 환경을 고려한 최적화 필수.

### 연구 목표
1. 모델 성능 유지 및 경량화를 위한 다양한 기법 조합.
2. QuickDraw 데이터셋 활용한 이미지 분류 태스크 수행.
3. Jetson Nano에서의 최적화 방안 도출.

### 주요 기법
- **Knowledge Distillation**: Teacher-Student 구조로 성능 유지.
- **Pruning**: 가중치 삭제를 통해 모델 크기 축소.
- **Quantization**: 낮은 비트 수를 활용한 양자화 기법.

---

## 실험 설계

### 기본 모델 구조
- Teacher Model: VGG 기반 아키텍처.
- Student Model: Compact CNN 아키텍처.

### 실험 방법
1. Knowledge Distillation을 기본 프레임워크로 설정.
2. Pruning 및 Quantization 기법 적용 후 성능 비교.
3. 각 설정에서 3회 반복 실험 진행.

### 성능 평가 지표
- 모델 크기 및 연산량.
- 추론 속도(FPS) 및 정확도.
- 메모리 사용량.

---

## 실험 결과

### 주요 결과
- Student 모델은 Teacher 모델 대비 파라미터 수 99.7% 감소.
- 정확도 손실은 1.09%p로 미미함 (96.53% → 95.44%).

### 최적화 기법 성능
- Pruning: 최대 43.42% FPS 향상.
- Quantization: 최대 41.26% FPS 향상.
- 두 기법 모두 메모리 사용량 약 46~47% 유지.

### 결론
- 경량화 기법 적용 후에도 높은 정확도와 실시간 추론 성능 유지.
- Jetson Nano에서 실용적으로 활용 가능한 모델 개발 성공.

---

## 실행 가이드

### 필수 파일
- 데이터셋 준비: QuickDraw 데이터셋 ([링크](https://quickdraw.withgoogle.com/data)).
- 코드 실행:
  - `QuickDraw_Teacher_Model_Training.ipynb`
  - `QuickDraw_Student_Model_Training.ipynb`
  - `pruned_student_train.py`
  - `quantized_student_train.py`
- 평가 스크립트: `model_evaluator.py`.

### 실행 단계
1. Jetson Nano에서 필요한 의존성 설치.
2. 각 경량화 모델 학습 및 추론.
3. `Run.txt` 참고하여 실행.

---

## 참고 자료
- [UMass LFW Dataset](https://vis-www.cs.umass.edu/lfw/)
- [Google QuickDraw Dataset](https://quickdraw.withgoogle.com/data)
