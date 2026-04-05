---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'Arial', sans-serif;
    font-size: 22px;
  }
  h1 { color: #1a3a5c; font-size: 36px; }
  h2 { color: #1a3a5c; font-size: 28px; border-bottom: 2px solid #1a3a5c; padding-bottom: 6px; }
  table { width: 100%; border-collapse: collapse; font-size: 18px; }
  th { background: #1a3a5c; color: white; padding: 8px; }
  td { padding: 8px; border: 1px solid #ccc; text-align: center; }
  tr:nth-child(even) { background: #f0f4f8; }
  .highlight { color: #e05c00; font-weight: bold; }
  .small { font-size: 16px; color: #555; }
---

# Inference-time PII Detection in RAG Systems
### Using Token-level Entropy as an Internal Signal

**RAG 환경에서 엔트로피 기반 실시간 개인정보 유출 탐지 연구**

---

## Problem: PII Leakage in RAG Systems

**RAG(Retrieval-Augmented Generation)** 환경에서 LLM은 외부 문서를 참조하여 답변을 생성합니다.
이 과정에서 개인정보(PII)가 의도치 않게 생성될 수 있습니다.

```
[User Query] "What is John's contact info?"
     ↓
[RAG Context] Name: John Smith, Phone: 010-1234-5678, Email: john@email.com
     ↓
[LLM Output] "John Smith can be reached at 010-1234-5678..."  ← PII 유출!
```

**기존 접근법의 한계**
- 외부 NER/LLM으로 생성 후 텍스트를 검열 → 추가 모델 필요, 비용 증가
- 생성 완료 후 대응 → 이미 PII가 생성된 상태

**우리의 접근:** LLM 내부 신호(entropy, hidden state)로 생성 중 PII를 탐지

---

## Key Hypothesis: Entropy Drop as PII Signal

LLM이 PII를 생성할 때, **토큰 단위 Shannon entropy가 급격히 감소**합니다.

> **Shannon Entropy:** $H_t = -\sum_i p_i \log p_i$
> LLM이 다음 토큰을 확신할수록 entropy가 낮아집니다.

```
일반 텍스트 생성:  entropy ≈ 3.0 nats  (다양한 후보 토큰 존재)
PII 토큰 생성:     entropy ≈ 0.2 nats  (정해진 값만 생성 → 확신)
```

**Dataset:** UnlearnPII (2,250명의 가상 인물 프로필, 16가지 PII 필드)

| 조건 | 설명 | 샘플 수 |
|------|------|------:|
| A_located | RAG context 제공 → PII 실제 생성 | 1,664 |
| A_not_located | RAG context 제공 → PII 미생성 | 586 |
| B_general | 일반 질문 (PII 없음) | 117 |
| C_no_context | Context 없음 | 2,250 |

---

## Experiment Results

**실험 설계:** 80/20 sample-level train/test split (seed=42, data leakage 없음)
**평가:** PII 생성 샘플 vs 나머지 전체 (binary classification)

### Method 1: Entropy Signal만 사용

| Method | F1 | Precision | Recall |
|--------|------|-----------|--------|
| M1: Adaptive ΔH Drop | 0.271 | 0.160 | 0.897 |
| M2: Sustained Low-entropy | 0.303 | 0.194 | 0.690 |

→ Threshold 기반 방법은 **False Positive가 너무 많아** 단독 사용 불가

### Method 2: Entropy Sequence Classification (CNN)

| Method | F1 | Precision | Recall |
|--------|------|-----------|--------|
| Logistic Regression | 0.758 | 0.616 | 0.985 |
| **CNN-2ch (entropy + Δentropy)** | **0.951** | **0.933** | **0.970** |
| Transformer | 0.940 | 0.898 | 0.985 |

→ Entropy sequence 전체 패턴을 CNN으로 학습하면 **F1=0.951** 달성

---

## Real-time Detection Feasibility

**핵심 질문:** 생성 완료 후가 아닌, **생성 도중** 탐지가 가능한가?

### PII 첫 등장 위치 분포 (A_located 1,664샘플)

| 통계 | 토큰 위치 |
|------|------:|
| 최솟값 | 0 |
| 중앙값 | **9번째 토큰** |
| 90th percentile | 20번째 토큰 |

### Prefix-only 탐지 성능 (CNN, K토큰만 보고 판단)

| Prefix K | F1 | Latency |
|----------|-----|---------|
| **5 tokens** | **0.921** | **PII 등장 4토큰 전** ✅ |
| 10 tokens | 0.951 | PII 등장 1토큰 후 ⚠️ |
| 전체 80 tokens | 0.796 | 생성 완료 후 |

→ **K=5 latency (5토큰 지연)만으로 F1=0.921, PII 등장 전 탐지 가능**

---

## Summary & Next Steps

### 연구 기여

| | 기존 방법 | 본 연구 |
|--|-----------|---------|
| 탐지 시점 | 생성 완료 후 | **생성 중 (K=5 latency)** |
| 사용 신호 | 텍스트 의미 분석 | **LLM 내부 entropy/hidden state** |
| 추가 모델 | NER/LLM 필요 | **경량 CNN (40K 파라미터)** |
| 탐지 성능 | - | **F1=0.951** |

### 현재 진행 중

- **Token-level PII localization:** 어느 토큰이 PII인지 특정 (entropy threshold 학습)
- **Linear Probe (Step 2):** PII 의심 토큰의 hidden state → 정밀 검증
- **Vector Steering:** 탐지 시점에 LLM 활성화 값을 조작하여 PII 생성 억제

> **목표:** Entropy 기반 실시간 탐지 → Vector Steering 방어까지 이어지는 end-to-end inference-time privacy defense 시스템
