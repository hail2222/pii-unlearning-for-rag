---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'Arial', sans-serif;
    font-size: 21px;
    padding: 40px 50px;
  }
  h1 { color: #c00000; font-size: 34px; margin-bottom: 8px; }
  h2 { color: #c00000; font-size: 26px; border-bottom: 2px solid #c00000; padding-bottom: 5px; }
  h3 { color: #1a1a1a; font-size: 20px; margin: 10px 0 4px; }
  table { width: 100%; border-collapse: collapse; font-size: 17px; margin-top: 8px; }
  th { background: #c00000; color: white; padding: 7px 10px; }
  td { padding: 7px 10px; border: 1px solid #ccc; }
  tr:nth-child(even) { background: #fdf0f0; }
  .tag { background: #c00000; color: white; border-radius: 4px; padding: 2px 8px; font-size: 14px; margin-right: 4px; }
  .box { background: #f9f9f9; border-left: 4px solid #c00000; padding: 10px 15px; margin: 8px 0; }
  .small { font-size: 15px; color: #555; }
---

# Privacy-Aware RAG Systems
## Research on Real-time PII Detection in LLM Generation

**SK hynix SSD Architecture — Job PT**

<br>

> **Research Focus:** RAG 환경에서 LLM이 개인정보(PII)를 생성하는 순간을 내부 신호(Entropy)로 탐지하는 inference-time 방어 시스템 연구

<br>

<span class="tag">#RAG</span> <span class="tag">#LLM Inference</span> <span class="tag">#AI Security</span> <span class="tag">#Local GPU (LLaMA-8B)</span>

---

## Why This Research Matters for SSD Architecture

**SK hynix SSD Architecture팀의 목표:**
> *"SSD 아키텍처 설계 지식 DB 구축 + RAG 기반 Q&A 시스템 개발"*

<div class="box">
RAG 시스템이 내부 설계 문서를 참조할 때, <strong>민감한 기술 정보·IP가 의도치 않게 유출될 수 있습니다.</strong><br>
→ RAG 기반 Q&A 시스템을 실제로 구축하려면 <strong>보안과 신뢰성</strong>이 핵심 과제입니다.
</div>

**내 연구가 직접 연결되는 지점:**

| SK hynix SSD Architecture 직무 | 내 연구 경험 |
|-------------------------------|-------------|
| 최신 RAG 기술 분석 및 내부 도입 평가 | RAG 환경에서 LLM 동작 분석 및 실험 |
| Local GPU 환경에서 RAG 구현·평가 | LLaMA-3.1-8B, 2250샘플 실험 (A100) |
| AI 솔루션 개발 경험 | CNN, Transformer, Linear Probe 구현 |
| AI IDE 도구 활용 (Cursor, **Claude Code**) | **Claude Code로 전체 연구 진행 중** |

---

## Research Overview: PII Detection in RAG

**문제 정의:**

```
[User Query] "What is John's phone number?"
      ↓
[RAG System] Context: "John Smith, 010-1234-5678, john@email.com"
      ↓
[LLM Output] "John can be reached at 010-1234-5678..."  ← PII 유출
```

**핵심 가설:** LLM이 PII를 생성할 때 **토큰별 Shannon Entropy가 급격히 감소**한다

```
일반 텍스트:  entropy ≈ 3.0 nats  (다음 토큰 불확실 → 높은 entropy)
PII 토큰:    entropy ≈ 0.2 nats  (정해진 값 → 낮은 entropy)
```

**실험 환경:**
- Model: LLaMA-3.1-8B-Instruct (Local GPU, A100)
- Dataset: UnlearnPII (2,250개 가상 인물 프로필, 16가지 PII 필드)
- Tool: **Claude Code** (전체 실험 코드 개발 및 관리)

---

## Experiment Results

**2-Step Detection Pipeline:**

```
[Step 1] Entropy Sequence → CNN Classifier → "PII 의심 샘플"
[Step 2] Hidden State → Linear Probe → "PII 최종 확정"
```

### Step 1: Entropy-based Sequence Classifier (entropy만 사용)

| Method | F1 | Precision | Recall |
|--------|-----|-----------|--------|
| Threshold (M1: Adaptive ΔH) | 0.271 | 0.160 | 0.897 |
| Logistic Regression | 0.758 | 0.616 | 0.985 |
| **CNN-2ch (entropy + Δentropy)** | **0.951** | **0.933** | **0.970** |
| Transformer | 0.940 | 0.898 | 0.985 |

### Step 2: Linear Probe (hidden state 사용)

| Method | F1 | Precision | Recall |
|--------|-----|-----------|--------|
| Probe Only (hidden state) | 0.983 | 0.967 | 1.000 |
| Entropy + Probe (Pipeline) | 0.800 | 0.722 | 0.897 |

---

## Real-time Feasibility & Contribution

**실시간 탐지 가능성 실험 (Prefix-only CNN):**

| 관찰 토큰 수 (K) | F1 | PII 등장 대비 |
|----------------|-----|--------------|
| **5 tokens** | **0.921** | **PII 등장 4토큰 전 탐지** ✅ |
| 10 tokens | 0.951 | PII 등장 1토큰 후 |
| 전체 (80 tokens) | 0.796 | 생성 완료 후 |

→ **K=5 latency만으로 F1=0.92, PII 등장 전 선제 탐지 가능**

<br>

**SK hynix SSD Architecture RAG 시스템에 적용 시 기대 효과:**

<div class="box">
내부 설계 문서 기반 RAG Q&A 시스템에서 <strong>민감 정보 유출을 실시간으로 탐지</strong>하고,
경량 CNN (파라미터 ~157K)으로 추가 inference 비용 최소화하면서 보안성 확보
</div>

**Next:** Token-level PII localization → Vector Steering으로 생성 자체 억제