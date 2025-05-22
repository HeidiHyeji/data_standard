참고자료) https://velog.io/@dohee1121/Text-Generation-Metric%EC%9E%90%EC%97%B0%EC%96%B4-%EC%83%9D%EC%84%B1-%ED%8F%89%EA%B0%80-%EC%A7%80%ED%91%9C

## 📌 1. ROUGE란 무엇인가?

**ROUGE**(Recall-Oriented Understudy for Gisting Evaluation)는 **기계가 생성한 텍스트(요약문 등)가 사람이 만든 참조 텍스트(정답)에 얼마나 가까운지 평가하는 대표적인 지표**

주로 **요약**이나 **생성형 모델** 평가에서 활용.

### 📌 주요 ROUGE 지표 유형:

* **ROUGE-N** (ROUGE-1, ROUGE-2):

  * N-gram 기반 지표로서, N개의 연속된 단어 조합을 기준으로 정확도를 측정합니다.
  * **ROUGE-1**: unigram(단어 한 개씩) 기준
  * **ROUGE-2**: bigram(단어 두 개씩) 기준

* **ROUGE-L**:

  * 가장 긴 공통 부분 수열 (**Longest Common Subsequence, LCS**)을 기반으로 측정.
  * 생성된 문장과 참조 문장의 전체적인 흐름 및 일관성 평가에 적합.

---

### 📍 ROUGE 계산 방법 예시 (직관적 이해):

* **정답 문장 (참조)**:

  > "나는 오늘 저녁에 치킨과 피자를 먹었다."

* **모델이 생성한 문장**:

  > "오늘 저녁에 나는 피자와 치킨을 먹었다."

이 두 문장의 ROUGE 점수는 매우 높습니다. 단어와 문장 구조가 유사하기 때문입니다.

* ROUGE-1: unigram 수준에서 단어 중복을 평가합니다.
* ROUGE-2: bigram 수준에서 연속된 두 단어의 중복을 평가합니다.
* ROUGE-L: 문장 흐름에서 공통된 순서를 평가합니다.

---

## 🛠️ ROUGE 사용법 (Python 예제):

```python
from rouge_score import rouge_scorer

reference = "나는 오늘 저녁에 치킨과 피자를 먹었다."
hypothesis = "오늘 저녁에 나는 피자와 치킨을 먹었다."

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(reference, hypothesis)

print(scores)
```

출력 예시:

```json
{
 'rouge1': Score(precision=1.0, recall=1.0, fmeasure=1.0),
 'rouge2': Score(precision=0.5, recall=0.5, fmeasure=0.5),
 'rougeL': Score(precision=0.857, recall=0.857, fmeasure=0.857)
}
```

---

## 🔍 2. ROUGE와 혼합하여 사용할 수 있는 다른 평가 지표 추천

ROUGE는 주로 recall 지향적이기 때문에, 다른 지표를 추가적으로 혼합하여 더욱 견고하게 평가하는 것이 좋습니다.

| 지표명           | 특징                                       | 주 용도          |
| ------------- | ---------------------------------------- | ------------- |
| **BLEU**      | 참조문과의 정확한 단어 순서를 중심으로 평가. Precision 지향적. | 기계번역 및 생성형 평가 |
| **METEOR**    | 동의어와 어근(stemming) 고려하여 더 유연하게 평가         | 텍스트 생성 모델 평가  |
| **BERTScore** | BERT 기반 문맥 임베딩을 사용하여 의미적으로 평가            | 문장 수준 의미 평가   |

---

### 📌 BLEU (BiLingual Evaluation Understudy)

* N-gram Precision 기반 지표
* 기계 번역 및 생성형 텍스트 평가에서 가장 유명한 지표 중 하나
* **장점**: 단어 순서에 민감하여 정확한 표현 평가 가능
* **단점**: 문장 길이나 의미적 유사성 측면에서의 평가가 부족할 수 있음

#### 예제 코드:

```python
import nltk
reference = ["나는 오늘 저녁에 치킨과 피자를 먹었다.".split()]
candidate = "오늘 저녁에 나는 피자와 치킨을 먹었다.".split()

bleu_score = nltk.translate.bleu_score.sentence_bleu(reference, candidate)
print(f"BLEU Score: {bleu_score:.2f}")
```

---

### 📌 METEOR (Metric for Evaluation of Translation with Explicit ORdering)

* Precision과 Recall을 모두 반영
* **동의어(synonyms)**, **어근(stemming)** 및 **의미적 유사성** 고려
* BLEU나 ROUGE보다 인간 평가와 더 높은 상관관계

#### 예제 코드:

```python
from nltk.translate.meteor_score import meteor_score

reference = "나는 오늘 저녁에 치킨과 피자를 먹었다."
candidate = "오늘 저녁에 나는 피자와 치킨을 먹었다."

score = meteor_score([reference], candidate)
print(f"METEOR Score: {score:.2f}")
```

---

### 📌 BERTScore (Semantic Similarity)

* 최신 평가 지표로 BERT와 같은 사전 학습된 모델의 Embedding을 이용
* 문장의 의미적, 맥락적 유사성 평가에 매우 강력한 지표
* 문맥과 의미적 유사성에 민감하고, 최근 많은 연구에서 사용

#### 예제 코드:

```python
import bert_score

reference = ["나는 오늘 저녁에 치킨과 피자를 먹었다."]
candidate = ["오늘 저녁에 나는 피자와 치킨을 먹었다."]

P, R, F1 = bert_score.score(candidate, reference, lang='ko', model_type='bert-base-multilingual-cased')
print(f"BERTScore - Precision: {P.mean():.2f}, Recall: {R.mean():.2f}, F1: {F1.mean():.2f}")
```

---

## 🎯 권장하는 평가 지표 혼합 사용 예시 (실제 PoC를 위한 최적 조합):

* **ROUGE-L**: 전체 문장 흐름과 일관성 평가
* **BLEU**: 정확한 단어 순서 평가
* **METEOR**: 유연한 단어 평가 및 의미 유사성 평가
* **BERTScore**: 심도 있는 의미적 일관성 평가


---

## ✅ 최종 권장 평가 지표 조합:

| 지표 조합                             | 평가 대상             | 장점                              |
| --------------------------------- | ----------------- | ------------------------------- |
| ROUGE + BLEU + METEOR + BERTScore | 생성형 모델 (GPT기반) 평가 | 단어 정확도, 흐름, 의미적 유사성 모두 균형 있게 평가 |


