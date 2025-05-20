아래 코드 및 명령어를 순서대로 Google Colab에서 실행하시면 됩니다.

---

## 🚀 ① 라이브러리 설치 (Colab용)

Colab 환경에서 다음과 같이 필수 라이브러리를 설치하세요:

```bash
!pip install openai faiss-cpu numpy
```

---

## 🚀 ② OpenAI API 설정

본인의 OpenAI API 키로 설정하세요.

```python
import openai
openai.api_key = "여기에_당신의_OpenAI_API_키"
```

---

## 🚀 ③ 표준 데이터 임베딩 및 추천 예제

다음 코드를 Colab에서 순차적으로 실행하면 됩니다.

```python
import openai
import faiss
import numpy as np

# OpenAI API 키 설정
openai.api_key = "여기에_당신의_OpenAI_API_키"

# 표준 데이터 메타데이터 예시
standard_metadata = [
    "고객_아이디 (Customer ID, 정수형)",
    "구매_날짜 (Purchase Date, YYYY-MM-DD)",
    "상품_코드 (Product Code, 문자열)"
]

# OpenAI 임베딩 함수 정의
def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response['data'][0]['embedding'])

# 표준 데이터 임베딩
standard_embeddings = np.array([get_embedding(text) for text in standard_metadata])

# FAISS 인덱스 생성 및 데이터 추가
dimension = standard_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(standard_embeddings)

# 사용자 데이터 메타데이터 예시 (비표준)
user_metadata = "고객번호 (CustomerNumber, 숫자형)"

# 사용자 데이터 임베딩
user_embedding = get_embedding(user_metadata).reshape(1, -1)

# 추천 표준 데이터 찾기 (가장 유사한 데이터 1개 추천)
distances, indices = index.search(user_embedding, k=1)
recommended_standard = standard_metadata[indices[0][0]]

print("🎯 추천된 표준 데이터:", recommended_standard)
```

---

## 🚀 ④ ChatGPT API를 활용한 모델 성능 평가 예제

다음 코드는 표준 데이터 여부에 따른 GPT 응답 차이를 확인하는 코드입니다.

```python
# ChatGPT 호출 함수
def get_gpt_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# 표준화된 데이터 프롬프트 예시
standard_prompt = """
아래 표준화된 데이터를 보고 고객의 최근 구매일자를 알려주세요:

고객_아이디: 123456
구매_날짜: 2024-04-15
"""

# 비표준화된 데이터 프롬프트 예시
non_standard_prompt = """
아래 데이터를 보고 고객의 최근 구매일자를 알려주세요:

고객번호: '123456번'
최근 구매: '15/04/2024'
"""

# ChatGPT 응답 비교
standard_response = get_gpt_response(standard_prompt)
non_standard_response = get_gpt_response(non_standard_prompt)

print("\n✅ 표준 데이터에 대한 GPT 응답:\n", standard_response)
print("\n⚠️ 비표준 데이터에 대한 GPT 응답:\n", non_standard_response)
```

---

## 🚀 ⑤ ROUGE를 이용한 응답 평가 (GPT 응답 평가)

ROUGE 지표로 GPT 응답 품질을 평가합니다.

```bash
!pip install rouge_score
```

```python
from rouge_score import rouge_scorer

reference_answer = "고객의 최근 구매일자는 2024년 4월 15일입니다."

# ROUGE 점수 계산
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

standard_scores = scorer.score(reference_answer, standard_response)
non_standard_scores = scorer.score(reference_answer, non_standard_response)

print("\n📌 표준 데이터 기반 응답 ROUGE 점수:\n", standard_scores)
print("\n📌 비표준 데이터 기반 응답 ROUGE 점수:\n", non_standard_scores)
```

---

## 🚩 (선택) Python 버전 확인 및 변경 명령어 (Colab 기본환경)

```bash
# Colab에서 기본적으로 Python 3.10 이상이 설치됨
!python --version
```

Colab 환경은 기본적으로 이미 Python이 설치되어 있으므로 별도의 Python 설치 명령어는 필요하지 않습니다.

---

## 📝 실행 순서 정리 (요약)

| 순서 | 실행 코드                                | 설명             |
| -- | ------------------------------------ | -------------- |
| 1  | `pip install openai faiss-cpu numpy` | 환경 설치          |
| 2  | OpenAI API 키 설정                      | API 인증 설정      |
| 3  | 표준 데이터 추천 코드                         | 임베딩 생성 및 추천    |
| 4  | GPT API 모델 성능 평가 코드                  | GPT 호출하여 응답 비교 |
| 5  | ROUGE 평가 코드                          | 응답 평가          |

위 내용을 순서대로 실행하면,
실제 **추천 알고리즘 구현 및 GPT 기반 모델 성능 평가 PoC**를 명확하게 실행해 볼 수 있습니다.

추가적인 실행 문제나 코드 수정이 필요하면 알려주세요!
