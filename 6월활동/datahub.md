## 2. 비즈니스 분류 및 태그 관리

### 기능
- 태그는 검색 및 발견에 도움
- 데이터세트, 데이터세트 스키마 또는 컨테이너에 추가
- 비즈니스 용어집이나 어휘와 연관시키지 않고도 엔티티에 레이블을 지정하거나 분류 (테이블별/스키마별)

### 태그 활용
- **Web SQL**을 통하여 co-worker가 같은 데이터셋을 쿼리할 수 있도록 태깅 검색기능 사용
- 프로젝트에 자산으로 매핑할 수 있도록 태깅 검색기능 사용

### SMARTLLION
- **새 태그 자동완성 기능**: 이미 등록된 태그를 토대로 데이터셋에 추천하여 표기
- **태그 기반 업무영역 자동 분류 기능**: 스키마 기준으로 관리

---

## 3. 용어사전 자동추천 (베타)

### 기능
- **LLM 사용**: 테이블과 컬럼에 대한 용어 제안
- 비즈니스 용어(도메인)와 호환
- 사용자 수정 기능 제공
- 기존 사전 기반: 어디에서 사용되던 용어를 추천한 건지 정보 제공

---

## 4. 용어사전 자동 리니지

### 기능
- 기존 영향도 파악과 동일
- 자동/수동 리니지
- 용어를 모든 하위 리니지에 전파
- 테이블, 뷰에만 지원되며 대시보드, 파이프라인, 데이터 작업을 포함한 다른 자산 유형에는 전파되지 않음

---

## 커넥션

### 보유 DBMS
- YB, Hive, BigQuery, Airflow, Vertica >>> 전체 가능

### Airflow
#### DataHub Airflow 플러그인 지원 내용
- **지원 버전**
  - Approach: plugin v2
  - Airflow Versions: 2.5 (2.5-2.8 더 이상 지원하지 않음: 리니지 기능/오류알람 기능 없음)

- **지원 내용**
  - 다양한 연산자(예: SQL 연산자(PostgresOperator, SnowflakeOperator, BigQueryInsertJobOperator 등), S3FileTransformOperator 및 기타)에서 자동으로 열 수준 계보를 추출
  - Airflow DAG 및 속성, 소유권, 태그를 포함한 작업
  - 작업 성공 및 실패를 포함한 작업 실행 정보
  - Airflow 연산자를 사용하여 수동 계보 주석을 `inlets`, `outlets` 추가
  - 자동 리니지 추출 지원 (PostgresOperator, BigQueryOperator, TrinoOperator)
  - 수동 리니지 추출 지원 (`inlets`, `outlets` Airflow 연산자 설정으로 리니지에 수동으로 주석 추가)
  - 더 이상 사용되지 않은 파이프라인 작업 정리 (DAG에서 제거 후): Cluster들이 DAG 수집하면서 DataHub에서 설정된 값을 기준으로 오래된 파이프라인과 작업을 제거

### Hive
#### DataHub Airflow 플러그인 지원 내용
- **지원 내용**
  - 다양한 연산자(예: SQL 연산자(PostgresOperator, SnowflakeOperator, BigQueryInsertJobOperator 등), S3FileTransformOperator 및 기타)에서 자동으로 열 수준 계보를 추출
  - Airflow DAG 및 속성, 소유권, 태그를 포함한 작업
  - 작업 성공 및 실패를 포함한 작업 실행 정보
  - Airflow 연산자를 사용하여 수동 계보 주석을 `inlets`, `outlets` 추가

---

## 정책 관리

### 플랫폼 정책
#### 권한 내용
1. 사용자 및 그룹 관리
2. DataHub 분석 페이지 보기
3. 정책 자체 관리

#### 정책 구분
1. **행위자**: 정책이 적용되는 사람(사용자 또는 그룹)
2. **권한**: 액터에게 할당해야 하는 권한(예: "플레이그라운드 보기")

### 메타데이터 정책
#### 권한 내용
1. 누가 데이터 세트 문서와 링크를 편집할 수 있나요?
2. 차트에 소유자를 추가할 수 있는 사람은 누구인가요?
3. 대시보드에 태그를 추가할 수 있는 사람은 누구인가요? 등

#### 정책 구분
1. **리소스**: '어떤'. 정책이 적용되는 리소스(예: "모든 데이터 세트").
2. **권한**: '무엇'. 정책에 의해 허용되는 작업(예: "태그 추가").
3. **행위자**: '누구'. 정책이 적용되는 특정 사용자, 그룹.

#### 리소스는 다양한 방법으로 정책과 연관될 수 있음
1. 리소스 유형 목록 - 엔터티 유형 (예: 데이터 세트, 차트, 대시보드)
2. 리소스 URN 목록
3. 태그 목록
4. 도메인 목록