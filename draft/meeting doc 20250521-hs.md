* DataHub란?

  LinkedIn에서 수많은 데이터 파이프라인, 테이블, 대시보드가 산재되어 있는 환경에서 "지금 이 데이터가 뭔지, 누가 만들었고, 어디에 쓰이고 있는지" 파악하기 위해 DataHub를 구축

![image](https://github.com/user-attachments/assets/5477b9f4-052f-4d42-bb1f-1f849abf05f7)


* 예시: 서비스 구축 시 유기적 흐름

  목표: 분석가가 신뢰할 수 있는 ‘고객 이탈률 데이터셋’ 찾기
  
	step1. 검색: "고객 이탈" 관련 키워드로 검색 ⮕ 관련 테이블, dbt 모델, 대시보드가 카탈로그에서 검색됨
 
	step2. Lineage 확인: 해당 테이블이 어디서 왔고, 누가 가공했고, 어디에 사용되는지 확인 ⮕ 데이터 신뢰도 확보 가능
 
	step3. 프로파일링 확인: 컬럼 값 분포, null 비율 등 확인 ⮕ 분석 가능 여부 판단
 
	step4. 거버넌스 확인: 이 데이터에 **보안 정책(Personal Data 등)**이 적용되었는지 확인 ⮕ 활용 가능 여부 판단
 
	step5. 이력 확인: 최근 데이터 스키마가 변경되었는지 확인 ⮕ 분석 정확도 확인
 
    -> 결과적으로, 각 기능은 서로 연결되어 "데이터 신뢰성"과 "분석 효율"을 높이는 체계를 만들기



* 참고) dbt (data build tool)

  분석 엔지니어들이 SQL만으로도 데이터 변환(Transform)을 수행하고, 
  
  모델 버전 관리와 테스트를 하고,
  
  Lineage 추적을 하도록 도와주는 오픈소스 도구
  
  dbt는 T(Transform)에 집중한 ELT 구조에서 중간 단계 역할을 함
  
  함수로 파라미터를 넘겨서 사용하는 방법
  
  ![image](https://github.com/user-attachments/assets/278cfa1a-3f36-4ed4-a89d-dd0519afd415)



* 흐름도 
  ![image](https://github.com/user-attachments/assets/64393a2e-aa01-4191-95aa-cfef55b9c519)
  (추후 방법 모색: 리니지 -> 프로파일링/ 카탈로그 -> 프로파일링)


* Case1) 리니지 중심 시각화**

Kafka Topic (로그 수집) -> ETL Job (Airflow) -> 데이터 웨어하우스 테이블 -> dbt 모델 - BI 대시보드

  :: 이 흐름을 데이터 카탈로그 내부에서 시각화하고 추적할 수 있게 만드는 것이 중요할것으로 보임


* Case2) 도메인 기반 분류 체계 설정 (주제-업무 중심) 시각화 : 태깅처리 모색  (#3차 미팅준비)**

  **고객(Customer)**
  
  └─ 테이블: customer_info, churn_log
  
  └─ 대시보드: 고객 분석 KPI
  
  └─ 스트리밍: customer_event_topic


  **매출(Sales)**

  └─ 테이블: sales_summary

  └─ 대시보드: 일별 매출 리포트
  
  -> 다양한 유형 자산을 하나의 업무 맥락 내에서 탐색할 수 있어 사용자 경험도 좋아짐


* Lineage 사용 예

  BI 대시보드가 이상할 때 → "이 지표가 어떤 원천 테이블과 모델을 기반으로 만들어졌는지 추적" → 데이터 흐름 전체를 살펴봐서 문제 원인을 찾음


* 영향도 분석 사용 예
  
  테이블 컬럼 하나를 지우려고 할 때 → "이 컬럼이 쓰이는 downstream 모델이나 대시보드가 뭔지 알고 싶음" → 해당 컬럼을 참조하는 모든 요소를 목록으로 출력


 ![image](https://github.com/user-attachments/assets/0c7834b3-6fff-426f-8b16-b47730d83863)

 
* 예시 흐름 1

Kafka Topic (stream) -> Row Table (hive) -> dbt Model -> ML Feature Table -> ML Model Deployment



* 예시 흐름 2

PostgreSQL Table -> Airflow ETL Job -> DW Table -> BI Dashboard


* 적용 예상 흐름 

데이터 소스 (RDS, Kafka 등) --수집--> Trino + Airflow --변환, 적재--> S3 (저장소) --분석--> JupyterLab (모델분석) --결과 생성--> 보고서 (PDF, PPT, HTML 등) 


![image](https://github.com/user-attachments/assets/6cd22750-5d3f-422d-9462-0f463111612b)



* 참고
  
  먼저 Airflow/Trino → S3까지 Lineage 확보 → Jupyter 수동 연결

  DataHub Actions나 Slack 연동으로 메타데이터 변화 알림도 가능


  

