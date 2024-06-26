
## [3rd 프로젝트]: 네이버 쇼핑 리뷰 자연어 처리(NLP) 데이터 파이프라인 구축 
- 작성: 2022년 11월 3일   
- 데이터 분석 코드 파일: [ai_15_공주_s3_code.ipynb](https://github.com/kongju7/my_project3/blob/main/ai_15_%EA%B3%B5%EC%A3%BC_s3_code.ipynb)
- 웹 애플리케이션 파일: [s3_app 폴더](https://github.com/kongju7/my_project3/tree/main/s3_app)


### 프로젝트 개요

![데이터 파이프라인](/img/s3_data_pipeline.png "데이터 파이프라인")

1. 주제 및 데이터 소개: 쇼핑 리뷰 자연어 처리 감성 분석  
    - 쇼핑 리뷰 분석의 필요성 
      - 사람들의 구매 의사결정에 중요한 영향을 미치는 요인 
      - 고객들의 제품 (불)만족 정도나 요소 파악 
        → 제품 성능 향상 및 고객 성향에 맞는 맞춤형 정보 제공 가능  
       
    - 쇼핑 리뷰 분석의 어려움
      - 다양한 매체 및 형태, 방대한 양 축적 → 일일이 확인하고 분석하기 어려움
        - 홈페이지, 개인 블로그, 소셜 미디어(블로그, 인스타그램, 트위터 등)
      - 대부분이 일상적인 용어로 작성 → 컴퓨터가 바로 처리하기 어려움  
   
    ☞ 자연어 처리 기술을 통한 분석 필요 ☞ 감성 분석
  
    - 데이터 소개: 네이버 쇼핑 리뷰 
      - 데이터 특징
        - 변수: 별점 (1~5점), 리뷰 (*3점 리뷰 제외)
        - 데이터 건수: 20만 건 → 분석 사용 199,391건 
        - 언어: 한국어
        - 출처: 네이버 쇼핑(https://shopping.naver.com/)
        - 수집 기간: 2020.06~2020.07
 
      - 출처: https://github.com/bab2min/corpus/tree/master/sentiment

2. 데이터 저장: 로컬 데이터베이스 구축, MySQL 활용 
3. 데이터 분석: 네이버 쇼핑 리뷰 자연어 처리(NLP) 감성 분석 
    - 자연어 처리를 위한 전처리 
    - 딥러닝(Deep Learning) GRU 모델 활용 
      - Tensorflow API 활용 
      ☞ 최종 모델 테스트 정확도: 0.918
  
4. API 서비스 및 웹 애플리케이션 개발: Flask 활용 
    - 자연어 처리 모델 적용 API 서비스 개발 
      - 상품 리뷰를 입력 받아, 감성 분석 예측 결과 반환하도록 구현
    - 웹 애플리케이선 Front-end 구현 
      - 함수 처리를 위해 HTML에 Jinja template 적용 
      - 폰트 디자인 등을 위해 CSS 적용 
    
5. 대시보드 작성: Metabase 활용 (MySQL DB 연동)
  
<img src = "https://raw.githubusercontent.com/kongju7/my_project3/main/ai_15_%EA%B3%B5%EC%A3%BC_s3_dashboard.png" width="50%" height="50%">
  
6. 배포 및 향후 계획 
    - Heroku를 통한 배포 시도 → 실패 
      - 딥러닝 라이브러리 용량(500M+) 문제로 실패 
    - 네이버 Open API 활용 블로그 리뷰 스케쥴링 수집 → 활용 못함 
      - 블로그 글의 첫 부분 일부만 제공 → 리뷰 분석 자동화 구현 어려움 
    - 향후 계획: 상품 리뷰 수집 및 분석 자동화 완성 
