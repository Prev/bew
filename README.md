# bew
Back End Work  
SWMaestro 3번째 기초기술과제 (백엔드/머신러닝) 프로젝트입니다.

  
  

## 목표
`[4%즉시할인쿠폰]Intel Pentium G3258 3.2 GHz Processor BX/329456` 같은 상품 제목을 보고  
`디지털/가전;PC부품;CPU`형식의 카테고리를 유추하는 프로그램을 머신러닝을 이용하여 제작
   

## 사용 라이브러리
- scikit-learn
- numpy 
- konlpy
- flask

  
  

## 분류법
- scikit-learn 내부의 SVM(Support Vector Machine) 사용한 분류
- konlpy를 활용한 한글 형태소 분석 (명사 추출, Mecab)
- 단어의 갯수는 중요하지 않음 (2개 이상의 단어가 쓰였다고 중요하다는 보장 X) -> 행렬의 최대값 1로 고정
- 영어-한글 띄어쓰기 없이 이어져있는 단어 분리하여 목록에 추가
- 연이어 쓰여있는 영단어 대소문자 기준으로 분리하여 추가 (ex. TouchScreen -> Touch, Screen)
- 무의미하게 반복되는 단어들 (ex. 현대백화점) 무시 (CountVectorizer의 max_df를 0.15로 설정)
- 제품 시리얼번호의 경우 앞부분을 보고 공통으로 처리 (ex. TZ-621과 TZ-541는 같은 카테고리일 가능성이 높음 -> TZ_Series로 치환)

  
  

## 버전별 정확도

- v1.1
  - Score 	0.698979591837

- v1.2 
  - Local	0.7042
  - Score 	0.70612244898

-  v1.3
  - Local	0.7033
  - Score 	0.70387755102

- v1.4
  - Local	0.7078
  - Score 	0.710612244898

- v1.5
  - Local	0.7078
  - Score 	0.723673469388

- v1.6
  -	Local	0.7088 (logspace20 => 0.7085)
  -	Score 	0.72387755102