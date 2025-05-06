# Deep_Learning_Project_2

- 본 프로젝트는 프로젝트는 자연어 처리(NLP)를 중심으로, 주어진 단어로 시작하는 시를 생성하는 것을 목표로 하였습니다.
- 저는 두 가지 버전을 업로드했으며, 하나는 LSTM 모델을 사용한 것이고, 다른 하나는 KOGPT2 모델을 활용한 것입니다.
- LSTM 모델은 make_poetry 파일에 포함되어 있으며, 웹 크롤링을 위한 코드도 함께 포함되어 있습니다.

## 🔧 사용한 기술

- Python
- Numpy
- konlpy
- Pytorch
- transformers
- sklearn

## 📂 활용 데이터셋 출처

- EBS 현대시 100선 + 행복과 관련된 시 약 10편 (파일 미첨부)
- 시 모음 사이트 (https://raincat.com/)

## 📂 파일 설명 

- make_poetry.ipynb : LSTM 모델 기반 시 생성기
- train.py : KoGPT2 모델 기반 시 생성기 학습 파일 (fine-tuning)
- inference.inynb : KoGPT2 모델 기반 시 생성기
- crawling_poetry.ipynb : 시 모음 사이트에서 시 크롤링하는 코드

## 🎯 프로젝트 목표 

- Matplotlib 라이브러리를 활용한 시각화
- 딥러닝 / 전이학습을 활용한 시 생성 모델 만들기
- Ai 발전에 따른 문학의 변화에 대한 고찰
