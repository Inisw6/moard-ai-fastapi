# moard-model-server

## 개요
이 프로젝트는 **강화학습 기반의 추천 시스템**을 FastAPI를 통해 서비스하는 프로젝트입니다. 
Dueling DQN(Deep Q-Network) 알고리즘의 베이스 모델을 토대로 **Online Learning**하고, 이를 토대로 추천 서비스를 REST API를 통해 제공합니다.

## 환경
- Python 3.10
- FastAPI
- PyTorch
- Redis
- MySQL
- Docker & Docker Compose

## 실행방법

### Docker를 사용한 실행
```bash
# Docker Compose로 서비스 실행
docker-compose up --build

# 백그라운드에서 실행
docker-compose up -d
```

### 로컬 환경에서 실행
```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt

# 서버 실행
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 프로젝트 구조
```
moard-model-server/
├── app/                           # 메인 애플리케이션 디렉토리
│   ├── api/                       # API 엔드포인트 정의
│   │   ├── endpoints/             # API 엔드포인트 구현
│   │   │   ├── predict.py         # 예측 관련 엔드포인트
│   │   │   ├── embedding.py       # 임베딩 관련 엔드포인트
│   │   │   ├── model.py           # 모델 관리 엔드포인트
│   │   │   └── online_learning.py # 온라인 학습 엔드포인트
│   │   └── router.py              # API 라우터 설정
│   │
│   ├── core/                      # 핵심 설정 및 유틸리티
│   │   ├── config.py              # 환경 설정
│   │   └── database.py            # 데이터베이스 설정
│   │
│   ├── ml/                        # 머신러닝 관련 코드
│   │   ├── online_learning.py     # 온라인 학습 구현
│   │   ├── q_network.py           # Q-Network 모델 구현
│   │   └── doc2vec_embedder.py    # 문서 임베딩 구현
│   │
│   ├── models/                    # 데이터베이스 모델
│   │   └── orm.py                 # SQLAlchemy ORM 모델 정의
│   │
│   ├── schemas/                   # Pydantic 스키마
│   └── services/                  # 비즈니스 로직
│
├── dqn/                           # DQN 알고리즘 구현
├── models/                        # 학습된 모델 저장
├── Dockerfile                     # Docker 이미지 설정
├── docker-compose.yml             # Docker Compose 설정
└── requirements.txt               # Python 의존성
```

## 주요 컴포넌트 설명

### API 엔드포인트

#### 1. 예측 API (`/api/v1/predict`)
- `POST /bulk`: 사용자 임베딩과 콘텐츠 임베딩 리스트로부터 Q-Value 예측
- `POST /top-contents`: 사용자 ID와 콘텐츠 ID 리스트를 받아 상위 콘텐츠 추천

#### 2. 임베딩 API (`/api/v1/embedding`)
- `POST /doc2vec`: 단일 콘텐츠의 Doc2Vec 임베딩 생성
- `POST /doc2vec/bulk`: 여러 콘텐츠의 임베딩을 일괄 생성

#### 3. 모델 관리 API (`/api/v1/model`)
- `GET /list`: 사용 가능한 모델 목록 조회
- `POST /change`: 현재 사용 중인 모델 변경
- `DELETE /{model_path}`: 저장된 모델 삭제

#### 4. 온라인 학습 API (`/api/v1/online-learning`)
- `GET /tasks`: 모든 학습 작업의 상태 조회
- `POST /async-train`: 비동기 모델 학습 시작
- `GET /async-batch-status/{task_id}`: 비동기 배치 처리 작업 상태 조회
- `DELETE /async-batch-status/{task_id}`: 비동기 작업 상태 정보 삭제
- `POST /collect-training-data`: 학습 데이터 수집
- `POST /train`: 수집된 데이터로 모델 학습

### 핵심 기능
1. **Dueling DQN 구현**
   - Deep Q-Network 알고리즘 기반의 강화학습
   - PyTorch 기반의 신경망 모델
   - 온라인 학습 지원

2. **문서 임베딩**
   - Doc2Vec 기반의 문서 임베딩
   - 추천 시스템을 위한 아이템 표현

3. **FastAPI 서버**
   - 비동기 처리 지원
   - 자동 API 문서화 (Swagger UI)
   - Redis를 통한 캐싱

4. **데이터베이스**
   - MySQL 데이터베이스 연동
   - SQLAlchemy ORM을 통한 데이터 관리

5. **캐싱 시스템**
   - Redis를 통한 고성능 캐싱
   - 비동기 작업 관련 저장

## API 문서
서버가 실행되면 다음 URL에서 API 문서를 확인할 수 있습니다:
- Swagger UI: `http://localhost:8000/docs`
