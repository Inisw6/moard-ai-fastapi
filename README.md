# Moard API

FastAPI를 사용한 백엔드 API 서버입니다.

## 기술 스택

- FastAPI
- Supabase
- Python 3.8+

## 시작하기

### 1. 가상환경 설정

```bash
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 환경 변수 설정

`.env` 파일을 프로젝트 루트 디렉토리에 생성하고 다음 변수들을 설정합니다:

```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

### 4. 서버 실행

```bash
uvicorn main:app --reload
```

서버가 실행되면 다음 URL에서 API 문서를 확인할 수 있습니다:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 프로젝트 구조

```
moard-api/
├── api/            # API 라우터
├── models/         # 데이터 모델
├── services/       # 비즈니스 로직
├── main.py         # 애플리케이션 진입점
├── requirements.txt # 의존성 목록
└── .env           # 환경 변수 (git에 포함되지 않음)
```

## API 엔드포인트

- `/api/v1/...` - API 엔드포인트들