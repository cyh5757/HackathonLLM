
# 🚀 RAG 기반 과자 정보 검색 시스템
## 🚀 코드프레소 250329 팀 프로젝트

## 프로젝트 개요
이 프로젝트는 과자와 식품 첨가물에 대한 정보를 RAG(Retrieval-Augmented Generation) 기반으로 검색하고 답변하는 시스템입니다.
이 프로젝트는 [fastapi/full-stack-fastapi-template](https://github.com/tiangolo/full-stack-fastapi-template)를 기반으로 구성되었습니다.

## 주요 기능
- FastAPI 기반 백엔드 서버
- PostgreSQL + pgvector를 활용한 벡터 데이터베이스
- LangChain을 활용한 AI 에이전트 구현
- Chainlit을 활용한 대화형 인터페이스

- 과자 정보 벡터 DB 저장 및 검색
- 식품 첨가물 정보 검색
- 알레르기 정보 제공
- 실시간 질의응답

## 기술 스택
- **백엔드**: FastAPI, PostgreSQL, pgvector
- **AI/ML**: LangChain, OpenAI API
- **개발 도구**: Docker, Chainlit
- **기타**: Python 3.12+

## 프로젝트 구조


# 📦 프로젝트 세트와 실행 방법

## 1. Docker로 FastAPI 서버 생성 및 실행

**Docker로 백업드(Postgres + FastAPI) 서버를 실행합니다.**

### 1-1. Docker 환경 준비
- Docker Desktop 설치 필요
- `docker-compose.yml`, `Dockerfile`이 준비되어 있어야 합니다.

### 1-2. Docker 서버 실행 명령어

```bash
# 루트 디렉토리 (docker-compose.yml이 있는 포범) 에서 실행
docker-compose up --build
```

- Postgres + FastAPI API 서버가 자동으로 실행됩니다.
- FastAPI 서버는 `http://localhost:8000`에서 접속할 수 있습니다.
- Swagger 문서: `http://localhost:8000/docs`

### 1-3. 콘테이너 설명
- `postgres-pgvector`: Postgres + pgvector 확장 설치된 DB 콘테이너
- `snack-api`: FastAPI 앱이 실행되는 콘테이너

## 2. 로컬(.venv) 환경에서 Chainlit 사용 방법

Chainlit은 **로컬 개발용**입니다. (FastAPI 서버는 Docker에서 따뜻도로 통신)

### 2-1. 가상환경 생성 및 활성화

```bash
# Python 3.12 기준
python3.12 -m venv .venv
.venv\Scripts\activate  # 윈도우
# 또는
source .venv/bin/activate  # 맵/리누스
```

### 2-2. 패키지 설치

```bash
# 의존성 설치
pip install --upgrade pip
pip install uv
uv sync
```

**다시: 새로운 패키지 추가시**

```bash
uv add 패키지명
```

### 2-3. Chainlit 앱 실행

```bash
# .venv 활성화 후
# scripts 폴더의 파일은 개별적으로 진행함.
set PYTHONPATH=.
chainlit run scripts/chainlit_app.py --port 4785
# 아래 처럼 사용하는 걸 권장함 port 번호는 겹치지 안흔 것으로 사용
chainlit run chainlit/main.py --port 8501
```

- Chainlit 웹은 `http://localhost:4785`에서 접속합니다.
- 백업드는 Docker로 따뜻되는 FastAPI 서버(`http://localhost:8000`)를 가리고 통신합니다.

---

# 📋 참고

- **환경변수 설정**  
  `.env` 파일이 필요하며, 예시로는 다음을 포함합니다:

```env
POSTGRES_USER=postgres
POSTGRES_PASSWORD=123123
POSTGRES_DB=test
OPENAI_API_KEY=your-openai-key
DATABASE_URL=postgresql+asyncpg://postgres:123123@localhost:5432/test
```

- **주의사항**
  - Chainlit은 자체 DB 연결 기능이 있는데, 언제든 에러를 피하려면 `.chainlit/config.toml`에서 DB 연결 기능을 비활성화 하는 것을 권장합니다.
  - FastAPI 서버를 따뜻 통신해야 Chainlit과 정적으로 관리됩니다.

---

# ✨ 요약 정리

1. Docker로 FastAPI + Postgres 서버를 먼저 띄운다.
2. 로컬(.venv) 환경에서 Chainlit을 실행한다.
3. Chainlit은 FastAPI 서버를 바라보며 실시간으로 답변을 받는다.

---

(추가로 docker-compose 실행시 주의사항이나 Chainlit 환경변수 세팅 포함된 버전을 웹역을 간단하게 변경해주기를 원하면 다시 요청해주세요!)



## 🏆 해커톤 및 결과 이미지

![첫 결과 이미지](images\frist.png)

![변경 이미지](images\last.JPG)


