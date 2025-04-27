# snack

코드프레소 250329 팀 프로젝트

> 이
> 프로젝트는 [fastapi/full-stack-fastapi-template](https://github.com/fastapi/full-stack-fastapi-template/tree/master/backend)를
> 기반으로 구성되었습니다.

## 프로젝트 세팅

- python 3.12 이상
- 의존성은 `uv` 사용해서 pyproject.toml을 읽어 uv.lock 생성.
- PGVector 사용시 `sql/pgvector.sql` 테이블 정의 필요

가상환경 활성화 (비활성화 명령어: `deactivate` )

```shell
python3.12 -m venv .venv
source .venv/bin/activate
(.venv) $
```

`- (pip config set global.index-url https://pypi.org/simple`)

```shell
pip install --upgrade pip
pip install uv
```
패키지 추가는 이렇게
```
uv add asyncpg
uv pip install asyncpg # 직접 설치

```

```shell
# 패키지 추가 관련 명령어 (ex. asyncpg)
uv sync # uv.lock 읽어 패키지 추가, 없으면 프로젝트를 돌면서 uv.lock 생성
```

## 로컬 세팅

### 1. 패키지 설치 (패키지 업데이트시 uv sync 실행 필요)

 ```shell
uv sync
 ```

### 2. `fastapi` 서버 실행

Postgres DB 정보를 로컬 DB로 변경하고, `OPENAPI_API_KEY` 도 추가해주어야 합니다.

```shell
(.venv) $ fastapi run --workers 2 app/main.py
```

```shell
# 로컬에 있는 .env 파일 사용 시 `-m dotenv run --file /path/to/.env`
(.venv) $ python -m dotenv run -- fastapi run --workers 2 app/main.py
```

서버는 8000 포트에서 실행됩니다. Swagger 문서는 `http://localhost:8000/docs` 에서 확인할 수 있습니다.
