FROM python:3.11-buster

ENV PYTHONUNBUFFERED=1

WORKDIR /src

#pip로 poetry 설치
RUN pip install "poetry==1.6.1"

#poetry의 정의 파일 복사(존재하는 경우)
COPY pyproject.toml* poetry.lock* ./
COPY api api
COPY translate.py ./

#poetry로 라이브러리 설치(pyproject.toml이 이미 존재하는 경우)
RUN poetry config installer.parallel false
RUN poetry config virtualenvs.create false
RUN if [ -f pyproject.toml ]; then poetry install --no-root; fi

EXPOSE 8000

# uvicorn 서버 실행
ENTRYPOINT [ "poetry", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0"]
