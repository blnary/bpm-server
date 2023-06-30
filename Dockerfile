FROM python:3.11.3

WORKDIR /srv

RUN python3 -m pip config set global.index-url http://10.249.181.72:8888/mirrors_bfsu_edu_cn/pypi/web/simple
RUN python3 -m pip config set global.trusted_host 10.249.181.72:8888
RUN python3 -m pip install --user pipenv

COPY ./Pipfile ./
COPY ./Pipfile.lock ./

RUN python3 -m pipenv install

COPY ./main.py ./

CMD ["python3", "-m", "pipenv", "run", "uvicorn", "--host", "0.0.0.0", "main:app"]
