FROM python:3.12

COPY requirements.txt .

RUN set -ex \
    && apt update \
    && apt install \
    && pip install -r requirements.txt --no-cache-dir \
    && apt --yes autoremove \
    && apt --yes clean \
    && groupadd -g 1001 python \
    && useradd --no-log-init -u 1001 -g python python

RUN mkdir -p /home/python/ && chown -R python:python /home/python/
WORKDIR /home/python/app/
COPY --chown=python:python . .

USER python

CMD ["python", "main.py"]