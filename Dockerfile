FROM python:3.10.11 AS builder
WORKDIR /usr/src/app
COPY requirement.txt ./
COPY main.py ./
COPY bow ./bow
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirement.txt
RUN pyarmor gen -r main.py && pyarmor gen -r -i bow

FROM python:3.10.11-slim
COPY msyh.ttc /usr/share/fonts/
WORKDIR /usr/src/app
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin/ /usr/local/bin/
COPY --from=builder /usr/src/app/dist .
COPY images ./images

CMD [ "python", "./main.py" ]