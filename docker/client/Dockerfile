FROM python:3.10

WORKDIR /

ADD ./docker/client/requirements.txt /
RUN pip install --no-cache-dir -r requirements.txt


ADD ./client/src /client
ADD ./.cache/certificates .cache/certificates



CMD [ "python","-m","client.main" ]
