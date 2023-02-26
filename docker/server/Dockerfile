FROM python:3.10



ADD ./docker/server/requirements.txt /
RUN pip install --no-cache-dir -r requirements.txt


ADD ./server/src /server
ADD ./.cache/certificates .cache/certificates

EXPOSE 8080


CMD [ "python","-u","-m","server.main" ]
