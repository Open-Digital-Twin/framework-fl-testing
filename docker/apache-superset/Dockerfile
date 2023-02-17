FROM apache/superset
# Switching to root to install the required packages
USER root
COPY ./docker/apache-superset/entrypoint.sh /
RUN chmod 777 /entrypoint.sh
USER superset

RUN superset fab create-admin \
              --username admin \
              --firstname Superset \
              --lastname Admin \
              --email admin@superset.com \
              --password admin
RUN superset db upgrade
RUN superset init

# RUN ["/entrypoint.sh"]
# ["superset", "init"]
