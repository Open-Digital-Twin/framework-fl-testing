FROM apache/superset
# Switching to root to install the required packages


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
