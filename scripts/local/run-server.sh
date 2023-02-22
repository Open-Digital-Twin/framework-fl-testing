#!/bin/bash


export $(cat ./config/local/server.conf | egrep -v "(^#.*|^$)" | xargs)
python3.10 -m source.server.main

