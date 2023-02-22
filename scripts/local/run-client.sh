#!/bin/bash


export $(cat ./config/local/client.conf | egrep -v "(^#.*|^$)" | xargs)
python3.10 -m source.client.main

