#!/bin/bash
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

if [ -z "$1" ]; then
    CONFIG_FILE=${SCRIPTPATH}/client.conf
else
    CONFIG_FILE=$1
fi




export $(cat $CONFIG_FILE | egrep -v "(^#.*|^$)" | xargs)
(cd $SCRIPTPATH/../ && python3.10 -m client.src.main)

