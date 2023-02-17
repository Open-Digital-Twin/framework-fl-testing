#!/bin/sh

# Where $ENVSUBS is whatever command you are looking to run
$ENVSUBS < file1 > file2



# This will exec the CMD from your Dockerfile, i.e. "npm start"
#exec "$@"