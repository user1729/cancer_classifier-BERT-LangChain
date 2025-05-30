#!/bin/bash

source init-docker.sh

docker build -t "${IMAGENAME}" -f Dockerfile .
