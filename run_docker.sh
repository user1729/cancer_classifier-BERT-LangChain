#!/bin/bash


source init-docker.sh

docker run --runtime=nvidia \
  --mount type=bind,source=${PWD},target=/home/developer/ \
  -w /home/developer/ \
  --rm --ipc=host -it \
  --name research-dev \
  ${IMAGENAME}
