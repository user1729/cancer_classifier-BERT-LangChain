#!/bin/bash


source init-docker.sh

docker run --runtime=nvidia \
  --mount type=bind,source=${DATA_VOLM},target=/home/developer/ \
  -w /home/developer/ \
  --rm --ipc=host -it \
  --name research-dev \
  ${IMAGENAME}


#-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
# --ulimit memlock=-1 \
#--shm-size=16g \
#  -p 8888:8888 -p 8080:8080 \
#  -v /path/to/code:/home/developer/code \
#  -v /home/you/training_logs:/home/developer/logs \
## Forward remote Jupyter and VS Code ports to your local machine
#ssh -L 8888:localhost:8888 -L 8080:localhost:8080 username@your.server.ip
