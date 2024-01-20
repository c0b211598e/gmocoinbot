#!/bin/sh
docker run -it --rm --shm-size 2g -v $PWD:/work -w /work rogpeter/gmocoinbot-image python src/start_all_bots.py