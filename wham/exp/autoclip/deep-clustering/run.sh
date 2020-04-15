#!/bin/sh
tsp ./allocate.py 1 python main.py -exp wham/exp/autoclip/run6/config.gin all
echo Queuing tsp ./allocate.py 1 python main.py -exp wham/exp/autoclip/run6/config.gin all and sleeping while it starts...
sleep 30