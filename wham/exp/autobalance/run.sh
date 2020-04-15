#!/bin/sh
tsp ./allocate.py 1 python main.py -exp wham/exp/autobalance/run0:clip_percentile_10/config.gin all
echo Queuing tsp ./allocate.py 1 python main.py -exp wham/exp/autobalance/run0:clip_percentile_10/config.gin all and sleeping while it starts...
sleep 30