#!/bin/bash
time=$(date "+%Y-%m-%d-%H-%M-%S")

mkdir outfile/$time


#nohup python -u main.py --seg_data 5 --algorithm "pFedMe" --_balance True --lamda 0.0001 --lr 0.01 --_running_time $time > ./outfile/$time/running1.out 2>&1 &
#nohup python -u main.py --seg_data 5 --algorithm "pFedMe" --_balance False --lamda 0.0001 --lr 0.01 --_running_time $time > ./outfile/$time/running2.out 2>&1 &
#nohup python -u main.py --seg_data 10 --algorithm "pFedMe" --_balance True --lamda 0.0001 --lr 0.01 --_running_time $time > ./outfile/$time/running3.out 2>&1 &
#nohup python -u main.py --seg_data 10 --algorithm "pFedMe" --_balance False --lamda 0.0001 --lr 0.01 --_running_time $time > ./outfile/$time/running4.out 2>&1 &
#nohup python -u main.py --seg_data 20 --algorithm "pFedMe" --_balance True --lamda 0.0001 --lr 0.01 --_running_time $time > ./outfile/$time/running5.out 2>&1 &
#nohup python -u main.py --seg_data 20 --algorithm "pFedMe" --_balance False --lamda 0.0001 --lr 0.01 --_running_time $time > ./outfile/$time/running6.out 2>&1 &
#nohup python -u main.py --seg_data 50 --algorithm "pFedMe" --_balance True --lamda 0.0001 --lr 0.01 --_running_time $time > ./outfile/$time/running7.out 2>&1 &
#nohup python -u main.py --seg_data 50 --algorithm "pFedMe" --_balance False --lamda 0.0001 --lr 0.01 --_running_time $time > ./outfile/$time/running8.out 2>&1 &

#nohup python -u main.py --seg_data 50 --algorithm "pFedMe" --_balance False --lamda 0.0001 --lr 0.05 --_running_time $time > ./outfile/$time/running1.out 2>&1 &
#nohup python -u main.py --seg_data 50 --algorithm "pFedMe" --_balance False --lamda 0.0001 --lr 0.1 --_running_time $time > ./outfile/$time/running2.out 2>&1 &
#nohup python -u main.py --seg_data 50 --algorithm "pFedMe" --_balance False --lamda 0.0001 --lr 0.75 --_running_time $time > ./outfile/$time/running3.out 2>&1 &

#nohup python -u main.py --seg_data 50 --algorithm "pFedMe" --_balance True --lamda 0.0001 --lr 0.05 --_running_time $time > ./outfile/$time/running1.out 2>&1 &
#nohup python -u main.py --seg_data 50 --algorithm "pFedMe" --_balance True --lamda 0.0001 --lr 0.1 --_running_time $time > ./outfile/$time/running2.out 2>&1 &
#nohup python -u main.py --seg_data 50 --algorithm "pFedMe" --_balance True --lamda 0.0001 --lr 0.75 --_running_time $time > ./outfile/$time/running3.out 2>&1 &
#wait

#nohup python -u main.py --seg_data 5 --algorithm "pFedMe" --_balance False --lamda 0.0001 --lr 0.75 --_running_time $time > ./outfile/$time/running1.out 2>&1 &
#nohup python -u main.py --seg_data 10 --algorithm "pFedMe" --_balance False --lamda 0.0001 --lr 0.75 --_running_time $time > ./outfile/$time/running2.out 2>&1 &
#nohup python -u main.py --seg_data 20 --algorithm "pFedMe" --_balance False --lamda 0.0001 --lr 0.75 --_running_time $time > ./outfile/$time/running3.out 2>&1 &
#wait

nohup python -u main.py --seg_data 5 --algorithm "pFedMe" --_balance True --lamda 0.0001 --lr 0.01 --_running_time $time > ./outfile/$time/running1.out 2>&1 &
nohup python -u main.py --seg_data 10 --algorithm "pFedMe" --_balance True --lamda 0.0001 --lr 0.01 --_running_time $time > ./outfile/$time/running2.out 2>&1 &
nohup python -u main.py --seg_data 20 --algorithm "pFedMe" --_balance True --lamda 0.0001 --lr 0.01 --_running_time $time > ./outfile/$time/running3.out 2>&1 &
wait

done








