#PBS -q UCTlong
#PBS -l nodes=1:ppn=2:series600
#PBS -N kag_web_sub

#PBS -m abe
#PBS -M blake.rsa@gmail.com

/opt/exp_soft/anaconda/python3.4/bin/python -o /home/cnnbla001/rl_learner/pg_karpathy_30000.py
