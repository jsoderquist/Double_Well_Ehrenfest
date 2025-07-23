#!/software/anaconda3/2020.11/bin/python
#SBATCH -p polariton
#SBATCH -o output_multipar.log
#SBATCH --mem-per-cpu=4GB
#SBATCH -t 15:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

import os, sys
import subprocess
import time
import numpy as np
from pathlib import Path
# import parameters as par

try:
    os.rmdir("/scratch/jsoderqu/Simpkins/MFE/parallel/tmpdir")
except:
    print("No folder") 

NARRAY = str(99) # number of jobs
filename = "job"

manual = 0
JOBIDnum = 22951491
ARRAYJOBIDnum = 22951498

if(manual==1):
    JOBID = str(JOBIDnum)
    ARRAYJOBID = str(ARRAYJOBIDnum)
else:
    JOBID = str(os.environ["SLURM_JOB_ID"]) # get ID of this job

    Path("tmpdir").mkdir(parents=True, exist_ok=True) # make temporary directory for individual job files
    os.chdir("tmpdir") # change to temporary directory
    os.system('cp ../*.py .')
    command = str("sbatch --array [0-" + NARRAY + "] ../run.sh") # command to submit job array

    open(filename,'a').close()

    t0 = time.time()

    ARRAYJOBID = str(subprocess.check_output(command, shell=True)).replace("b'Submitted batch job ","").replace("\\n'","") # runs job array and saves job ID of the array

    t1 = time.time()
    print("Job ID: " + JOBID)
    print("Array time: ",t1-t0)

    os.chdir("..") # go back to original directory
