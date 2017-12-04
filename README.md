traffic-sign-recognition

run live on bc4 :
	* srun -N 1 -p gpu --pty bash  
or  * srun -N 1 -p gpu -- gres=gpu:1 --mem=4G --pty bash

	
submitting a job to bc4 :
	* sbatch submit_job.sh
	* to look at the results open the slurm_### file 


**don't forget to download the images from the drive folder 