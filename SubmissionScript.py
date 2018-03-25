import subprocess as sp 
import numpy as np


Galaxies = ['NGC1023', 'NGC1400', 'NGC2549', 'NGC2699', 'NGC2768', 'NGC2974', 'NGC3115', "NGC3607", 'NGC4111', 'NGC4459', \
'NGC4474', 'NGC4526', 'NGC4649', "NGC5866", 'NGC7457', "NGC720", "NGC821", 'NGC1407', "NGC3377", "NGC3608", "NGC4278", \
"NGC4365", "NGC4374", "NGC4473", "NGC4486", "NGC4494", "NGC4564", "NGC4697", 'NGC5846']
KrigingInput = True # determines whether the SLUGGS dataset is given to MCMC in terms of the raw datapoints, or the kriging pixels. 
ExistingPhotometry = False
TwoDatasets = True
Magneticum = False

for GalName in Galaxies:

	filename='B_D_Recomp_'+str(GalName)+'.sh'
	file=open(filename, 'w')
	file.write('#!/bin/bash\n')
	file.write('#PBS -q gstar\n')
	file.write('#PBS -l walltime=00:04:00:00\n')
	file.write('#PBS -l nodes=1:ppn=8\n')
	file.write('#PBS -l mem=1gb\n')
	file.write('\n')
	file.write('module load python\n')
	file.write('cd /nfs/cluster/gals/sbellstedt/Analysis/Mock_Kinematics/\n')
	file.write('python /nfs/cluster/gals/sbellstedt/Analysis/Mock_Kinematics/KinematicProducer_Parallel.py '+\
		str(GalName)+' '+str(KrigingInput)+' '+str(ExistingPhotometry)+' '+str(TwoDatasets)+' '+str(Magneticum)+'\n')
	file.write('\n')
	file.close()
	
	sp.call('qsub '+filename, shell=True)
	sp.call('rm '+filename, shell=True)