'''
This code is designed to produce mock kinematics based on a number of bulge and disc parameters. 


There are a number of things that will go into the production of mock kinematics. 
1) Photometric bulge-disc decompositions. 
	These will inform the code what portion of the galaxy light is made up of bulge or disc 
	at any given radius. 
2) Exponential light profile for the disc
3) Sersic light profile for the bulge. 
4) Bulge velocity dispersion (can make a free parameter)
5) Expression for the rotational velocity profile for a disc
6) Maximum rotational velocity (can make a free parameter)

At the end of the mock kinematics, a mock kriging map will be produced, which can be compared with the kriging map 
produced from observed data. 
'''

# retrieve dictionaries
import os, sys, random, time, pickle
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.path import Path
from chainconsumer import ChainConsumer

DropboxDirectory = os.getcwd().split('Dropbox')[0]
lib_path = os.path.abspath(DropboxDirectory+'Dropbox/PhD_Analysis/Library') 
sys.path.append(lib_path)
from galaxyParametersDictionary_v9 import *
from Sabine_Define import *
from KinematicProducer_def import *


GalName = 'NGC1023'
# instead of sampling a given range, I now sample the same pixels as given for an observed galaxy. 
ObservedGalaxyInput_Path = os.path.abspath(DropboxDirectory+'Dropbox/PhD_Analysis/Analysis/Angular Momentum/Mock_Kinematics')+'/'
X, Y, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed = krigingFileReadAll(ObservedGalaxyInput_Path, GalName)
	
t0 = time.time()

ndim, nwalkers = 14, 200 # NUMBER OF WALKERS


# setting the upper and lower bounds on the prior ranges of each parameter
ellipticity_bulge_lower, ellipticity_bulge_upper = 0.0, 0.5 # we want this to be fairly round...
I_Bulge_lower, I_Bulge_upper = 0, 10000
n_lower, n_upper = 3., 4.
Re_Bulge_lower, Re_Bulge_upper = 10, 100
BulgeRotationScale_lower, BulgeRotationScale_upper = 1, 50
Max_vel_bulge_lower, Max_vel_bulge_upper = -300, 300
CentralBulgeDispersion_lower, CentralBulgeDispersion_upper = 0, 100 # dispersion at R_e/2
alpha_Bulge_lower, alpha_Bulge_upper = 0.1, 2 # power law slope
# beta_Bulge_lower, beta_Bulge_upper = -5, 0 # slope of velocity dispersion profile
# gamma_Bulge_lower, gamma_Bulge_upper = -2, 0

# ellipticity_disc_lower, ellipticity_disc_upper = 0.5, 1.0
I_Disc_lower, I_Disc_upper = 0, 40
h_lower, h_upper = 0, 100
DiscRotationScale_lower, DiscRotationScale_upper = 1, 50
Max_vel_disc_lower, Max_vel_disc_upper = -400, 400
CentralDiscDispersion_lower, CentralDiscDispersion_upper = 0, 100 # dispersion at R_e/2
alpha_Disc_lower, alpha_Disc_upper = 0.1, 2 # power law slope
# beta_Disc_lower, beta_Disc_upper = -5, 0 # slope of velocity dispersion profile
# gamma_Disc_lower, gamma_Disc_upper = -2, 0


# defining the initial position of the walkers
pos_RotationAndDispersion = []
for ii in np.arange(nwalkers):  
	ellipticity_bulge_init = np.random.uniform(low=ellipticity_bulge_lower, high=ellipticity_bulge_upper) 
	I_Bulge_init = np.random.uniform(low=I_Bulge_lower, high=I_Bulge_upper)
	n_init = np.random.uniform(low=n_lower, high=n_upper) 
	Re_Bulge_init = np.random.uniform(low=Re_Bulge_lower, high=Re_Bulge_upper) 
	BulgeRotationScale_init = np.random.uniform(low=BulgeRotationScale_lower, high=BulgeRotationScale_upper) 
	Max_vel_bulge_init = np.random.uniform(low=Max_vel_bulge_lower, high=Max_vel_bulge_upper) 
	CentralBulgeDispersion_init = np.random.uniform(low=CentralBulgeDispersion_lower, high=CentralBulgeDispersion_upper) 
	alpha_Bulge_init = np.random.uniform(low=alpha_Bulge_lower, high=alpha_Bulge_upper) 
	# beta_Bulge_init = np.random.uniform(low=beta_Bulge_lower, high=beta_Bulge_upper) 
	# gamma_Bulge_init = np.random.uniform(low=gamma_Bulge_lower, high=gamma_Bulge_upper) 

	# ellipticity_disc_init = np.random.uniform(low=ellipticity_disc_lower, high=ellipticity_disc_upper) 
	I_Disc_init = np.random.uniform(low=I_Disc_lower, high=I_Disc_upper)
	h_init = np.random.uniform(low=h_lower, high=h_upper)
	DiscRotationScale_init = np.random.uniform(low=DiscRotationScale_lower, high=DiscRotationScale_upper) 
	Max_vel_disc_init = np.random.uniform(low=Max_vel_disc_lower, high=Max_vel_disc_upper) 
	CentralDiscDispersion_init = np.random.uniform(low=CentralDiscDispersion_lower, high=CentralDiscDispersion_upper) 
	alpha_Disc_init = np.random.uniform(low=alpha_Disc_lower, high=alpha_Disc_upper) 
	# beta_Disc_init = np.random.uniform(low=beta_Disc_lower, high=beta_Disc_upper) 
	# gamma_Disc_init = np.random.uniform(low=gamma_Disc_lower, high=gamma_Disc_upper) 

	pos_RotationAndDispersion.append([ellipticity_bulge_init, I_Bulge_init, n_init, Re_Bulge_init, BulgeRotationScale_init, \
		Max_vel_bulge_init, CentralBulgeDispersion_init, alpha_Bulge_init,  \
		# ellipticity_disc_init, \
		I_Disc_init, h_init, DiscRotationScale_init, Max_vel_disc_init, CentralDiscDispersion_init, \
		alpha_Disc_init])

# print pos_RotationAndDispersion
boundaries = [ellipticity_bulge_lower, ellipticity_bulge_upper, I_Bulge_lower, I_Bulge_upper, n_lower, n_upper, Re_Bulge_lower, Re_Bulge_upper, \
	BulgeRotationScale_lower, BulgeRotationScale_upper, Max_vel_bulge_lower, Max_vel_bulge_upper, \
	CentralBulgeDispersion_lower, CentralBulgeDispersion_upper, alpha_Bulge_lower, alpha_Bulge_upper, \
	# ellipticity_disc_lower, ellipticity_disc_upper, \
	I_Disc_lower, I_Disc_upper, h_lower, h_upper, DiscRotationScale_lower, DiscRotationScale_upper, \
	Max_vel_disc_lower, Max_vel_disc_upper, CentralDiscDispersion_lower, CentralDiscDispersion_upper, \
	alpha_Disc_lower, alpha_Disc_upper]

EffectiveRadius = Reff_Spitzer[GalName]
ObservedEllipticity = 1 - b_a[GalName]

# Setup MCMC sampler
import emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_RotationAndDispersion,
                  args=(boundaries, X, Y, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed, EffectiveRadius, ObservedEllipticity),
                  threads=16) #Threads gives the number of processors to use

############ implementing a burn-in period ###########
burnSteps = 50
pos, prob, state = sampler.run_mcmc(pos_RotationAndDispersion, burnSteps)
sampler.reset()

stepNumber = 200
outputMCMC = sampler.run_mcmc(pos, stepNumber) # uses the final position of the burn-in period as the starting point. 
######################################################


t = time.time() - t0
Date=time.strftime('%Y-%m-%d')
Time = time.asctime().split()[3]
print '########################################'
print 'time elapsed:', float(t)/3600, 'hours'
print '########################################'
print 'Mean acceptance fraction: ', (np.mean(sampler.acceptance_fraction))
OutputFilename = DropboxDirectory+'Dropbox/PhD_Analysis/Analysis/Angular Momentum/Mock_Kinematics/'+str(GalName)+'/'+str(GalName)+'_MCMCOutput.dat'
print 'output filename: ', OutputFilename

fileOut = open(OutputFilename, 'wb')
pickle.dump([sampler.chain, sampler.flatchain, sampler.lnprobability, sampler.flatlnprobability], fileOut)
fileOut.close()

# Making a notes file, if needed. 
# file=open('', 'w')
# file.write('\n')
 