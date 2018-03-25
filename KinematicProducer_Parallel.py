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

lib_path = os.path.abspath('/nfs/cluster/gals/sbellstedt/Library') 
sys.path.append(lib_path)
from galaxyParametersDictionary_v9 import *
from Sabine_Define import *
from KinematicProducer_def import *


GalName = str(sys.argv[1])
KrigingInput = str(sys.argv[2]) # determines whether the SLUGGS dataset is given to MCMC in terms of the raw datapoints, or the kriging pixels. 
ExistingPhotometry = str(sys.argv[3])
TwoDatasets = str(sys.argv[4])
Magneticum = str(sys.argv[5])

pathName = '/nfs/cluster/gals/sbellstedt/Analysis/Mock_Kinematics/'
MagneticumPathName = '/nfs/cluster/gals/sbellstedt/Analysis/Mock_Kinematics/'
KinematicProducer_mainCall(pathName, MagneticumPathName, GalName, ExistingPhotometry, TwoDatasets, KrigingInput, Magneticum)
# # instead of sampling a given range, I now sample the same pixels as given for an observed galaxy. 
# ObservedGalaxyInput_Path = os.path.abspath('/nfs/cluster/gals/sbellstedt/Analysis/Mock_Kinematics')+'/'
# if KrigingInput:
# 	X, Y, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed = krigingFileReadAll(ObservedGalaxyInput_Path, GalName)
# elif Magneticum:
# 	pathMajor = '/nfs/cluster/gals/sbellstedt/Analysis/Mock_Kinematics/MagneticumGalaxies/'+GalName+'_vor.dat'
# 	X, Y, Vel_Observed, VelErr_Observed = np.loadtxt(pathMajor, skiprows = 1, unpack = True, usecols = [0, 1, 2, 3])
# 	VelDisp_Observed, VelDispErr_Observed = 2 * np.ones(len(X)), 2 * np.ones(len(X)) # There are no uncertainties associated with the maps, at least not currently. 
# 																					 # Hence currently I am using a low "dummy" value. 
# 																					 # Need to clarify with Felix at some point whether as estimate has been made
# 																					 # as to the uncertainties that arise in the projected kinematics from numerical
# 																					 # resolution origin. 
# else:
# 	InputDataFilename = ObservedGalaxyInput_Path+str(GalName)+'/MockKinematics_Input_'+str(GalName)+'_SLUGGS.txt'
# 	X, Y, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed = np.loadtxt(InputDataFilename, unpack = True, comments = '#')

# if TwoDatasets:
# 	InputDataFilename = ObservedGalaxyInput_Path+str(GalName)+'/MockKinematics_Input_'+str(GalName)+'_ATLAS3D.txt'
# 	X_2, Y_2, Vel_Observed_2, VelErr_Observed_2, VelDisp_Observed_2, VelDispErr_Observed_2 = np.loadtxt(InputDataFilename, unpack = True, comments = '#')
	
# t0 = time.time() # record the time to output how long the total run took. 


# if ExistingPhotometry:

# 	if TwoDatasets:
# 		ndim, nwalkers = 15, 1000 # NUMBER OF WALKERS
# 								  # if both datasets are being used, then two hyperparameters are required, which increases
# 								  # the number of free parameters. 
# 	else:
# 		ndim, nwalkers = 13, 1000 # NUMBER OF WALKERS
	
	
# 	# setting the upper and lower bounds on the prior ranges of each parameter
# 	ellipticity_bulge_lower, ellipticity_bulge_upper = 0.0, 0.6 # we want this to be fairly round...
# 	BulgeRotationScale_lower, BulgeRotationScale_upper = 1, 50
# 	Max_vel_bulge_lower, Max_vel_bulge_upper = -100, 100 # don't expect the bulge component to be rotating
# 	CentralBulgeDispersion_lower, CentralBulgeDispersion_upper = 0, 300 # dispersion at R_e/2
# 	alpha_Bulge_lower, alpha_Bulge_upper = 0, 0.2 # power law slope
	
	
# 	log_I_Disc_lower, log_I_Disc_upper = np.log10(1e-8), np.log10(1e2)
# 	Re_Disc_lower, Re_Disc_upper = 0, 100
# 	DiscRotationScale_lower, DiscRotationScale_upper = 1, 50
# 	Max_vel_disc_lower, Max_vel_disc_upper = -400, 400
# 	CentralDiscDispersion_lower, CentralDiscDispersion_upper = 0, 300 # dispersion at R_e/2
# 	alpha_Disc_lower, alpha_Disc_upper = 0, 0.5 # power law slope
	
# 	AzimuthVariationParameterBulge_lower, AzimuthVariationParameterBulge_upper = -1.0, 1.0
# 	AzimuthVariationParameterDisc_lower, AzimuthVariationParameterDisc_upper = -1.0, 1.0

# 	if TwoDatasets:
# 		# need to define the two hyperparameters for each dataset
# 		sluggsWeight_lower, sluggsWeight_upper = 0, 10
# 		atlasWeight_lower, atlasWeight_upper = 0, 10
# 		sluggsWeightPrior_lower, sluggsWeightPrior_upper = np.exp(-sluggsWeight_upper), np.exp(sluggsWeight_lower)
# 		atlasWeightPrior_lower, atlasWeightPrior_upper = np.exp(-atlasWeight_upper), np.exp(atlasWeight_lower)
	
	
# 	# defining the initial position of the walkers
# 	pos_RotationAndDispersion = []
# 	for ii in np.arange(nwalkers):  
# 		ellipticity_bulge_init = np.random.uniform(low=ellipticity_bulge_lower, high=ellipticity_bulge_upper) 
# 		BulgeRotationScale_init = np.random.uniform(low=BulgeRotationScale_lower, high=BulgeRotationScale_upper) 
# 		Max_vel_bulge_init = np.random.uniform(low=Max_vel_bulge_lower, high=Max_vel_bulge_upper) 
# 		CentralBulgeDispersion_init = np.random.uniform(low=CentralBulgeDispersion_lower, high=CentralBulgeDispersion_upper) 
# 		alpha_Bulge_init = np.random.uniform(low=alpha_Bulge_lower, high=alpha_Bulge_upper) 
	
# 		log_I_Disc_init = np.random.uniform(low=log_I_Disc_lower, high=log_I_Disc_upper) # sample in log space
# 		Re_Disc_init = np.random.uniform(low=Re_Disc_lower, high=Re_Disc_upper) 
# 		DiscRotationScale_init = np.random.uniform(low=DiscRotationScale_lower, high=DiscRotationScale_upper) 
# 		Max_vel_disc_init = np.random.uniform(low=Max_vel_disc_lower, high=Max_vel_disc_upper) 
# 		CentralDiscDispersion_init = np.random.uniform(low=CentralDiscDispersion_lower, high=CentralDiscDispersion_upper) 
# 		alpha_Disc_init = np.random.uniform(low=alpha_Disc_lower, high=alpha_Disc_upper) 
	
# 		AzimuthVariationParameterBulge_init = np.random.uniform(low=AzimuthVariationParameterBulge_lower, high=AzimuthVariationParameterBulge_upper) 
# 		AzimuthVariationParameterDisc_init = np.random.uniform(low=AzimuthVariationParameterDisc_lower, high=AzimuthVariationParameterDisc_upper) 

# 		if TwoDatasets:
# 			sluggsWeightPrior = (np.random.uniform(low=sluggsWeightPrior_lower, high=sluggsWeightPrior_upper))   # uniform sampling in expoential space
# 			atlasWeightPrior = (np.random.uniform(low=atlasWeightPrior_lower, high=atlasWeightPrior_upper)) # uniform sampling in exponential space  
# 			sluggsWeight = -np.log(sluggsWeightPrior)
# 			atlasWeight = -np.log(atlasWeightPrior)

# 			pos_RotationAndDispersion.append([ellipticity_bulge_init, \
# 				BulgeRotationScale_init, \
# 				Max_vel_bulge_init, CentralBulgeDispersion_init, alpha_Bulge_init,  \
# 				log_I_Disc_init, Re_Disc_init, DiscRotationScale_init, Max_vel_disc_init, CentralDiscDispersion_init, \
# 				alpha_Disc_init, AzimuthVariationParameterBulge_init, AzimuthVariationParameterDisc_init, \
# 				sluggsWeight, atlasWeight])
	
# 		else:
# 			pos_RotationAndDispersion.append([ellipticity_bulge_init, \
# 				BulgeRotationScale_init, \
# 				Max_vel_bulge_init, CentralBulgeDispersion_init, alpha_Bulge_init,  \
# 				log_I_Disc_init, Re_Disc_init, DiscRotationScale_init, Max_vel_disc_init, CentralDiscDispersion_init, \
# 				alpha_Disc_init, AzimuthVariationParameterBulge_init, AzimuthVariationParameterDisc_init])
	
# 	# print pos_RotationAndDispersion
# 	if TwoDatasets:
# 		boundaries = [ellipticity_bulge_lower, ellipticity_bulge_upper, \
# 			BulgeRotationScale_lower, BulgeRotationScale_upper, Max_vel_bulge_lower, Max_vel_bulge_upper, \
# 			CentralBulgeDispersion_lower, CentralBulgeDispersion_upper, alpha_Bulge_lower, alpha_Bulge_upper, \
# 			log_I_Disc_lower, log_I_Disc_upper, Re_Disc_lower, Re_Disc_upper, DiscRotationScale_lower, DiscRotationScale_upper, \
# 			Max_vel_disc_lower, Max_vel_disc_upper, CentralDiscDispersion_lower, CentralDiscDispersion_upper, \
# 			alpha_Disc_lower, alpha_Disc_upper, \
# 			AzimuthVariationParameterBulge_lower, AzimuthVariationParameterBulge_upper, AzimuthVariationParameterDisc_lower, AzimuthVariationParameterDisc_upper, \
# 			sluggsWeight_lower, sluggsWeight_upper, atlasWeight_lower, atlasWeight_upper]
# 	else:
# 		boundaries = [ellipticity_bulge_lower, ellipticity_bulge_upper, \
# 			BulgeRotationScale_lower, BulgeRotationScale_upper, Max_vel_bulge_lower, Max_vel_bulge_upper, \
# 			CentralBulgeDispersion_lower, CentralBulgeDispersion_upper, alpha_Bulge_lower, alpha_Bulge_upper, \
# 			log_I_Disc_lower, log_I_Disc_upper, Re_Disc_lower, Re_Disc_upper, DiscRotationScale_lower, DiscRotationScale_upper, \
# 			Max_vel_disc_lower, Max_vel_disc_upper, CentralDiscDispersion_lower, CentralDiscDispersion_upper, \
# 			alpha_Disc_lower, alpha_Disc_upper, \
# 			AzimuthVariationParameterBulge_lower, AzimuthVariationParameterBulge_upper, AzimuthVariationParameterDisc_lower, AzimuthVariationParameterDisc_upper]
	
# 	EffectiveRadius = Reff_Spitzer[GalName]
# 	ObservedEllipticity = 1 - b_a[GalName]
# 	n_bulge = SersicIndex_Bulge[GalName]
# 	Re_Bulge = EffectiveRadius_Bulge[GalName]
# 	mag_Bulge = MagnitudeRe_Bulge[GalName]
# 	Spitzer_Radius, Spitzer_Mag, Spitzer_MagErr = SpitzerMagProfileFinder(GalName) # I haven't yet done a thorough check on the impact of using a Spitzer lumiosity
# 																				   # profile, but using an optical magnitude for a previous photometric decomposition. 
# 																				   # My assumption here is that the impact is negligible. 
	
# 	# Setup MCMC sampler
# 	import emcee
# 	if TwoDatasets:
# 		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_RotationAndDispersion_hyperparameters,
# 		                  args=(boundaries, X, Y, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed, \
# 		                  	X_2, Y_2, Vel_Observed_2, VelErr_Observed_2, VelDisp_Observed_2, VelDispErr_Observed_2, \
# 		                  	EffectiveRadius, ObservedEllipticity, n_bulge, Re_Bulge, mag_Bulge, Spitzer_Radius, Spitzer_Mag, Spitzer_MagErr),
# 		                  threads=16) #Threads gives the number of processors to use
# 	else:
# 		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_RotationAndDispersion,
# 		                  args=(boundaries, X, Y, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed, \
# 		                  	EffectiveRadius, ObservedEllipticity, n_bulge, Re_Bulge, mag_Bulge, Spitzer_Radius, Spitzer_Mag, Spitzer_MagErr),
# 		                  threads=16) #Threads gives the number of processors to use
	
# 	############ implementing a burn-in period ###########
# 	burnSteps = 1000
# 	pos, prob, state = sampler.run_mcmc(pos_RotationAndDispersion, burnSteps)
# 	sampler.reset()
	
# 	stepNumber = 4000
# 	outputMCMC = sampler.run_mcmc(pos, stepNumber) # uses the final position of the burn-in period as the starting point. 
# 	######################################################
	
	
# 	t = time.time() - t0
# 	Date=time.strftime('%Y-%m-%d')
# 	Time = time.asctime().split()[3]
# 	print '########################################'
# 	if float(t)/3600 < 1:
# 		print 'time elapsed:', float(t)/60, 'minutes'
# 	else:
# 		print 'time elapsed:', float(t)/3600, 'hours'
# 	print '########################################'
# 	print 'Mean acceptance fraction: ', (np.mean(sampler.acceptance_fraction))
# 	if TwoDatasets:
# 		OutputFilename = '/nfs/cluster/gals/sbellstedt/Analysis/Mock_Kinematics/'+str(GalName)+'/'+str(GalName)+'_MCMCOutput_TwoDatasets.dat'
# 	else:
# 		OutputFilename = '/nfs/cluster/gals/sbellstedt/Analysis/Mock_Kinematics/'+str(GalName)+'/'+str(GalName)+'_MCMCOutput.dat'
# 	print 'output filename: ', OutputFilename
	
# 	fileOut = open(OutputFilename, 'wb')
# 	pickle.dump([sampler.chain, sampler.flatchain, sampler.lnprobability, sampler.flatlnprobability], fileOut)
# 	fileOut.close()

# # here implement an entirely independent version of the code for the purpose of fitting to simulated data. 
# else:
# 	if TwoDatasets:
# 		ndim, nwalkers = 19, 1000 # NUMBER OF WALKERS
# 	else:
# 		ndim, nwalkers = 17, 2000 # NUMBER OF WALKERS
	
	
# 	# setting the upper and lower bounds on the prior ranges of each parameter
# 	log_I_bulge_lower, log_I_bulge_upper = np.log10(1e-3), np.log10(1e7)
# 	Re_bulge_lower, Re_bulge_upper = 0, 100
# 	ellipticity_bulge_lower, ellipticity_bulge_upper = 0.0, 0.6 # we want this to be fairly round...
# 	n_lower, n_upper = 1, 8
# 	BulgeRotationScale_lower, BulgeRotationScale_upper = 1, 50
# 	Max_vel_bulge_lower, Max_vel_bulge_upper = -100, 100 # don't expect the bulge component to be rotating
# 	CentralBulgeDispersion_lower, CentralBulgeDispersion_upper = 0, 300 # dispersion at R_e/2
# 	alpha_Bulge_lower, alpha_Bulge_upper = 0, 0.2 # power law slope
	
	
# 	log_I_Disc_lower, log_I_Disc_upper = np.log10(1e-8), np.log10(1e2)
# 	Re_Disc_lower, Re_Disc_upper = 0, 100
# 	ellipticity_disc_lower, ellipticity_disc_upper = 0.0, 1.0 
# 	DiscRotationScale_lower, DiscRotationScale_upper = 1, 50
# 	Max_vel_disc_lower, Max_vel_disc_upper = -400, 400
# 	CentralDiscDispersion_lower, CentralDiscDispersion_upper = 0, 300 # dispersion at R_e/2
# 	alpha_Disc_lower, alpha_Disc_upper = 0, 0.5 # power law slope
	
# 	AzimuthVariationParameterBulge_lower, AzimuthVariationParameterBulge_upper = -1.0, 1.0
# 	AzimuthVariationParameterDisc_lower, AzimuthVariationParameterDisc_upper = -1.0, 1.0

# 	if TwoDatasets:
# 		# need to define the two hyperparameters for each dataset
# 		sluggsWeight_lower, sluggsWeight_upper = 0, 10
# 		atlasWeight_lower, atlasWeight_upper = 0, 10
# 		sluggsWeightPrior_lower, sluggsWeightPrior_upper = np.exp(-sluggsWeight_upper), np.exp(sluggsWeight_lower)
# 		atlasWeightPrior_lower, atlasWeightPrior_upper = np.exp(-atlasWeight_upper), np.exp(atlasWeight_lower)
	
	
# 	# defining the initial position of the walkers
# 	pos_RotationAndDispersion = []
# 	for ii in np.arange(nwalkers): 
# 		log_I_bulge_init = np.random.uniform(low=log_I_bulge_lower, high=log_I_bulge_upper)
# 		Re_bulge_init =  np.random.uniform(low=Re_bulge_lower, high=Re_bulge_upper)
# 		ellipticity_bulge_init = np.random.uniform(low=ellipticity_bulge_lower, high=ellipticity_bulge_upper) 
# 		n_init = np.random.uniform(low=n_lower, high=n_upper) 
# 		BulgeRotationScale_init = np.random.uniform(low=BulgeRotationScale_lower, high=BulgeRotationScale_upper) 
# 		Max_vel_bulge_init = np.random.uniform(low=Max_vel_bulge_lower, high=Max_vel_bulge_upper) 
# 		CentralBulgeDispersion_init = np.random.uniform(low=CentralBulgeDispersion_lower, high=CentralBulgeDispersion_upper) 
# 		alpha_Bulge_init = np.random.uniform(low=alpha_Bulge_lower, high=alpha_Bulge_upper) 
	
# 		log_I_Disc_init = np.random.uniform(low=log_I_Disc_lower, high=log_I_Disc_upper) # sample in log space
# 		Re_Disc_init = np.random.uniform(low=Re_Disc_lower, high=Re_Disc_upper) 
# 		ellipticity_disc_init = np.random.uniform(low=ellipticity_disc_lower, high=ellipticity_disc_upper) 
# 		DiscRotationScale_init = np.random.uniform(low=DiscRotationScale_lower, high=DiscRotationScale_upper) 
# 		Max_vel_disc_init = np.random.uniform(low=Max_vel_disc_lower, high=Max_vel_disc_upper) 
# 		CentralDiscDispersion_init = np.random.uniform(low=CentralDiscDispersion_lower, high=CentralDiscDispersion_upper) 
# 		alpha_Disc_init = np.random.uniform(low=alpha_Disc_lower, high=alpha_Disc_upper) 
	
# 		AzimuthVariationParameterBulge_init = np.random.uniform(low=AzimuthVariationParameterBulge_lower, high=AzimuthVariationParameterBulge_upper) 
# 		AzimuthVariationParameterDisc_init = np.random.uniform(low=AzimuthVariationParameterDisc_lower, high=AzimuthVariationParameterDisc_upper) 
	
# 		if TwoDatasets:
# 			sluggsWeightPrior = (np.random.uniform(low=sluggsWeightPrior_lower, high=sluggsWeightPrior_upper))   # uniform sampling in expoential space
# 			atlasWeightPrior = (np.random.uniform(low=atlasWeightPrior_lower, high=atlasWeightPrior_upper)) # uniform sampling in exponential space  
# 			sluggsWeight = -np.log(sluggsWeightPrior)
# 			atlasWeight = -np.log(atlasWeightPrior)

# 			pos_RotationAndDispersion.append([log_I_bulge_init, Re_bulge_init, ellipticity_bulge_init, n_init, \
# 				BulgeRotationScale_init, \
# 				Max_vel_bulge_init, CentralBulgeDispersion_init, alpha_Bulge_init,  \
# 				log_I_Disc_init, Re_Disc_init, ellipticity_disc_init, DiscRotationScale_init, Max_vel_disc_init, CentralDiscDispersion_init, \
# 				alpha_Disc_init, AzimuthVariationParameterBulge_init, AzimuthVariationParameterDisc_init, \
# 				sluggsWeight, atlasWeight])
	
# 		else:
# 			pos_RotationAndDispersion.append([log_I_bulge_init, Re_bulge_init, ellipticity_bulge_init, n_init, \
# 				BulgeRotationScale_init, \
# 				Max_vel_bulge_init, CentralBulgeDispersion_init, alpha_Bulge_init,  \
# 				log_I_Disc_init, Re_Disc_init, ellipticity_disc_init, DiscRotationScale_init, Max_vel_disc_init, CentralDiscDispersion_init, \
# 				alpha_Disc_init, AzimuthVariationParameterBulge_init, AzimuthVariationParameterDisc_init])
	
# 	# print pos_RotationAndDispersion
# 	if TwoDatasets:
# 		boundaries = [log_I_bulge_lower, log_I_bulge_upper, Re_bulge_lower, Re_bulge_upper, ellipticity_bulge_lower, ellipticity_bulge_upper, \
# 			n_lower, n_upper, BulgeRotationScale_lower, BulgeRotationScale_upper, Max_vel_bulge_lower, Max_vel_bulge_upper, \
# 			CentralBulgeDispersion_lower, CentralBulgeDispersion_upper, alpha_Bulge_lower, alpha_Bulge_upper, \
# 			log_I_Disc_lower, log_I_Disc_upper, Re_Disc_lower, Re_Disc_upper, ellipticity_disc_lower, ellipticity_disc_upper, \
# 			DiscRotationScale_lower, DiscRotationScale_upper, \
# 			Max_vel_disc_lower, Max_vel_disc_upper, CentralDiscDispersion_lower, CentralDiscDispersion_upper, \
# 			alpha_Disc_lower, alpha_Disc_upper, \
# 			AzimuthVariationParameterBulge_lower, AzimuthVariationParameterBulge_upper, AzimuthVariationParameterDisc_lower, AzimuthVariationParameterDisc_upper, \
# 			sluggsWeight_lower, sluggsWeight_upper, atlasWeight_lower, atlasWeight_upper]
# 	else:
# 		boundaries = [log_I_bulge_lower, log_I_bulge_upper, Re_bulge_lower, Re_bulge_upper, ellipticity_bulge_lower, ellipticity_bulge_upper, \
# 			n_lower, n_upper, BulgeRotationScale_lower, BulgeRotationScale_upper, Max_vel_bulge_lower, Max_vel_bulge_upper, \
# 			CentralBulgeDispersion_lower, CentralBulgeDispersion_upper, alpha_Bulge_lower, alpha_Bulge_upper, \
# 			log_I_Disc_lower, log_I_Disc_upper, Re_Disc_lower, Re_Disc_upper, ellipticity_disc_lower, ellipticity_disc_upper, \
# 			DiscRotationScale_lower, DiscRotationScale_upper, \
# 			Max_vel_disc_lower, Max_vel_disc_upper, CentralDiscDispersion_lower, CentralDiscDispersion_upper, \
# 			alpha_Disc_lower, alpha_Disc_upper, \
# 			AzimuthVariationParameterBulge_lower, AzimuthVariationParameterBulge_upper, AzimuthVariationParameterDisc_lower, AzimuthVariationParameterDisc_upper]
	
# 	# Setup MCMC sampler
# 	import emcee
# 	if TwoDatasets:
# 		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_RotationAndDispersion_sims_TwoDatasets,
# 		                  args=(boundaries, X, Y, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed, \
# 		                  	X_2, Y_2, Vel_Observed_2, VelErr_Observed_2, VelDisp_Observed_2, VelDispErr_Observed_2),
# 		                  threads=16) #Threads gives the number of processors to use
# 	else:
# 		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_RotationAndDispersion_sims,
# 		                  args=(boundaries, X, Y, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed),
# 		                  threads=16) #Threads gives the number of processors to use
	
# 	############ implementing a burn-in period ###########
# 	burnSteps = 4000
# 	pos, prob, state = sampler.run_mcmc(pos_RotationAndDispersion, burnSteps)
# 	sampler.reset()
	
# 	stepNumber = 2000
# 	outputMCMC = sampler.run_mcmc(pos, stepNumber) # uses the final position of the burn-in period as the starting point. 
# 	######################################################
	
	
# 	t = time.time() - t0
# 	Date=time.strftime('%Y-%m-%d')
# 	Time = time.asctime().split()[3]
# 	print '########################################'
# 	if float(t)/3600 < 1:
# 		print 'time elapsed:', float(t)/60, 'minutes'
# 	else:
# 		print 'time elapsed:', float(t)/3600, 'hours'
# 	print '########################################'
# 	print 'Mean acceptance fraction: ', (np.mean(sampler.acceptance_fraction))
# 	if TwoDatasets:
# 		OutputFilename = '/nfs/cluster/gals/sbellstedt/Analysis/Mock_Kinematics/'+str(GalName)+'/'+str(GalName)+'_MCMCOutput_simsversion_TwoDatasets.dat'
# 	else:
# 		OutputFilename = '/nfs/cluster/gals/sbellstedt/Analysis/Mock_Kinematics/'+str(GalName)+'/'+str(GalName)+'_MCMCOutput_simsversion.dat'
# 	print 'output filename: ', OutputFilename
	
# 	fileOut = open(OutputFilename, 'wb')
# 	pickle.dump([sampler.chain, sampler.flatchain, sampler.lnprobability, sampler.flatlnprobability], fileOut)
# 	fileOut.close()	