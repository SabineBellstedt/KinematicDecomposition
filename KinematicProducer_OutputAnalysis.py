'''
Producing the final output files to analyse the MCMC output for the mock kinematics generation. 
'''

# retrieve dictionaries
import os, sys, random, time, pickle, glob
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.path import Path
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib as mpl
from chainconsumer import ChainConsumer

DropboxDirectory = os.getcwd().split('Dropbox')[0]
lib_path = os.path.abspath(DropboxDirectory+'Dropbox/PhD_Analysis/Library') 
sys.path.append(lib_path)
from galaxyParametersDictionary_v9 import *
from Sabine_Define import *
from KinematicProducer_def import *
from KinematicProducer_PlottingFunctions import *

def parameterExtractor(inputDict, name):
	value = inputDict[name][1]
	lower = inputDict[name][1] - inputDict[name][0]
	upper = inputDict[name][2] - inputDict[name][1]
	print 'Parameter extraction successful for', name
	return value, lower, upper

# Galaxies = ['NGC2768']
# Galaxies = ['NGC1023', 'NGC2549', 'NGC2699', 'NGC2768', 'NGC2974', "NGC3607", 'NGC4459', 'NGC4111', \
# 'NGC4474', 'NGC4526', 'NGC4649', "NGC5866", 'NGC7457', "NGC720", "NGC821", "NGC3377", "NGC3608", "NGC4278", \
#  "NGC4473", "NGC4486", "NGC4494", "NGC4564", "NGC4697", 'NGC5846']
Galaxies = ['gal_26', 'gal_28', 'gal_49', 'gal_59', 'gal_75', 'gal_247', 'gal_302', 'gal_500']

Photometry = False # if this and simulations is set to false, then the code analyses the MCMC outputs of observational data with no photometric constraints. 
TwoDatasets = False
Magneticum = True

for GalName in Galaxies:
	# figuring out which version of the output file I want to read, given my input constraints. 	
	suffix = ''
	if TwoDatasets:
		suffix = suffix + 'TwoDatasets_'
	if Photometry:
		suffix = suffix + 'Photometry_'
	if Magneticum:
		suffix = suffix + 'Magneticum_'
	if GlobularClusters:
		suffix = suffix + 'GC_'

	print GalName

	# identify the possible file locations given the input variations. 
	MatchingDirectories = glob.glob(DropboxDirectory+'Dropbox/PhD_Analysis/Analysis/Angular Momentum/Mock_Kinematics/'+str(GalName)+'/*'+suffix+'FreeParam-*/')
	if len(MatchingDirectories) > 1:
		print 'multiple versions of this MCMC output have been detected'
	for entry in MatchingDirectories:
		if len(glob.glob(entry+'*_parameters.txt')) > 0:
			Recalculate = True
			print glob.glob(entry+'*_parameters.txt')[0]
			ParameterFile = glob.glob(entry+'*_parameters.txt')[0]

			# now to check whether the code has already been run on the most recent parameter file:
			if len(glob.glob(entry+'*_IntensityProfile.pdf')) > 0:
				ParameterFileModificationTime = os.path.getmtime(ParameterFile) 
				AnalysisFileModificationTime = os.path.getmtime(glob.glob(entry+'*_IntensityProfile.pdf')[0]) 

				if AnalysisFileModificationTime > ParameterFileModificationTime:
					Recalculate = False
					print 'Output previously analysed'
	
			if Recalculate:
				Parameters = np.loadtxt(ParameterFile, usecols = [0], dtype = 'str', comments = '#')
				Values, LowerErr, UpperErr = np.loadtxt(ParameterFile, usecols = [1, 2, 3], comments = '#', unpack = True)
		
				ellipticity_bulge = Values[np.where(Parameters == 'BulgeEllipticity')]
				if 'BulgeVelocity' in Parameters:
					Max_vel_bulge = Values[np.where(Parameters == 'BulgeVelocity')]
					BulgeRotationScale = Values[np.where(Parameters == 'BulgeRotationScale')]
				elif 'BulgeRotationScale' in Parameters:
					raise ValueError("bulge rotation scale given, but not bulge rotational velocity. Check which values are included. ")
				else:
					BulgeRotationScale = 0
					Max_vel_bulge = 0
				Re_Bulge = Values[np.where(Parameters == 'BulgeSize')]
				CentralBulgeDispersion = Values[np.where(Parameters == 'BulgeDispersion')]
				alpha_Bulge = Values[np.where(Parameters == 'BulgeDispersionDropOff')]
				log_I_Disc = Values[np.where(Parameters == 'DiscIntensity')]
				Re_Disc = Values[np.where(Parameters == 'DiscSize')]
				DiscRotationScale = Values[np.where(Parameters == 'DiscRotationScale')]
				Max_vel_disc = Values[np.where(Parameters == 'DiscVelocity')]
			
				if 'DiscDispersion' in Parameters:
					CentralDiscDispersion = Values[np.where(Parameters == 'DiscDispersion')]
					alpha_Disc = Values[np.where(Parameters == 'DiscDispersionDropOff')]
				elif 'DiscDispersionDropOff' in Parameters:
					raise ValueError("Disc dispersion drop off value given, but not central disc dispersion. Check which values are included. ")
				else:
					# in this scenario, the disc dispersion has not been determined. 
					# This means that we assume the full dispersion contribution comes from the bulge component of the galaxy. 
					CentralDiscDispersion = 0
					alpha_Disc = 0
			
				AzimuthVariationParameterBulge = Values[np.where(Parameters == 'BulgeAzimuthVariation')]
				AzimuthVariationParameterDisc = Values[np.where(Parameters == 'DiscAzimuthVariation')]
			
			
				if 'BulgeIntensity' in Parameters:
					log_I_Bulge = Values[np.where(Parameters == 'BulgeIntensity')] 
				else:
					try:
						log_I_Bulge = - MagnitudeRe_Bulge[GalName] / 2.5
					except:
						raise ValueError("Observed bulge magnitude not provided, and bulge intensity has not been set as a free parameter. ")
			
			
				if 'BulgeSize' in Parameters:
					Re_Bulge = Values[np.where(Parameters == 'BulgeSize')] 
				else:
					try:
						Re_Bulge = EffectiveRadius_Bulge[GalName]
					except:
						raise ValueError("Observed bulge Re not provided, and has not been set as a free parameter. ")
			
			
				if 'BulgeSersicIndex' in Parameters:
					n = Values[np.where(Parameters == 'BulgeSersicIndex')] 
				else:
					try:
						n = SersicIndex_Bulge[GalName]
					except:
						raise ValueError("Observed bulge sersic index not provided, and has not been set as a free parameter. ")
			
				# now to check whether the user provided a disc ellipticity
				if 'DiscEllipticity' in Parameters:
					ellipticity_disc = Values[np.where(Parameters == 'DiscEllipticity')]
				else:
					try:
						ObservedEllipticity = 1 - b_a[GalName]
						EffectiveRadius = Reff_Spitzer[GalName]
				
						BulgeIntensity_ellipticityTest = BulgeIntensityFunction(log_I_Bulge, EffectiveRadius, Re_Bulge, n)
						DiscIntensity_ellipticityTest = DiscIntensityFunction(log_I_Disc, EffectiveRadius, Re_Disc)
						
						BulgeFraction_ellipticityTest = BulgeIntensity_ellipticityTest / (BulgeIntensity_ellipticityTest + DiscIntensity_ellipticityTest)
						DiscFraction_ellipticityTest = DiscIntensity_ellipticityTest / (BulgeIntensity_ellipticityTest + DiscIntensity_ellipticityTest)
						
						ellipticity_disc = (ObservedEllipticity - BulgeFraction_ellipticityTest*ellipticity_bulge) / DiscFraction_ellipticityTest
			
					except:
						raise ValueError("Observed ellipticity not provided, and disc ellipticity has not been set as a free parameter. ")
			
				
				PA = 90*np.pi/180
				phi=(PA-np.pi/2.0) # accounting for the different 0 PA convention in astronomy to mathematics
			
				
				# X, Y, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed = krigingFileReadAll(ObservedGalaxyInput_Path, GalName)
				#creating the 2D grid over which to generate the galaxy
				if Magneticum:
					ObservedEllipticity = 0.5 # need to update this to the actual values
					EffectiveRadius = 20
				else:
					ObservedEllipticity = 1 - b_a[GalName]
					EffectiveRadius = Reff_Spitzer[GalName]
				gridSizex = int(4*EffectiveRadius/np.sqrt(1-ObservedEllipticity))
				gridSizey = int(4*EffectiveRadius*np.sqrt(1-ObservedEllipticity))
				X, Y = [], []
				for xx in np.arange(-gridSizex, gridSizex, 2):
					for yy in np.arange(-gridSizey, gridSizey, 2):
						X.append(xx)
						Y.append(yy)
				X = np.array(X)
				Y = np.array(Y)
				
				X, Y, BulgeIntensity, DiscIntensity, DiscRotation, BulgeRotation, TotalRotation, DiscDispersion, BulgeDispersion, TotalDispersion, \
					TotalIntensity, BulgeRotationField, DiscRotationField, MinimumRotation, MaximumRotation, MinimumDispersion, MaximumDispersion = \
					MockKinematicsModelAnalysis(X, Y, phi, ellipticity_bulge, ellipticity_disc, log_I_Bulge, Re_Bulge, n, log_I_Disc, Re_Disc, \
					Max_vel_disc, DiscRotationScale, AzimuthVariationParameterDisc, Max_vel_bulge, BulgeRotationScale, AzimuthVariationParameterBulge, \
					CentralBulgeDispersion, CentralDiscDispersion, alpha_Bulge, alpha_Disc, gridSizex, gridSizey)
				
				# setting up a numerical axis large enough to numerically calculate the total intensity
				Size_IntensityTest = 500
				X_extrapolatedIntensity = []
				Y_extrapolatedIntensity = []
				for xx in np.arange(-Size_IntensityTest, Size_IntensityTest, 1):
				    for yy in np.arange(-Size_IntensityTest, Size_IntensityTest, 1):
				        X_extrapolatedIntensity.append(xx)
				        Y_extrapolatedIntensity.append(yy)
				X_extrapolatedIntensity = np.array(X_extrapolatedIntensity)
				Y_extrapolatedIntensity = np.array(Y_extrapolatedIntensity)
				 
				Radius_Bulge_extrapolatedIntensity, Radius_Disc_extrapolatedIntensity = \
				    ComponentRadiusFunction(X_extrapolatedIntensity, Y_extrapolatedIntensity, phi, ellipticity_bulge, ellipticity_disc)
				BulgeIntensity_extrapolatedIntensity = BulgeIntensityFunction(log_I_Bulge, Radius_Bulge_extrapolatedIntensity, Re_Bulge, n)
				DiscIntensity_extrapolatedIntensity = DiscIntensityFunction(log_I_Disc, Radius_Disc_extrapolatedIntensity, Re_Disc)
				 
				# calculating the bulge/total ratio
				print '------------------------------------'
				print 'Bulge/Total:', np.sum(BulgeIntensity_extrapolatedIntensity) / ( np.sum(BulgeIntensity_extrapolatedIntensity) + np.sum(DiscIntensity_extrapolatedIntensity) )
				print 'Bulge/Disc:', np.sum(BulgeIntensity_extrapolatedIntensity) / np.sum(DiscIntensity_extrapolatedIntensity)
				print '------------------------------------'
				 
				TotalIntensity_extrapolatedIntensity = BulgeIntensity_extrapolatedIntensity + DiscIntensity_extrapolatedIntensity
				 
				# # first do a coarse iteration to identify the half light radius
				Radius_EllipProfile, Ellipticity_EllipProfile = [], []
				for testRadius in np.arange(10, 200, 4):
				    # first identify the pixel that is closest in radius to the test radius
				    testIntensity = TotalIntensity_extrapolatedIntensity[np.where((Y_extrapolatedIntensity == 0) & \
				        (X_extrapolatedIntensity > (testRadius - 1)) & (X_extrapolatedIntensity < (testRadius + 1)))][0]
				    # now identifying the radius of the associated intensity along the semiminor axis
				    # will have to iterate to identify the closest value, given the finite resolution
				    diff = 100000
				    for testMinorRadius in np.arange(5, testRadius, 2):
				        minorIntensity = TotalIntensity_extrapolatedIntensity[np.where((X_extrapolatedIntensity == 0) & (Y_extrapolatedIntensity == testMinorRadius))]
				        if abs(testIntensity - minorIntensity) < diff:
				            diff = abs(testIntensity - minorIntensity)
				            minorRadius = testMinorRadius
				    testEllipticity = 1.-(float(minorRadius)/testRadius)
				    Radius_EllipProfile.append(testRadius)
				    Ellipticity_EllipProfile.append(testEllipticity)		
				 
				Radius_EllipProfile, Ellipticity_EllipProfile = np.array(Radius_EllipProfile), np.array(Ellipticity_EllipProfile)	
						
				Ellipticity = ObservedEllipticity
				IntensityPlottingFunction(X, Y, BulgeIntensity, DiscIntensity, TotalIntensity, gridSizex, gridSizey, phi, Ellipticity, \
					EffectiveRadius, filename = entry+str(GalName)+'_ModelPhotometry.pdf')
				RotationPlottingFunction(X, Y, BulgeRotationField, DiscRotationField, TotalRotation, MinimumRotation, MaximumRotation, \
					EffectiveRadius, 1-Ellipticity, filename = entry+str(GalName)+'_ModelVelocity.pdf')
				DispersionPlottingFunction(X, Y, BulgeDispersion, DiscDispersion, TotalDispersion, MinimumDispersion, MaximumDispersion, \
					EffectiveRadius, 1-Ellipticity, filename = entry+str(GalName)+'_ModelDispersion.pdf')
				
				fig=plt.figure(figsize=(5, 5))
				ax1=fig.add_subplot(211)
				ax2 = fig.add_subplot(212)
				
				MaximumRadius = abs(np.max(X))
				
				x = np.arange(1, MaximumRadius, 2)
		
				y_bulge = BulgeIntensityFunction(log_I_Bulge, x, Re_Bulge, n)
				y_disc = DiscIntensityFunction(log_I_Disc, x, Re_Disc)
				
				y_bulge[np.where(y_bulge < 0)] = 0
				y_disc[np.where(y_disc < 0)] = 0
				
				ax1.plot(x, -2.5*np.log10(y_bulge), c = 'orange', label = 'Bulge')
				ax1.plot(x, -2.5*np.log10(y_disc), c = 'b', label = 'Disc')
				ax1.plot(x, -2.5*np.log10(y_disc + y_bulge), c = 'k', label = 'Total')
				
				if not Magneticum:
					# check if the Spitzer profiles exist:
					if os.path.exists('SpitzerProfiles/ngc'+GalName.split('C')[1]+'*logscale.ell'): 
						Spitzer_Radius, Spitzer_Mag, Spitzer_MagErr = SpitzerMagProfileFinder(GalName, 'SpitzerProfiles/')
						ax1.plot(Spitzer_Radius, Spitzer_Mag, c = 'gray', label = 'Spitzer', linestyle = '--')
						ax1.fill_between(Spitzer_Radius, Spitzer_Mag-Spitzer_MagErr, Spitzer_Mag+Spitzer_MagErr, color = 'gray', alpha = 0.8)
					else:
						print 'Spitzer luminosity profile does not exist'
				
				ax2.plot(Radius_EllipProfile, Ellipticity_EllipProfile)
				ax2.set_ylabel(r'$\epsilon$')
				ax1.set_xticklabels([])	
				ax1.set_ylim([22, 10])	
				ax1.set_xlim([0, MaximumRadius])
				ax1.set_xlim([0, MaximumRadius])
				
				ax2.set_xlabel(r'$r$')
				ax1.set_ylabel('magnitude')
				
				handles, labels=ax1.get_legend_handles_labels()
				ax1.legend(handles, labels, loc=4, fontsize=8, scatterpoints = 1)
				
				plt.subplots_adjust(hspace = 0., wspace = 0.2)
				OutputFilename = entry+str(GalName)+'_IntensityProfile.pdf'
				plt.savefig(OutputFilename)
		else:
			print 'no text file in this folder'
			plt.close()			