'''

'''

# retrieve dictionaries
import os, sys, random, time, pickle
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

def parameterExtractor(inputDict, name):
	value = inputDict[name][1]
	lower = inputDict[name][1] - inputDict[name][0]
	upper = inputDict[name][2] - inputDict[name][1]
	print 'Parameter extraction successful for', name
	return value, lower, upper

GalName = 'NGC1023'
print GalName

OutputFileLocation = DropboxDirectory+'Dropbox/PhD_Analysis/Analysis/Angular Momentum/Mock_Kinematics/'+str(GalName)+'/'
ObservedGalaxyInput_Path = os.path.abspath(DropboxDirectory+'Dropbox/PhD_Analysis/Analysis/Angular Momentum/Mock_Kinematics')+'/'

TwoDatasets = FalseΩ

if TwoDatasets:
	OutputFilename = OutputFileLocation+str(GalName)+'_MCMCOutput_TwoDatasets.dat'
	
	# now looking at the output of the MCMC and analysing it. 

	fileIn = open(OutputFilename, 'rb')
	chain, flatchain, lnprobability, flatlnprobability, = pickle.load(fileIn) 
	fileIn.close()
	
	params = [r"$\epsilon_b$", \
			"BulgeRotScale", \
			r"$v_b$", \
			r"$\sigma_{c,b}$", \
			r"$\alpha_b$", \
			r"$\log(I_d)$", \
			r"$R_{e,d}$", 
			"DiscRotScale", \
			r"$v_d$", \
			r"$\sigma_{c,d}$", \
			r"$\alpha_d$", \
			r"$\theta_{\rm b}$", \
			r"$\theta_{\rm d}$", \
			r"$\omega_{\rm SLUGGS}$", \
			r"$\omega_{\rm ATLAS}$"]
	
	triangleFilename = OutputFileLocation+str(GalName)+'_MCMCOutput_triangle_TwoDatasets.pdf'
	
	c = ChainConsumer().add_chain(flatchain, parameters=params)
	c.configure(statistics='max_shortest', summary=True)
	fig = c.plotter.plot(figsize = 'PAGE', filename = triangleFilename)
	
	
	
	
	
	ellipticity_bulge, ellipticity_bulge_lower, ellipticity_bulge_upper = parameterExtractor(c.analysis.get_summary(), '$\epsilon_b$')
	BulgeRotationScale, BulgeRotationScale_lower, BulgeRotationScale_upper = parameterExtractor(c.analysis.get_summary(), 'BulgeRotScale')
	Max_vel_bulge, Max_vel_bulge_lower, Max_vel_bulge_upper = parameterExtractor(c.analysis.get_summary(), '$v_b$')
	CentralBulgeDispersion, CentralBulgeDispersion_lower, CentralBulgeDispersion_upper = parameterExtractor(c.analysis.get_summary(), r'$\sigma_{c,b}$')
	alpha_Bulge, alpha_Bulge_lower, alpha_Bulge_upper = parameterExtractor(c.analysis.get_summary(), r'$\alpha_b$')
	
	log_I_Disc, log_I_Disc_lower, log_I_Disc_upper = parameterExtractor(c.analysis.get_summary(), r'$\log(I_d)$')
	Re_Disc, Re_Disc_lower, Re_Disc_upper = parameterExtractor(c.analysis.get_summary(), '$R_{e,d}$')
	DiscRotationScale, DiscRotationScale_lower, DiscRotationScale_upper = parameterExtractor(c.analysis.get_summary(), 'DiscRotScale')
	Max_vel_disc, Max_vel_disc_lower, Max_vel_disc_upper = parameterExtractor(c.analysis.get_summary(), '$v_d$')
	CentralDiscDispersion, CentralDiscDispersion_lower, CentralDiscDispersion_upper = parameterExtractor(c.analysis.get_summary(), r'$\sigma_{c,d}$')
	alpha_Disc, alpha_Disc_lower, alpha_Disc_upper = parameterExtractor(c.analysis.get_summary(), r'$\alpha_d$')
	
	AzimuthVariationParameterBulge, AzimuthVariationParameterBulge_lower, AzimuthVariationParameterBulge_upper = \
		parameterExtractor(c.analysis.get_summary(), r'$\theta_{\rm b}$')
	AzimuthVariationParameterDisc, AzimuthVariationParameterDisc_lower, AzimuthVariationParameterDisc_upper = \
		parameterExtractor(c.analysis.get_summary(), r'$\theta_{\rm d}$')

	sluggsWeight, sluggsWeight_lower, sluggsWeight_upper = parameterExtractor(c.analysis.get_summary(), r'$\omega_{\rm SLUGGS}$')
	atlasWeight, atlasWeight_lower, atlasWeight_upper = parameterExtractor(c.analysis.get_summary(), r'$\omega_{\rm ATLAS}$')
	
	
	PA = 90*np.pi/180
	phi=(PA-np.pi/2.0) # accounting for the different 0 PA convention in astronomy to mathematics
	
	EffectiveRadius = Reff_Spitzer[GalName]
	ObservedEllipticity = 1 - b_a[GalName]
	n = SersicIndex_Bulge[GalName]
	Re_Bulge = EffectiveRadius_Bulge[GalName]
	mag_Bulge = MagnitudeRe_Bulge[GalName]
	
	# X, Y, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed = krigingFileReadAll(ObservedGalaxyInput_Path, GalName)
	#creating the 2D grid over which to generate the galaxy
	gridSizex = int(4*EffectiveRadius/np.sqrt(1-ObservedEllipticity))
	gridSizey = int(4*EffectiveRadius*np.sqrt(1-ObservedEllipticity))
	X, Y = [], []
	for xx in np.arange(-gridSizex, gridSizex, 2):
		for yy in np.arange(-gridSizey, gridSizey, 2):
			X.append(xx)
			Y.append(yy)
	X = np.array(X)
	Y = np.array(Y)
	
	
	# saving all the extracted parameters to an output file
	Parameters = np.c_[ellipticity_bulge, ellipticity_bulge_lower, ellipticity_bulge_upper, \
	# log_I_Bulge, log_I_Bulge_lower, log_I_Bulge_upper, \
	BulgeRotationScale, BulgeRotationScale_lower, BulgeRotationScale_upper, Max_vel_bulge, Max_vel_bulge_lower, Max_vel_bulge_upper, \
	CentralBulgeDispersion, CentralBulgeDispersion_lower, CentralBulgeDispersion_upper, alpha_Bulge, alpha_Bulge_lower, alpha_Bulge_upper, \
	log_I_Disc, log_I_Disc_lower, log_I_Disc_upper, Re_Disc, Re_Disc_lower, Re_Disc_upper, DiscRotationScale, DiscRotationScale_lower, DiscRotationScale_upper, \
	Max_vel_disc, Max_vel_disc_lower, Max_vel_disc_upper, CentralDiscDispersion, CentralDiscDispersion_lower, CentralDiscDispersion_upper, \
	alpha_Disc, alpha_Disc_lower, alpha_Disc_upper, \
	AzimuthVariationParameterBulge, AzimuthVariationParameterBulge_lower, AzimuthVariationParameterBulge_upper, \
	AzimuthVariationParameterDisc, AzimuthVariationParameterDisc_lower, AzimuthVariationParameterDisc_upper, sluggsWeight, sluggsWeight_lower, sluggsWeight_upper, \
	atlasWeight, atlasWeight_lower, atlasWeight_upper]
	Parameter_Header = 'e_bulge (- +), \tBulgeRotationScale (- +), \tMax_vel_bulge (- +), \tCentralBulgeDispersion (- +), \talpha_Bulge (- +), \
	\tlog_I_Disc (- +), \tRe_Disc (- +), \tDiscRotationScale (- +), \tMax_vel_disc (- +), \tCentralDiscDispersion (- +), \talpha_Disc (- +), \
	\tAzimuthVariationParameterBulge (- +), \tAzimuthVariationParameterDisc (- +), \tHyperparameterSLUGGS (- +), \tHyperparameterATLAS3D (- +)'
	
	# print np.c_(Parameters)
	np.savetxt(OutputFileLocation+GalName+'_Parameters_TwoDatasets.txt', Parameters, header = Parameter_Header)
else:

	OutputFilename = OutputFileLocation+str(GalName)+'_MCMCOutput.dat'
	
	# now looking at the output of the MCMC and analysing it. 

	fileIn = open(OutputFilename, 'rb')
	chain, flatchain, lnprobability, flatlnprobability, = pickle.load(fileIn) 
	fileIn.close()
	
	params = [r"$\epsilon_b$", \
			"BulgeRotScale", \
			r"$v_b$", \
			r"$\sigma_{c,b}$", \
			r"$\alpha_b$", \
			r"$\log(I_d)$", \
			r"$R_{e,d}$", 
			"DiscRotScale", \
			r"$v_d$", \
			r"$\sigma_{c,d}$", \
			r"$\alpha_d$", \
			r"$\theta_{\rm b}$", \
			r"$\theta_{\rm d}$"]
	
	triangleFilename = OutputFileLocation+str(GalName)+'_MCMCOutput_triangle.pdf'
	
	c = ChainConsumer().add_chain(flatchain, parameters=params)
	c.configure(statistics='max_shortest', summary=True)
	fig = c.plotter.plot(figsize = 'PAGE', filename = triangleFilename)
	
	
	
	
	
	ellipticity_bulge, ellipticity_bulge_lower, ellipticity_bulge_upper = parameterExtractor(c.analysis.get_summary(), '$\epsilon_b$')
	BulgeRotationScale, BulgeRotationScale_lower, BulgeRotationScale_upper = parameterExtractor(c.analysis.get_summary(), 'BulgeRotScale')
	Max_vel_bulge, Max_vel_bulge_lower, Max_vel_bulge_upper = parameterExtractor(c.analysis.get_summary(), '$v_b$')
	CentralBulgeDispersion, CentralBulgeDispersion_lower, CentralBulgeDispersion_upper = parameterExtractor(c.analysis.get_summary(), r'$\sigma_{c,b}$')
	alpha_Bulge, alpha_Bulge_lower, alpha_Bulge_upper = parameterExtractor(c.analysis.get_summary(), r'$\alpha_b$')
	
	log_I_Disc, log_I_Disc_lower, log_I_Disc_upper = parameterExtractor(c.analysis.get_summary(), r'$\log(I_d)$')
	Re_Disc, Re_Disc_lower, Re_Disc_upper = parameterExtractor(c.analysis.get_summary(), '$R_{e,d}$')
	DiscRotationScale, DiscRotationScale_lower, DiscRotationScale_upper = parameterExtractor(c.analysis.get_summary(), 'DiscRotScale')
	Max_vel_disc, Max_vel_disc_lower, Max_vel_disc_upper = parameterExtractor(c.analysis.get_summary(), '$v_d$')
	CentralDiscDispersion, CentralDiscDispersion_lower, CentralDiscDispersion_upper = parameterExtractor(c.analysis.get_summary(), r'$\sigma_{c,d}$')
	alpha_Disc, alpha_Disc_lower, alpha_Disc_upper = parameterExtractor(c.analysis.get_summary(), r'$\alpha_d$')
	
	AzimuthVariationParameterBulge, AzimuthVariationParameterBulge_lower, AzimuthVariationParameterBulge_upper = \
		parameterExtractor(c.analysis.get_summary(), r'$\theta_{\rm b}$')
	AzimuthVariationParameterDisc, AzimuthVariationParameterDisc_lower, AzimuthVariationParameterDisc_upper = \
		parameterExtractor(c.analysis.get_summary(), r'$\theta_{\rm d}$')
	
	
	PA = 90*np.pi/180
	phi=(PA-np.pi/2.0) # accounting for the different 0 PA convention in astronomy to mathematics
	
	EffectiveRadius = Reff_Spitzer[GalName]
	ObservedEllipticity = 1 - b_a[GalName]
	n = SersicIndex_Bulge[GalName]
	Re_Bulge = EffectiveRadius_Bulge[GalName]
	mag_Bulge = MagnitudeRe_Bulge[GalName]
	
	# X, Y, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed = krigingFileReadAll(ObservedGalaxyInput_Path, GalName)
	#creating the 2D grid over which to generate the galaxy
	gridSizex = int(4*EffectiveRadius/np.sqrt(1-ObservedEllipticity))
	gridSizey = int(4*EffectiveRadius*np.sqrt(1-ObservedEllipticity))
	X, Y = [], []
	for xx in np.arange(-gridSizex, gridSizex, 2):
		for yy in np.arange(-gridSizey, gridSizey, 2):
			X.append(xx)
			Y.append(yy)
	X = np.array(X)
	Y = np.array(Y)
	
	
	# saving all the extracted parameters to an output file
	Parameters = np.c_[ellipticity_bulge, ellipticity_bulge_lower, ellipticity_bulge_upper, \
	# log_I_Bulge, log_I_Bulge_lower, log_I_Bulge_upper, \
	BulgeRotationScale, BulgeRotationScale_lower, BulgeRotationScale_upper, Max_vel_bulge, Max_vel_bulge_lower, Max_vel_bulge_upper, \
	CentralBulgeDispersion, CentralBulgeDispersion_lower, CentralBulgeDispersion_upper, alpha_Bulge, alpha_Bulge_lower, alpha_Bulge_upper, \
	log_I_Disc, log_I_Disc_lower, log_I_Disc_upper, Re_Disc, Re_Disc_lower, Re_Disc_upper, DiscRotationScale, DiscRotationScale_lower, DiscRotationScale_upper, \
	Max_vel_disc, Max_vel_disc_lower, Max_vel_disc_upper, CentralDiscDispersion, CentralDiscDispersion_lower, CentralDiscDispersion_upper, \
	alpha_Disc, alpha_Disc_lower, alpha_Disc_upper, \
	AzimuthVariationParameterBulge, AzimuthVariationParameterBulge_lower, AzimuthVariationParameterBulge_upper, \
	AzimuthVariationParameterDisc, AzimuthVariationParameterDisc_lower, AzimuthVariationParameterDisc_upper]
	Parameter_Header = 'e_bulge (- +), \tBulgeRotationScale (- +), \tMax_vel_bulge (- +), \tCentralBulgeDispersion (- +), \talpha_Bulge (- +), \
	\tlog_I_Disc (- +), \tRe_Disc (- +), \tDiscRotationScale (- +), \tMax_vel_disc (- +), \tCentralDiscDispersion (- +), \talpha_Disc (- +), \
	\tAzimuthVariationParameterBulge (- +), \tAzimuthVariationParameterDisc (- +)'
	
	# print np.c_(Parameters)
	np.savetxt(OutputFileLocation+GalName+'_Parameters.txt', Parameters, header = Parameter_Header)


# recreating the kinematics in order to make plots. 
log_I_Bulge = -mag_Bulge / 2.5

BulgeIntensity_ellipticityTest = BulgeIntensityFunction(log_I_Bulge, EffectiveRadius, Re_Bulge, n)
DiscIntensity_ellipticityTest = DiscIntensityFunction(log_I_Disc, EffectiveRadius, Re_Disc)

BulgeFraction_ellipticityTest = BulgeIntensity_ellipticityTest / (BulgeIntensity_ellipticityTest + DiscIntensity_ellipticityTest)
DiscFraction_ellipticityTest = DiscIntensity_ellipticityTest / (BulgeIntensity_ellipticityTest + DiscIntensity_ellipticityTest)

ellipticity_disc = (ObservedEllipticity - BulgeFraction_ellipticityTest*ellipticity_bulge) / DiscFraction_ellipticityTest
Radius_Bulge, Radius_Disc = ComponentRadiusFunction(X, Y, phi, ellipticity_bulge, ellipticity_disc)

'''
set up the phyical distribution of points from a bulge component with a sersic profile
'''
BulgeIntensity = BulgeIntensityFunction(log_I_Bulge, Radius_Bulge, Re_Bulge, n)

'''
set up the phyical distribution of points from a disc component with an exponential profile
'''
DiscIntensity = DiscIntensityFunction(log_I_Disc, Radius_Disc, Re_Disc)

'''
building mock rotational maps. 
'''
# transforming the Angular term
Angles = positionAngle(X, Y, 0, 0)
# AngularTerm = np.zeros(len(Angles))
# SelOne = np.where((Angles >= 0) & (Angles <= 90))
# AngularTerm[SelOne] = np.sin(radian(Angles[SelOne] - 90))+1 
# SelTwo = np.where((Angles > 90) & (Angles <= 180))
# AngularTerm[SelTwo] = np.sin(radian(Angles[SelTwo] +90))+1 
# SelThree = np.where((Angles > 180) & (Angles <= 270))
# AngularTerm[SelThree] =  np.sin(radian(Angles[SelThree] - 90))-1 
# SelFour = np.where((Angles > 270) & (Angles <= 360))
# AngularTerm[SelFour] = np.sin(radian(Angles[SelFour] + 90))-1 
# AngularTerm = AngularVariationEpsilon(Angles, AzimuthVariationParameter)

DiscRotation = (Max_vel_disc* Radius_Disc / (DiscRotationScale+ Radius_Disc) ) * AngularVariationEpsilon(Angles, AzimuthVariationParameterDisc)
BulgeRotation = (Max_vel_bulge* Radius_Bulge / (BulgeRotationScale+ Radius_Bulge) ) * AngularVariationEpsilon(Angles, AzimuthVariationParameterBulge)
BulgeFraction = BulgeIntensity/(BulgeIntensity+DiscIntensity)
DiscFraction = DiscIntensity/(BulgeIntensity+DiscIntensity)
TotalRotation = (BulgeFraction * BulgeRotation + DiscFraction * DiscRotation)



'''
Calculating the velocity dispersions
'''
BulgeDispersion = CentralBulgeDispersion * Radius_Bulge**(-alpha_Bulge)
DiscDispersion = CentralDiscDispersion * Radius_Disc**(-alpha_Disc)

# check that the dispersion profiles never go below 0
BulgeDispersion[np.where(BulgeDispersion < 0)] = 0
DiscDispersion[np.where(DiscDispersion < 0)] = 0

TotalDispersion = BulgeDispersion * BulgeFraction + DiscDispersion * DiscFraction

#calculating the central intensity of light, as a normalisation factor. 
PeakIntensity = BulgeIntensity[np.where((X == 0) & (Y == 0))] + DiscIntensity[np.where((X == 0) & (Y == 0))]

# sizeMapx, sizeMapy = 80, 80
X = np.array(X).reshape(gridSizex,gridSizey)
Y = np.array(Y).reshape(gridSizex,gridSizey)
BulgeIntensity = np.array(BulgeIntensity).reshape(gridSizex,gridSizey)
DiscIntensity = np.array(DiscIntensity).reshape(gridSizex,gridSizey)
DiscRotation = np.array(DiscRotation).reshape(gridSizex,gridSizey)
BulgeRotation = np.array(BulgeRotation).reshape(gridSizex,gridSizey)
TotalRotation = np.array(TotalRotation).reshape(gridSizex,gridSizey)
DiscDispersion = np.array(DiscDispersion).reshape(gridSizex,gridSizey)
BulgeDispersion = np.array(BulgeDispersion).reshape(gridSizex,gridSizey)
TotalDispersion = np.array(TotalDispersion).reshape(gridSizex,gridSizey)
Angles = np.array(Angles).reshape(gridSizex,gridSizey)


TotalIntensity = BulgeIntensity + DiscIntensity

# BulgeRotationField = (BulgeIntensity * BulgeRotation)/np.max(BulgeIntensity)
# DiscRotationField = (DiscIntensity * DiscRotation)/np.max(DiscIntensity)
BulgeRotationField = BulgeRotation
DiscRotationField = DiscRotation

MinimumRotation = np.min([np.min(BulgeRotationField), np.min(DiscRotationField), np.min(TotalRotation)])
MaximumRotation = np.max([np.max(BulgeRotationField), np.max(DiscRotationField), np.max(TotalRotation)])

MinimumDispersion = np.min([np.min(BulgeDispersion), np.min(DiscDispersion), np.min(TotalDispersion)])
MaximumDispersion = np.max([np.max(BulgeDispersion), np.max(DiscDispersion), np.max(TotalDispersion)])

# calculate the half-light radius for the galaxy

# in an iterative way, identify a radius along the semimajor axis, identify the radius along the semiminor axis that 
# has the same intensity, and then identify the percentage of light enclosed within the ellipse of that ellipticity. 

# NB: for this to work, it is not enough that the total luminosity is calculated as the sum of all pixels, as the luminosity 
# will extend beyond the observational field-of-view. This will need to be extrapolated somehow...

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

TotalIntensity_extrapolatedIntensity = BulgeIntensity_extrapolatedIntensity + DiscIntensity_extrapolatedIntensity

# first do a coarse iteration to identify the half light radius
Radius_EllipProfile, Ellipticity_EllipProfile = [], []
percentageDiff = 100
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
	
	referenceRadius = radiusArray(X_extrapolatedIntensity, Y_extrapolatedIntensity, phi, testEllipticity)
	
	testEffectiveRadius = referenceRadius[np.where( (Y_extrapolatedIntensity == 0) & (X_extrapolatedIntensity == testRadius) )]
	percentage = np.sum(TotalIntensity_extrapolatedIntensity[np.where( (referenceRadius <= testEffectiveRadius) & ( (X_extrapolatedIntensity != 0) &  (Y_extrapolatedIntensity != 0)) )]) / \
	np.sum(TotalIntensity_extrapolatedIntensity[np.where( ( (X_extrapolatedIntensity != 0) &  (Y_extrapolatedIntensity != 0)) )])
	if abs(percentage - 0.5) < percentageDiff:
		percentageDiff = abs(percentage - 0.5)
		coarseRadius = testRadius

Radius_EllipProfile, Ellipticity_EllipProfile = np.array(Radius_EllipProfile), np.array(Ellipticity_EllipProfile)

# now to do the exact same thing but on a finer scale
percentageDiff = 100
for testRadius in np.arange(coarseRadius-2, coarseRadius+2, 1):
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
	
	referenceRadius = radiusArray(X_extrapolatedIntensity, Y_extrapolatedIntensity, phi, testEllipticity)
	
	testEffectiveRadius = referenceRadius[np.where( (Y_extrapolatedIntensity == 0) & (X_extrapolatedIntensity == testRadius) )][0]
	percentage = np.sum(TotalIntensity_extrapolatedIntensity[np.where( (referenceRadius <= testEffectiveRadius) & ( (X_extrapolatedIntensity != 0) &  (Y_extrapolatedIntensity != 0)) )]) / \
	np.sum(TotalIntensity_extrapolatedIntensity[np.where( ( (X_extrapolatedIntensity != 0) &  (Y_extrapolatedIntensity != 0)) )])
	if abs(percentage - 0.5) < percentageDiff:
		percentageDiff = abs(percentage - 0.5)
		Ellipticity = testEllipticity
		EffectiveRadius = testEffectiveRadius

print '------------------------------------'
print 'Observed Ellipticity:', ObservedEllipticity
print 'Model Ellipticity:', Ellipticity
print 'Observed Effective Radius:', Reff_Spitzer[GalName]
print 'Model Effective Radius:', EffectiveRadius
print '------------------------------------'


Linewidth_parameter = 0.8

# X_Observed, Y_Observed, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed = krigingFileReadAll(ObservedGalaxyInput_Path, GalName)

# Vel_Observed = np.array(Vel_Observed).reshape(sizeMapx,sizeMapy)
# VelDisp_Observed = np.array(VelDisp_Observed).reshape(sizeMapx,sizeMapy)
# VelErr_Observed = np.array(VelErr_Observed).reshape(sizeMapx,sizeMapy)
# VelDispErr_Observed = np.array(VelDispErr_Observed).reshape(sizeMapx,sizeMapy)
if TwoDatasets:
	IntensityPlottingFunction(X, Y, BulgeIntensity, DiscIntensity, TotalIntensity, gridSizex, gridSizey, phi, Ellipticity, \
		EffectiveRadius, filename = OutputFileLocation+str(GalName)+'_MCMCOutput_Photometry_TwoDatasets.pdf')
	RotationPlottingFunction(X, Y, BulgeRotationField, DiscRotationField, TotalRotation, MinimumRotation, MaximumRotation, \
		EffectiveRadius, 1-Ellipticity, filename = OutputFileLocation+str(GalName)+'_MCMCOutput_Velocity_TwoDatasets.pdf')
	DispersionPlottingFunction(X, Y, BulgeDispersion, DiscDispersion, TotalDispersion, MinimumDispersion, MaximumDispersion, \
		EffectiveRadius, 1-Ellipticity, filename = OutputFileLocation+str(GalName)+'_MCMCOutput_Dispersion_TwoDatasets.pdf')
else:
	IntensityPlottingFunction(X, Y, BulgeIntensity, DiscIntensity, TotalIntensity, gridSizex, gridSizey, phi, Ellipticity, \
		EffectiveRadius, filename = OutputFileLocation+str(GalName)+'_MCMCOutput_Photometry.pdf')
	RotationPlottingFunction(X, Y, BulgeRotationField, DiscRotationField, TotalRotation, MinimumRotation, MaximumRotation, \
		EffectiveRadius, 1-Ellipticity, filename = OutputFileLocation+str(GalName)+'_MCMCOutput_Velocity.pdf')
	DispersionPlottingFunction(X, Y, BulgeDispersion, DiscDispersion, TotalDispersion, MinimumDispersion, MaximumDispersion, \
		EffectiveRadius, 1-Ellipticity, filename = OutputFileLocation+str(GalName)+'_MCMCOutput_Dispersion.pdf')

# RelativeResidualPlottingFunction(X, Y, TotalRotation, Vel_Observed, VelErr_Observed, \
# 	EffectiveRadius, 1-Ellipticity, filename = OutputFileLocation+str(GalName)+'_MCMCOutput_Velocity.pdf')
# RelativeResidualPlottingFunction(X, Y, TotalDispersion, VelDisp_Observed, VelDispErr_Observed, \
# 	EffectiveRadius, 1-Ellipticity, filename = OutputFileLocation+str(GalName)+'_MCMCOutput_Dispersion.pdf')


fig=plt.figure(figsize=(5, 5))
ax1=fig.add_subplot(211)
ax2 = fig.add_subplot(212)

MaximumRadius = abs(np.max(X))

x = np.arange(1, MaximumRadius, 2)
# using a gNFW approach to describe the velocity dispersion
y_bulge = BulgeIntensityFunction(log_I_Bulge, x, Re_Bulge, n)
y_disc = DiscIntensityFunction(log_I_Disc, x, Re_Disc)


y_bulge[np.where(y_bulge < 0)] = 0
y_disc[np.where(y_disc < 0)] = 0

ax1.plot(x, -2.5*np.log10(y_bulge), c = 'orange', label = 'Bulge')
ax1.plot(x, -2.5*np.log10(y_disc), c = 'b', label = 'Disc')
ax1.plot(x, -2.5*np.log10(y_disc + y_bulge), c = 'k', label = 'Total')

# plot the Spitzer luminosity profile
Spitzer_Radius, Spitzer_Mag, Spitzer_MagErr = SpitzerMagProfileFinder(GalName)
# print '################'
# print Spitzer_Radius, Spitzer_Luminosity, Spitzer_LuminosityErr
# print '################'

ax1.plot(Spitzer_Radius, Spitzer_Mag, c = 'gray', label = 'Spitzer', linestyle = '--')
ax1.fill_between(Spitzer_Radius, Spitzer_Mag-Spitzer_MagErr, Spitzer_Mag+Spitzer_MagErr, color = 'gray', alpha = 0.8)

ax2.plot(Radius_EllipProfile, Ellipticity_EllipProfile)
ax2.set_ylabel(r'$\epsilon$')
ax1.set_xticklabels([])

# ax1.set_yscale('log')
ax1.set_ylim([22, 10])

ax1.set_xlim([0, MaximumRadius])
ax1.set_xlim([0, MaximumRadius])

ax2.set_xlabel(r'$r$')
ax1.set_ylabel('magnitude')

handles, labels=ax1.get_legend_handles_labels()
ax1.legend(handles, labels, loc=4, fontsize=8, scatterpoints = 1)

plt.subplots_adjust(hspace = 0., wspace = 0.2)
OutputFilename = OutputFileLocation+str(GalName)+'_IntensityProfile.pdf'
plt.savefig(OutputFilename)
plt.close()