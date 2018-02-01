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

OutputFilename = DropboxDirectory+'Dropbox/PhD_Analysis/Analysis/Angular Momentum/Mock_Kinematics/'+str(GalName)+'/'+str(GalName)+'_MCMCOutput.dat'

# now looking at the output of the MCMC and analysing it. 
# depending on how old the output file is, I may or may not have saved the acceptance fraction. 
# try:
# 	fileIn = open(OutputFilename, 'rb')
# 	chain, flatchain, lnprobability, flatlnprobability, acceptanceFraction = pickle.load(fileIn) 
# 	# I don't want to bother running this code if the acceptance fraction is zero. 
# 	# if acceptanceFraction == 0.:
# 	# 	print 'Acceptance Fraction is zero. Output is garbage!'
# 	# 	sys.exit(0)
# 	# else:
# 	# 	print 'Acceptance Fraction:', acceptanceFraction
# except:
fileIn = open(OutputFilename, 'rb')
chain, flatchain, lnprobability, flatlnprobability, = pickle.load(fileIn) 
fileIn.close()

params = [r"$\epsilon_b$", \
		r"$I_b$", \
		r"$n$", \
		r"$R_{e,b}$", \
		"BulgeRotScale", \
		r"$v_b$", \
		r"$\sigma_{c,b}$", \
		r"$\alpha_b$", \
		# r"$\beta_b$", \
		# r"$\gamma_b$", \
		# r"$\epsilon_d$", \
		r"$I_d$", \
		r"$R_{e,d}$", 
		"DiscRotScale", \
		r"$v_d$", \
		r"$\sigma_{c,d}$", \
		r"$\alpha_d$"]
		# r"$\beta_d$"]
		# r"$\gamma_d$"]

triangleFilename = DropboxDirectory+'Dropbox/PhD_Analysis/Analysis/Angular Momentum/Mock_Kinematics/'+str(GalName)+'/'+str(GalName)+'_MCMCOutput_triangle.pdf'

c = ChainConsumer().add_chain(flatchain, parameters=params)
c.configure(statistics='max_shortest', summary=True)
fig = c.plotter.plot(figsize = 'PAGE', filename = triangleFilename)





ellipticity_bulge, ellipticity_bulge_lower, ellipticity_bulge_upper = parameterExtractor(c.analysis.get_summary(), '$\epsilon_b$')
I_Bulge, I_Bulge_lower, I_Bulge_upper = parameterExtractor(c.analysis.get_summary(), '$I_b$')
n, n_lower, n_upper = parameterExtractor(c.analysis.get_summary(), '$n$')
Re_Bulge, Re_Bulge_lower, Re_Bulge_upper = parameterExtractor(c.analysis.get_summary(), '$R_{e,b}$')
BulgeRotationScale, BulgeRotationScale_lower, BulgeRotationScale_upper = parameterExtractor(c.analysis.get_summary(), 'BulgeRotScale')
Max_vel_bulge, Max_vel_bulge_lower, Max_vel_bulge_upper = parameterExtractor(c.analysis.get_summary(), '$v_b$')
CentralBulgeDispersion, CentralBulgeDispersion_lower, CentralBulgeDispersion_upper = parameterExtractor(c.analysis.get_summary(), r'$\sigma_{c,b}$')
alpha_Bulge, alpha_Bulge_lower, alpha_Bulge_upper = parameterExtractor(c.analysis.get_summary(), r'$\alpha_b$')
# beta_Bulge, beta_Bulge_lower, beta_Bulge_upper = parameterExtractor(c.analysis.get_summary(), r'$\beta_b$')
# gamma_Bulge, gamma_Bulge_lower, gamma_Bulge_upper = parameterExtractor(c.analysis.get_summary(), r'$\gamma_b$')

# ellipticity_disc, ellipticity_disc_lower, ellipticity_disc_upper = parameterExtractor(c.analysis.get_summary(), '$\epsilon_d$')
I_Disc, I_Disc_lower, I_Disc_upper = parameterExtractor(c.analysis.get_summary(), '$I_d$')
h, h_lower, h_upper = parameterExtractor(c.analysis.get_summary(), '$R_{e,d}$')
DiscRotationScale, DiscRotationScale_lower, DiscRotationScale_upper = parameterExtractor(c.analysis.get_summary(), 'DiscRotScale')
Max_vel_disc, Max_vel_disc_lower, Max_vel_disc_upper = parameterExtractor(c.analysis.get_summary(), '$v_d$')
CentralDiscDispersion, CentralDiscDispersion_lower, CentralDiscDispersion_upper = parameterExtractor(c.analysis.get_summary(), r'$\sigma_{c,d}$')
alpha_Disc, alpha_Disc_lower, alpha_Disc_upper = parameterExtractor(c.analysis.get_summary(), r'$\alpha_d$')
# beta_Disc, beta_Disc_lower, beta_Disc_upper = parameterExtractor(c.analysis.get_summary(), r'$\beta_d$')
# gamma_Disc, gamma_Disc_lower, gamma_Disc_upper = parameterExtractor(c.analysis.get_summary(), r'$\gamma_d$')

ObservedGalaxyInput_Path = os.path.abspath(DropboxDirectory+'Dropbox/PhD_Analysis/Analysis/Angular Momentum/Mock_Kinematics')+'/'
X, Y, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed = krigingFileReadAll(ObservedGalaxyInput_Path, GalName)

PA = 90*np.pi/180
phi=(PA-np.pi/2.0) # accounting for the different 0 PA convention in astronomy to mathematics

EffectiveRadius = Reff_Spitzer[GalName]
ObservedEllipticity = 1 - b_a[GalName]

BulgeIntensity_ellipticityTest = BulgeIntensityFunction(I_Bulge, EffectiveRadius, Re_Bulge, n)
DiscIntensity_ellipticityTest = DiscIntensityFunction(I_Disc, EffectiveRadius, h)

BulgeFraction_ellipticityTest = BulgeIntensity_ellipticityTest / (BulgeIntensity_ellipticityTest + DiscIntensity_ellipticityTest)
DiscFraction_ellipticityTest = DiscIntensity_ellipticityTest / (BulgeIntensity_ellipticityTest + DiscIntensity_ellipticityTest)

ellipticity_disc = (ObservedEllipticity - BulgeFraction_ellipticityTest*ellipticity_bulge) / DiscFraction_ellipticityTest
Radius_Bulge, Radius_Disc = ComponentRadiusFunction(X, Y, phi, ellipticity_bulge, ellipticity_disc)

'''
set up the phyical distribution of points from a bulge component with a sersic profile
'''
BulgeIntensity = BulgeIntensityFunction(I_Bulge, Radius_Bulge, Re_Bulge, n)

'''
set up the phyical distribution of points from a disc component with an exponential profile
'''
DiscIntensity = DiscIntensityFunction(I_Disc, Radius_Disc, h)

'''
building mock rotational maps. 
'''
Angles = AnglesFunction(X, Y)

# transforming the Angular term
AngularTerm = []
for angle in Angles:
	if ((angle >= 0) & (angle <= 90)):
		AngularTerm.append( np.sin(radian(angle - 90))+1 )
	elif ((angle > 90) & (angle <= 180)):
		AngularTerm.append( np.sin(radian(angle +90))+1 )
	elif ((angle > 180) & (angle <= 270)):
		AngularTerm.append( np.sin(radian(angle - 90))-1 )
	elif ((angle > 270) & (angle <= 360)):
		AngularTerm.append( np.sin(radian(angle + 90))-1 )

AngularTerm = np.array(AngularTerm)

DiscRotation = (Max_vel_disc* Radius_Disc / (DiscRotationScale+ Radius_Disc) ) * AngularTerm
BulgeRotation = (Max_vel_bulge* Radius_Bulge / (BulgeRotationScale+ Radius_Bulge) ) * AngularTerm
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

sizeMapx, sizeMapy = 80, 80
X = np.array(X).reshape(sizeMapx,sizeMapy)
Y = np.array(Y).reshape(sizeMapx,sizeMapy)
BulgeIntensity = np.array(BulgeIntensity).reshape(sizeMapx,sizeMapy)
DiscIntensity = np.array(DiscIntensity).reshape(sizeMapx,sizeMapy)
DiscRotation = np.array(DiscRotation).reshape(sizeMapx,sizeMapy)
BulgeRotation = np.array(BulgeRotation).reshape(sizeMapx,sizeMapy)
TotalRotation = np.array(TotalRotation).reshape(sizeMapx,sizeMapy)
DiscDispersion = np.array(DiscDispersion).reshape(sizeMapx,sizeMapy)
BulgeDispersion = np.array(BulgeDispersion).reshape(sizeMapx,sizeMapy)
TotalDispersion = np.array(TotalDispersion).reshape(sizeMapx,sizeMapy)
Angles = np.array(Angles).reshape(sizeMapx,sizeMapy)


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
BulgeIntensity_extrapolatedIntensity = BulgeIntensityFunction(I_Bulge, Radius_Bulge_extrapolatedIntensity, Re_Bulge, n)
DiscIntensity_extrapolatedIntensity = DiscIntensityFunction(I_Disc, Radius_Disc_extrapolatedIntensity, h)

# calculating the bulge/total ratio
print 'Bulge/Total:', np.sum(BulgeIntensity_extrapolatedIntensity) / ( np.sum(BulgeIntensity_extrapolatedIntensity) + np.sum(DiscIntensity_extrapolatedIntensity) )
print 'Bulge/Disc:', np.sum(BulgeIntensity_extrapolatedIntensity) / np.sum(DiscIntensity_extrapolatedIntensity)

TotalIntensity_extrapolatedIntensity = BulgeIntensity_extrapolatedIntensity + DiscIntensity_extrapolatedIntensity

# first do a coarse iteration to identify the half light radius
percentageDiff = 100
percentage = 0
for testRadius in np.arange(50, 200, 20):
	if percentage < 0.55: # don't bother continuing this process for radii that are larger than Re
		# first identify the pixel that is closest in radius to the test radius
		testIntensity = TotalIntensity_extrapolatedIntensity[np.where((Y_extrapolatedIntensity == 0) & \
			(X_extrapolatedIntensity > (testRadius - 1)) & (X_extrapolatedIntensity < (testRadius + 1)))][0]
		# now identifying the radius of the associated intensity along the semiminor axis
		# will have to iterate to identify the closest value, given the finite resolution
		diff = 100000
		for testMinorRadius in np.arange(5, testRadius, 4):
			minorIntensity = TotalIntensity_extrapolatedIntensity[np.where((X_extrapolatedIntensity == 0) & (Y_extrapolatedIntensity == testMinorRadius))]
			if abs(testIntensity - minorIntensity) < diff:
				diff = abs(testIntensity - minorIntensity)
				minorRadius = testMinorRadius
		testEllipticity = 1.-(float(minorRadius)/testRadius)
	
		referenceRadius = []
		for ii in range(len(X_extrapolatedIntensity)):
			referenceRadius.append(radius(X_extrapolatedIntensity[ii], Y_extrapolatedIntensity[ii], 0, 0, phi, testEllipticity))
		referenceRadius = np.array(referenceRadius)
	
		testEffectiveRadius = referenceRadius[np.where( (Y_extrapolatedIntensity == 0) & (X_extrapolatedIntensity == testRadius) )]
		percentage = np.sum(TotalIntensity_extrapolatedIntensity[np.where(referenceRadius <= testEffectiveRadius)]) / np.sum(TotalIntensity_extrapolatedIntensity)
		print 'testRadius:', testRadius, 'percentage:', percentage
		if abs(percentage - 0.5) < percentageDiff:
			percentageDiff = abs(percentage - 0.5)
			coarseRadius = testRadius

# now to do the exact same thing but on a finer scale
percentageDiff = 100
percentage = 0
for testRadius in np.arange(coarseRadius-10, coarseRadius+10, 5):
	if percentage < 0.55: # don't bother continuing this process for radii that are larger than Re
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
		
		referenceRadius = []
		for ii in range(len(X_extrapolatedIntensity)):
			referenceRadius.append(radius(X_extrapolatedIntensity[ii], Y_extrapolatedIntensity[ii], 0, 0, phi, testEllipticity))
		referenceRadius = np.array(referenceRadius)
	
		testEffectiveRadius = referenceRadius[np.where( (Y_extrapolatedIntensity == 0) & (X_extrapolatedIntensity == testRadius) )][0]
		percentage = np.sum(TotalIntensity_extrapolatedIntensity[np.where(referenceRadius <= testEffectiveRadius)]) / np.sum(TotalIntensity_extrapolatedIntensity)
		print 'testRadius:', testRadius, 'percentage:', percentage, 'ellipticity:', testEllipticity
		if abs(percentage - 0.5) < percentageDiff:
			percentageDiff = abs(percentage - 0.5)
			Ellipticity = testEllipticity
			EffectiveRadius = testEffectiveRadius

print 'Observed Ellipticity:', ObservedEllipticity, 'model Ellipticity:', Ellipticity
print 'Observed Effective Radius', Reff_Spitzer[GalName], 'model Effective Radius:', EffectiveRadius



Linewidth_parameter = 0.8

filenamePrefix = DropboxDirectory+'Dropbox/PhD_Analysis/Analysis/Angular Momentum/Mock_Kinematics/'+str(GalName)+'/'

ObservedGalaxyInput_Path = os.path.abspath(DropboxDirectory+'Dropbox/PhD_Analysis/Analysis/Angular Momentum/Mock_Kinematics')+'/'
X_Observed, Y_Observed, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed = krigingFileReadAll(ObservedGalaxyInput_Path, GalName)

Vel_Observed = np.array(Vel_Observed).reshape(sizeMapx,sizeMapy)
VelDisp_Observed = np.array(VelDisp_Observed).reshape(sizeMapx,sizeMapy)

IntensityPlottingFunction(X, Y, BulgeIntensity, DiscIntensity, TotalIntensity, sizeMapx, sizeMapy, phi, Ellipticity, \
	EffectiveRadius, filename = filenamePrefix+str(GalName)+'_MCMCOutput_Photometry.pdf')
RotationPlottingFunction(X, Y, BulgeRotationField, DiscRotationField, TotalRotation, Vel_Observed, MinimumRotation, MaximumRotation, \
	EffectiveRadius, 1-Ellipticity, filename = filenamePrefix+str(GalName)+'_MCMCOutput_Velocity.pdf')
DispersionPlottingFunction(X, Y, BulgeDispersion, DiscDispersion, TotalDispersion, VelDisp_Observed, MinimumDispersion, MaximumDispersion, \
	EffectiveRadius, 1-Ellipticity, filename = filenamePrefix+str(GalName)+'_MCMCOutput_Dispersion.pdf')



fig=plt.figure(figsize=(5, 5))
ax1=fig.add_subplot(111)

x = np.arange(0, 140, 2)
# using a gNFW approach to describe the velocity dispersion
y_bulge = CentralBulgeDispersion * x**(-alpha_Bulge)
y_disc = CentralDiscDispersion * x**(-alpha_Disc)


y_bulge[np.where(y_bulge < 0)] = 0
y_disc[np.where(y_disc < 0)] = 0

ax1.plot(x, y_bulge, c = 'orange', label = 'Bulge')
ax1.plot(x, y_disc, c = 'b', label = 'Disc')


ax1.set_xlabel(r'$r$')
ax1.set_ylabel(r'$\sigma$ [km/s]')

handles, labels=ax1.get_legend_handles_labels()
ax1.legend(handles, labels, loc=4, fontsize=8, scatterpoints = 1)

plt.subplots_adjust(hspace = 0., wspace = 0.2)
OutputFilename = DropboxDirectory+'Dropbox/PhD_Analysis/Analysis/Angular Momentum/Mock_Kinematics/'+str(GalName)+'/'+str(GalName)+'_OutputDispersionProfile.pdf'
plt.savefig(OutputFilename)
plt.close()

fig=plt.figure(figsize=(5, 5))
ax1=fig.add_subplot(111)

x = np.arange(0, 140, 2)
# using a gNFW approach to describe the velocity dispersion
y_bulge = BulgeIntensityFunction(I_Bulge, x, Re_Bulge, n)
y_disc = DiscIntensityFunction(I_Disc, x, h)


y_bulge[np.where(y_bulge < 0)] = 0
y_disc[np.where(y_disc < 0)] = 0

ax1.plot(x, y_bulge, c = 'orange', label = 'Bulge')
ax1.plot(x, y_disc, c = 'b', label = 'Disc')
ax1.plot(x, y_disc + y_bulge, c = 'k', label = 'Total')

ax1.set_yscale('log')

ax1.set_xlabel(r'$r$')
ax1.set_ylabel(r'$\sigma$ [km/s]')

handles, labels=ax1.get_legend_handles_labels()
ax1.legend(handles, labels, loc=4, fontsize=8, scatterpoints = 1)

plt.subplots_adjust(hspace = 0., wspace = 0.2)
OutputFilename = DropboxDirectory+'Dropbox/PhD_Analysis/Analysis/Angular Momentum/Mock_Kinematics/'+str(GalName)+'/'+str(GalName)+'_IntensityProfile.pdf'
plt.savefig(OutputFilename)
plt.close()