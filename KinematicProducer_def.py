# retrieve dictionaries
import os, sys, random
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.path import Path

DropboxDirectory = os.getcwd().split('Dropbox')[0]
lib_path = os.path.abspath(DropboxDirectory+'Dropbox/PhD_Analysis/Library') 
sys.path.append(lib_path)
from galaxyParametersDictionary_v9 import *
from Sabine_Define import *



def krigingFileReadAll(Kriging_path, GalName):
	dtype1 = np.dtype([('ra', 'float'), ('dec', 'float'), ('vel', '|S17'), ('velerr', '|S17')])
	Array = np.loadtxt(Kriging_path+str(GalName)+'/'+str(GalName)+'_gridKrig_vel.txt', dtype = dtype1)
	
	RA = Array['ra']
	Dec = Array['dec']
	Vel = np.asfarray(Array['vel'], dtype='float')
	VelErr = np.asfarray(Array['velerr'], dtype='float')
	
	dtype1 = np.dtype([('ra', 'float'), ('dec', 'float'), ('veldisp', '|S17'), ('veldisperr', '|S17')])
	Array = np.loadtxt(Kriging_path+str(GalName)+'/'+str(GalName)+'_gridKrig_sigma.txt', dtype = dtype1)
	
	VelDisp = np.asfarray(Array['veldisp'], dtype='float')
	VelDispErr = np.asfarray(Array['veldisperr'], dtype='float')
	return (RA, Dec, Vel, VelErr, VelDisp, VelDispErr)

def SpitzerMagProfileFinder(GalName, Spitzer_path):
    DropboxDirectory = os.getcwd().split('Dropbox')[0]
    filename = glob.glob(Spitzer_path+'/ngc'+GalName.split('C')[1]+'*logscale.ell')[0]
    # print filename
    file=open(filename, 'r')
    lines=file.readlines()
    file.close()

    Radius, Mag, Error = [], [], []
    for line in lines:
        if line[0] != '#':
            try:
                Radius.append(1.22*float(line.split()[1])) # to convert pixel scale to arcsecond scale
                Mag.append(float(line.split()[18])) 
                # Luminosity.append(10**((3.24-AbsMag)/2.5))# the solar magnitude in the Spitzer 3.6 micron band is 3.24
                Error.append(np.max([float(line.split()[19]), float(line.split()[20])])) # need to calculate
                # print float(line.split()[19])
            except:
                # print 'empty line'
                Line = True

    Radius = np.array(Radius)
    Mag = np.array(Mag)
    Error = np.array(Error)
    return Radius, Mag, Error

def BulgeIntensityFunction(log_I_Bulge, Radius_Bulge, Re_Bulge, n):
	k = 1.9992*n-0.3271
	return (10**log_I_Bulge) * np.exp(-k * ((Radius_Bulge/Re_Bulge)**(1./n)-1))

def DiscIntensityFunction(log_I_Disc, Radius_Disc, Re_Disc):
	h = Re_Disc/1.678
	return (10**log_I_Disc) * np.exp(-Radius_Disc/h)

def ComponentRadiusFunction(X, Y, phi, ellipticity_bulge, ellipticity_disc):
	Radius_Bulge = radiusArray(X, Y, phi, ellipticity_bulge)
	Radius_Disc = radiusArray(X, Y, phi, ellipticity_disc)
	return (Radius_Bulge, Radius_Disc)

def AngularVariationFlippedSine(Angles):
	AngularTerm = np.zeros(len(Angles))
	SelOne = np.where((Angles >= 0) & (Angles <= 90))
	AngularTerm[SelOne] = np.sin(radian(Angles[SelOne] - 90))+1 
	
	SelTwo = np.where((Angles > 90) & (Angles <= 180))
	AngularTerm[SelTwo] = np.sin(radian(Angles[SelTwo] +90))+1 
	
	SelThree = np.where((Angles > 180) & (Angles <= 270))
	AngularTerm[SelThree] =  np.sin(radian(Angles[SelThree] - 90))-1 
	
	SelFour = np.where((Angles > 270) & (Angles <= 360))
	AngularTerm[SelFour] = np.sin(radian(Angles[SelFour] + 90))-1 
	return AngularTerm


def AngularVariationEpsilon(Angles, epsilon):
	epsilon = 0.3 * epsilon
	AngularTerm = np.zeros(len(Angles))
	SelOne = np.where((Angles >= 0) & (Angles <= 90))
	AngularTerm[SelOne] = epsilon * np.sin(radian(2*Angles[SelOne])) + radian(Angles[SelOne])* np.tan(np.arctan(2/np.pi))
	
	SelTwo = np.where((Angles > 90) & (Angles <= 180))
	AngularTerm[SelTwo] = epsilon * np.sin(radian(2*(Angles[SelTwo] - 90))) + radian((Angles[SelTwo] - 90))* np.tan(-np.arctan(2/np.pi)) + 1
	
	SelThree = np.where((Angles > 180) & (Angles <= 270))
	AngularTerm[SelThree] =  -epsilon * np.sin(radian(2*(Angles[SelThree] - 180))) + radian((Angles[SelThree] - 180))* np.tan(-np.pi - np.arctan(2/np.pi)) 
	
	SelFour = np.where((Angles > 270) & (Angles <= 360))
	AngularTerm[SelFour] = -epsilon * np.sin(radian(2*(Angles[SelFour] - 270))) + radian((Angles[SelFour] - 270))* np.tan(np.arctan(2/np.pi)) -1
	return AngularTerm

def MockKinematicsModel(X, Y, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed, phi, ellipticity_bulge, ellipticity_disc, log_I_Bulge, Re_Bulge, n, log_I_Disc, Re_Disc, \
	Max_vel_disc, DiscRotationScale, AzimuthVariationParameterDisc, Max_vel_bulge, BulgeRotationScale, AzimuthVariationParameterBulge, \
	CentralBulgeDispersion, CentralDiscDispersion, alpha_Bulge, alpha_Disc):
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
	Angles = positionAngle(X, Y, 0, 0)		
	
	DiscRotation = (Max_vel_disc* Radius_Disc / (DiscRotationScale + Radius_Disc) ) * AngularVariationEpsilon(Angles, AzimuthVariationParameterDisc)
	BulgeRotation = (Max_vel_bulge* Radius_Bulge / (BulgeRotationScale + Radius_Bulge) ) * AngularVariationEpsilon(Angles, AzimuthVariationParameterBulge)
	BulgeFraction = BulgeIntensity/(BulgeIntensity+DiscIntensity)
	DiscFraction = DiscIntensity/(BulgeIntensity+DiscIntensity)
	TotalRotation = (BulgeFraction * BulgeRotation + DiscFraction * DiscRotation)
	
	
	'''
	Calculating the velocity dispersions
	'''
	# using the power law equation as given by Cappellari et al (2006)
	BulgeDispersion = CentralBulgeDispersion * Radius_Bulge**(-alpha_Bulge)
	DiscDispersion = CentralDiscDispersion * Radius_Disc**(-alpha_Disc)
	
	# check that the dispersion profiles never go below 0
	BulgeDispersion[np.where(BulgeDispersion < 0)] = 0
	DiscDispersion[np.where(DiscDispersion < 0)] = 0
	
	TotalDispersion = BulgeDispersion * BulgeFraction + DiscDispersion * DiscFraction
	
	#calculating the central intensity of light, as a normalisation factor. 
	# PeakIntensity = BulgeIntensity[np.where((X == 0) & (Y == 0))] + DiscIntensity[np.where((X == 0) & (Y == 0))]

	# calculating the chi-squared of the match of the rotation field and dispersion field to the observed galaxy
	ObservedSel = np.where(np.isfinite(Vel_Observed) & np.isfinite(TotalDispersion)) # avoid the values for which the model gives infinite 
																					 # velocity dispersion values at the centre of the bulge. 

	# needing to combine the velocity and velocity dispersion fits as though they were a single dataset

	ObservedData = np.append(Vel_Observed[ObservedSel], VelDisp_Observed[ObservedSel])
	ObservedUncertainties = np.append(VelErr_Observed[ObservedSel], VelDispErr_Observed[ObservedSel])
	ModelData = np.append(TotalRotation[ObservedSel], TotalDispersion[ObservedSel])

	return(ObservedData, ObservedUncertainties, ModelData)


def MockKinematicsModelAnalysis(X, Y, phi, ellipticity_bulge, ellipticity_disc, log_I_Bulge, Re_Bulge, n, log_I_Disc, Re_Disc, \
	Max_vel_disc, DiscRotationScale, AzimuthVariationParameterDisc, Max_vel_bulge, BulgeRotationScale, AzimuthVariationParameterBulge, \
	CentralBulgeDispersion, CentralDiscDispersion, alpha_Bulge, alpha_Disc, gridSizex,gridSizey): # simplified mock kinematics production for the sake of the 
																			 # analysis code. 
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
	# Angles = np.array(Angles).reshape(gridSizex,gridSizey)
	
	
	TotalIntensity = BulgeIntensity + DiscIntensity
	
	BulgeRotationField = BulgeRotation
	DiscRotationField = DiscRotation
	
	MinimumRotation = np.min([np.min(BulgeRotationField), np.min(DiscRotationField), np.min(TotalRotation)])
	MaximumRotation = np.max([np.max(BulgeRotationField), np.max(DiscRotationField), np.max(TotalRotation)])
	
	MinimumDispersion = np.min([np.min(BulgeDispersion), np.min(DiscDispersion), np.min(TotalDispersion)])
	MaximumDispersion = np.max([np.max(BulgeDispersion), np.max(DiscDispersion), np.max(TotalDispersion)])

	return (X, Y, BulgeIntensity, DiscIntensity, DiscRotation, BulgeRotation, TotalRotation, DiscDispersion, BulgeDispersion, TotalDispersion, \
		TotalIntensity, BulgeRotationField, DiscRotationField, MinimumRotation, MaximumRotation, MinimumDispersion, MaximumDispersion)

def IntensityPlottingFunction(X, Y, BulgeIntensity, DiscIntensity, TotalIntensity, sizeMapx, sizeMapy, phi, Ellipticity_measured, HalfLightRadius, Linewidth_parameter = 0.8, filename = 'BulgeDiscTest.pdf'):
	fig=plt.figure(figsize=(11, 3))
	ax1=fig.add_subplot(131, aspect = 'equal')
	ax2=fig.add_subplot(132, aspect = 'equal')
	ax3=fig.add_subplot(133, aspect = 'equal')
	ax1.pcolor(X, Y, np.log10(BulgeIntensity), cmap = 'jet', vmin=np.min(np.log10(TotalIntensity)), vmax=np.max(np.log10(TotalIntensity)))
	CS1 = ax1.contour(X, Y, np.log10(BulgeIntensity), colors='k', linewidths = Linewidth_parameter)
	ax1.clabel(CS1, fontsize=7, inline=1)
	ax1.set_title('Bulge')
	ax2.pcolor(X, Y, np.log10(DiscIntensity), cmap = 'jet', vmin=np.min(np.log10(TotalIntensity)), vmax=np.max(np.log10(TotalIntensity)))
	CS2 = ax2.contour(X, Y, np.log10(DiscIntensity), colors='k', linewidths = Linewidth_parameter)
	ax2.clabel(CS2, fontsize=7, inline=1)
	ax2.set_title('Disc')
	ax3.pcolor(X, Y, np.log10(TotalIntensity), cmap = 'jet', vmin=np.min(np.log10(TotalIntensity)), vmax=np.max(np.log10(TotalIntensity)))
	CS3 = ax3.contour(X, Y, np.log10(TotalIntensity), colors='k', linewidths = Linewidth_parameter)
	ax3.clabel(CS3, fontsize=7, inline=1)
	ax3.set_title('Total')
	
	AxialRatio = 1 - Ellipticity_measured
	radii = [1, 2, 3, 4]
	ellipses = [Ellipse(xy=[0,0], width=(2.*jj*HalfLightRadius/np.sqrt(AxialRatio)), 
		height=(2.*jj*HalfLightRadius*np.sqrt(AxialRatio)), angle=0, 
		edgecolor = 'white', facecolor = 'none', fill = False, linestyle = 'dashed', linewidth = 2) for jj in radii]
	for ee in ellipses:
		ax3.add_artist(ee)
	
	for ax in [ax2, ax3]:
		ax.set_yticklabels([])
	plt.subplots_adjust(left = 0.05, right = 0.95, hspace = 0., wspace = 0.01)
	plt.savefig(filename)
	plt.close()

	return (Ellipticity_measured, HalfLightRadius)

def RotationPlottingFunction(X, Y, BulgeRotationField, DiscRotationField, TotalRotation, MinimumRotation, MaximumRotation, \
	HalfLightRadius, AxialRatio, Linewidth_parameter = 0.8, filename = 'KinematicTest.pdf'):
	fig=plt.figure(figsize=(11, 3))
	ax1=fig.add_subplot(131, aspect = 'equal')
	ax2=fig.add_subplot(132, aspect = 'equal')
	ax3=fig.add_subplot(133, aspect = 'equal')
	ax1.pcolor(X, Y, BulgeRotationField, cmap = 'jet', vmin=MinimumRotation, vmax=MaximumRotation)
	CS1 = ax1.contour(X, Y, BulgeRotationField, colors='k', linewidths = Linewidth_parameter)
	ax1.clabel(CS1, fontsize=7, inline=1)
	ax1.set_title('Bulge Rotation')
	ax2.pcolor(X, Y, DiscRotationField, cmap = 'jet', vmin=MinimumRotation, vmax=MaximumRotation)
	CS2 = ax2.contour(X, Y, DiscRotationField, colors='k', linewidths = Linewidth_parameter)
	ax2.clabel(CS2, fontsize=7, inline=1)
	ax2.set_title('Disc Rotation')
	ax3.pcolor(X, Y, TotalRotation, cmap = 'jet', vmin=MinimumRotation, vmax=MaximumRotation)
	CS3 = ax3.contour(X, Y, TotalRotation, colors='k', linewidths = Linewidth_parameter)
	ax3.clabel(CS3, fontsize=7, inline=1)
	ax3.set_title('Total Rotation')
	radii = [1, 2, 3, 4]
	ellipses = [Ellipse(xy=[0,0], width=(2.*jj*HalfLightRadius/np.sqrt(AxialRatio)), 
		height=(2.*jj*HalfLightRadius*np.sqrt(AxialRatio)), angle=0, 
		edgecolor = 'white', facecolor = 'none', fill = False, linestyle = 'dashed', linewidth = 2) for jj in radii]
	for ee in ellipses:
		ax3.add_artist(ee)
	for ax in [ax2, ax3]:
		ax.set_yticklabels([])
	plt.subplots_adjust(left = 0.05, right = 0.95, hspace = 0., wspace = 0.01)
	plt.savefig(filename)
	plt.close()

	

def RelativeResidualPlottingFunction(X, Y, ModelProperty, ObservedProperty, ObservedPropertyErr, \
	HalfLightRadius, AxialRatio, Linewidth_parameter = 0.8, filename = 'RelativeResidual.pdf'):

	fig=plt.figure(figsize=(5, 3))
	ax1=fig.add_subplot(111, aspect = 'equal')
	ax1.pcolor(X, Y, ObservedProperty-ModelProperty, cmap = 'coolwarm', vmin=-50, vmax=50)
	CS1 = ax1.contour(X, Y, ObservedProperty-ModelProperty, colors='k', linewidths = Linewidth_parameter)
	ax1.clabel(CS1, fontsize=7, inline=1)
	ax1.set_title('Rotation Residual')

	radii = [1, 2, 3, 4]
	ellipses = [Ellipse(xy=[0,0], width=(2.*jj*HalfLightRadius/np.sqrt(AxialRatio)), 
		height=(2.*jj*HalfLightRadius*np.sqrt(AxialRatio)), angle=0, 
		edgecolor = 'white', facecolor = 'none', fill = False, linestyle = 'dashed', linewidth = 2) for jj in radii]
	for ee in ellipses:
		ax1.add_artist(ee)
	plt.subplots_adjust(left = 0.05, right = 0.95, hspace = 0., wspace = 0.01)
	plt.savefig(filename.split('.pdf')[0]+'_Residual.pdf')
	plt.close()

	fig=plt.figure(figsize=(5, 3))
	ax1=fig.add_subplot(111, aspect = 'equal')
	ax1.pcolor(X, Y, (ObservedProperty-ModelProperty) / ObservedPropertyErr, cmap = 'coolwarm', vmin=-20, vmax=20)
	CS1 = ax1.contour(X, Y, (ObservedProperty-ModelProperty) / ObservedPropertyErr, colors='k', linewidths = Linewidth_parameter)
	ax1.clabel(CS1, fontsize=7, inline=1)
	ax1.set_title('Relative Rotation Residual')

	radii = [1, 2, 3, 4]
	ellipses = [Ellipse(xy=[0,0], width=(2.*jj*HalfLightRadius/np.sqrt(AxialRatio)), 
		height=(2.*jj*HalfLightRadius*np.sqrt(AxialRatio)), angle=0, 
		edgecolor = 'white', facecolor = 'none', fill = False, linestyle = 'dashed', linewidth = 2) for jj in radii]
	for ee in ellipses:
		ax1.add_artist(ee)
	plt.subplots_adjust(left = 0.05, right = 0.95, hspace = 0., wspace = 0.01)
	plt.savefig(filename.split('.pdf')[0]+'_RelativeResidual.pdf')
	plt.close()


def DispersionPlottingFunction(X, Y, BulgeDispersion, DiscDispersion, TotalDispersion, MinimumDispersion, MaximumDispersion, \
	HalfLightRadius, AxialRatio, Linewidth_parameter = 0.8, filename = 'DispersionTest.pdf'):
	fig=plt.figure(figsize=(11, 3))
	ax1=fig.add_subplot(131, aspect = 'equal')
	ax2=fig.add_subplot(132, aspect = 'equal')
	ax3=fig.add_subplot(133, aspect = 'equal')
	ax1.pcolor(X, Y, BulgeDispersion, cmap = 'jet', vmin=MinimumDispersion, vmax=MaximumDispersion)
	ax1.set_title('Bulge Dispersion')
	CS1 = ax1.contour(X, Y, BulgeDispersion, colors='k', linewidths = Linewidth_parameter)
	ax1.clabel(CS1, fontsize=7, inline=1)
	ax2.pcolor(X, Y, DiscDispersion, cmap = 'jet', vmin=MinimumDispersion, vmax=MaximumDispersion)
	ax2.set_title('Disc Dispersion')
	CS2 = ax2.contour(X, Y, DiscDispersion, colors='k', linewidths = Linewidth_parameter)
	ax2.clabel(CS2, fontsize=7, inline=1)
	ax3.pcolor(X, Y, TotalDispersion, cmap = 'jet', vmin=MinimumDispersion, vmax=MaximumDispersion)
	CS3 = ax3.contour(X, Y, TotalDispersion, colors='k', linewidths = Linewidth_parameter)
	ax3.clabel(CS3, fontsize=7, inline=1)
	ax3.set_title('Total Dispersion')
	radii = [1, 2, 3, 4]
	ellipses = [Ellipse(xy=[0,0], width=(2.*jj*HalfLightRadius/np.sqrt(AxialRatio)), 
		height=(2.*jj*HalfLightRadius*np.sqrt(AxialRatio)), angle=0, 
		edgecolor = 'white', facecolor = 'none', fill = False, linestyle = 'dashed', linewidth = 2) for jj in radii]
	for ee in ellipses:
		ax3.add_artist(ee)
	for ax in [ax2, ax3]:
		ax.set_yticklabels([])
	plt.subplots_adjust(left = 0.05, right = 0.95, hspace = 0., wspace = 0.01)
	plt.savefig(filename)
	plt.close()

def CheckBoundaryConditions(Parameters, ParameterBoundaries):
	ParameterNumber = len(Parameters)

	# separate the boundaries array into a lower and an upper array
	# first selecting the lower bounds, which are indices 0, 2, 4...
	Indices_lower = np.arange(0, 2*ParameterNumber, 2)
	# then selecting the lower bounds, which are indices 1, 3, 5...
	Indices_upper = np.arange(1, 2*ParameterNumber, 2)

	ParameterBoundaries_lower = ParameterBoundaries[Indices_lower]
	ParameterBoundaries_upper = ParameterBoundaries[Indices_upper]

	Check = True
	for ii in range(ParameterNumber):
		if not ((Parameters[ii] <= ParameterBoundaries_upper[ii]) & (Parameters[ii] >= ParameterBoundaries_lower[ii])):
			Check = False
	return Check


def lnprior(theta, *args): 
	tmpInputArgs = args[0] 

	if CheckBoundaryConditions(np.array(theta), np.array(tmpInputArgs)):
		return 0.
	else:
		return -np.inf

def lnlike_RotationAndDispersion(theta, *args):
	try:
		ellipticity_bulge, BulgeRotationScale, Max_vel_bulge, CentralBulgeDispersion, alpha_Bulge, log_I_Disc, \
		Re_Disc, DiscRotationScale, Max_vel_disc, CentralDiscDispersion, alpha_Disc, AzimuthVariationParameterBulge, AzimuthVariationParameterDisc = theta
		tmpInputArgs = args[0]
		(X, Y, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed, \
			EffectiveRadius, ObservedEllipticity, n, Re_Bulge, mag_Bulge, Spitzer_Radius, Spitzer_Mag, Spitzer_MagErr) = tmpInputArgs
		PA = 90*np.pi/180
		phi=(PA-np.pi/2.0) # accounting for the different 0 PA convention in astronomy to mathematics
		log_I_Bulge = -mag_Bulge / 2.5
			
		'''
		in an attempt to reduce the number of free parameters, and also to use the fact that we know the projected 
		ellipticity of the modelled galaxy, here we reduce the info about the ellipticity of one component
		'''
		
		BulgeIntensity_ellipticityTest = BulgeIntensityFunction(log_I_Bulge, EffectiveRadius, Re_Bulge, n)
		DiscIntensity_ellipticityTest = DiscIntensityFunction(log_I_Disc, EffectiveRadius, Re_Disc)
		
		BulgeFraction_ellipticityTest = BulgeIntensity_ellipticityTest / (BulgeIntensity_ellipticityTest + DiscIntensity_ellipticityTest)
		DiscFraction_ellipticityTest = DiscIntensity_ellipticityTest / (BulgeIntensity_ellipticityTest + DiscIntensity_ellipticityTest)
		
		ellipticity_disc = (ObservedEllipticity - BulgeFraction_ellipticityTest*ellipticity_bulge) / DiscFraction_ellipticityTest
	
		# calculating the Chi2/dof from the fit to the luminosity profile
		# we only want to fit the profile in the region where we have actual data
		ModelMagProfile = -2.5*np.log10(BulgeIntensityFunction(log_I_Bulge, Spitzer_Radius, Re_Bulge, n) + \
			DiscIntensityFunction(log_I_Disc, Spitzer_Radius, Re_Disc))
		# print len(Spitzer_Radius), Spitzer_MagErr
		realChi2_lum = np.sum(np.log(2*np.pi*Spitzer_MagErr**2) + ((Spitzer_Mag - ModelMagProfile)/Spitzer_MagErr)**2.)
		if (ellipticity_disc > 1) | (ellipticity_disc < 0) | (np.isfinite(ellipticity_disc) == False):
			Chi2Total = np.inf
		else:		
			ObservedData, ObservedUncertainties, ModelData = MockKinematicsModel(X, Y, Vel_Observed, VelErr_Observed, \
				VelDisp_Observed, VelDispErr_Observed, phi, ellipticity_bulge, ellipticity_disc, log_I_Bulge, Re_Bulge, n, log_I_Disc, Re_Disc, \
				Max_vel_disc, DiscRotationScale, AzimuthVariationParameterDisc, Max_vel_bulge, BulgeRotationScale, AzimuthVariationParameterBulge, \
				CentralBulgeDispersion, CentralDiscDispersion, alpha_Bulge, alpha_Disc)
			
			realChi2_kin = np.sum(np.log(2*np.pi*ObservedUncertainties**2) + ((ObservedData - ModelData)/ObservedUncertainties)**2.)
			
			Chi2Total = realChi2_kin + realChi2_lum # now each of the datasets is appropriately weighted, and it is the likelihoods themselves that are being multiplied
			
	except:
		Chi2Total = np.inf
	return -Chi2Total/2


def lnprob_RotationAndDispersion(theta, *args):
	arg1 = args[0]
	arg2 = args[1:]

	lp = lnprior(theta, arg1)
	if not np.isfinite(lp):
	    return -np.inf
	return lp + lnlike_RotationAndDispersion(theta, arg2) #In logarithmic space, the multiplication becomes sum.

def lnlike_RotationAndDispersion_hyperparameters(theta, *args):
	try:
		ellipticity_bulge, BulgeRotationScale, Max_vel_bulge, CentralBulgeDispersion, alpha_Bulge, log_I_Disc, \
		Re_Disc, DiscRotationScale, Max_vel_disc, CentralDiscDispersion, alpha_Disc, AzimuthVariationParameterBulge, AzimuthVariationParameterDisc, \
		sluggsWeight, atlasWeight = theta
		tmpInputArgs = args[0]
		(X, Y, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed, \
			X_2, Y_2, Vel_Observed_2, VelErr_Observed_2, VelDisp_Observed_2, VelDispErr_Observed_2, \
			EffectiveRadius, ObservedEllipticity, n, Re_Bulge, mag_Bulge, Spitzer_Radius, Spitzer_Mag, Spitzer_MagErr) = tmpInputArgs
		PA = 90*np.pi/180
		phi=(PA-np.pi/2.0) # accounting for the different 0 PA convention in astronomy to mathematics
		log_I_Bulge = -mag_Bulge / 2.5
			
		'''
		in an attempt to reduce the number of free parameters, and also to use the fact that we know the projected 
		ellipticity of the modelled galaxy, here we reduce the info about the ellipticity of one component
		'''
		
		BulgeIntensity_ellipticityTest = BulgeIntensityFunction(log_I_Bulge, EffectiveRadius, Re_Bulge, n)
		DiscIntensity_ellipticityTest = DiscIntensityFunction(log_I_Disc, EffectiveRadius, Re_Disc)
		
		BulgeFraction_ellipticityTest = BulgeIntensity_ellipticityTest / (BulgeIntensity_ellipticityTest + DiscIntensity_ellipticityTest)
		DiscFraction_ellipticityTest = DiscIntensity_ellipticityTest / (BulgeIntensity_ellipticityTest + DiscIntensity_ellipticityTest)
		
		ellipticity_disc = (ObservedEllipticity - BulgeFraction_ellipticityTest*ellipticity_bulge) / DiscFraction_ellipticityTest
		
		# calculating the Chi2/dof from the fit to the luminosity profile
		# we only want to fit the profile in the region where we have actual data
		ModelMagProfile = -2.5*np.log10(BulgeIntensityFunction(log_I_Bulge, Spitzer_Radius, Re_Bulge, n) + \
			DiscIntensityFunction(log_I_Disc, Spitzer_Radius, Re_Disc))

		realChi2_lum = np.sum(np.log(2*np.pi*Spitzer_MagErr**2) + ((Spitzer_Mag - ModelMagProfile)/Spitzer_MagErr)**2.)
		if (ellipticity_disc > 1) | (ellipticity_disc < 0) | (np.isfinite(ellipticity_disc) == False):
			Chi2Total = np.inf
		else:		
			ObservedData_sluggs, ObservedUncertainties_sluggs, ModelData_sluggs = MockKinematicsModel(X, Y, Vel_Observed, VelErr_Observed, \
				VelDisp_Observed, VelDispErr_Observed, phi, ellipticity_bulge, ellipticity_disc, log_I_Bulge, Re_Bulge, n, log_I_Disc, Re_Disc, \
				Max_vel_disc, DiscRotationScale, AzimuthVariationParameterDisc, Max_vel_bulge, BulgeRotationScale, AzimuthVariationParameterBulge, \
				CentralBulgeDispersion, CentralDiscDispersion, alpha_Bulge, alpha_Disc)
		
			realChi2_sluggs = np.sum(np.log(2*np.pi*ObservedUncertainties_sluggs**2/sluggsWeight) + (sluggsWeight*(ObservedData_sluggs - ModelData_sluggs)/ObservedUncertainties_sluggs)**2.)
		
			ObservedData_atlas, ObservedUncertainties_atlas, ModelData_atlas = MockKinematicsModel(X_2, Y_2, Vel_Observed_2, VelErr_Observed_2, \
				VelDisp_Observed_2, VelDispErr_Observed_2, phi, ellipticity_bulge, ellipticity_disc, log_I_Bulge, Re_Bulge, n, log_I_Disc, Re_Disc, \
				Max_vel_disc, DiscRotationScale, AzimuthVariationParameterDisc, Max_vel_bulge, BulgeRotationScale, AzimuthVariationParameterBulge, \
				CentralBulgeDispersion, CentralDiscDispersion, alpha_Bulge, alpha_Disc)
		
			realChi2_atlas = np.sum(np.log(2*np.pi*ObservedUncertainties_atlas**2/atlasWeight) + (atlasWeight*(ObservedData_atlas - ModelData_atlas)/ObservedUncertainties_atlas)**2.)
			
			Chi2Total = realChi2_sluggs + realChi2_atlas + realChi2_lum # now each of the datasets is appropriately weighted, and it is the likelihoods themselves that are being multiplied
	except:
		Chi2Total = np.inf
	return -Chi2Total/2


def lnprob_RotationAndDispersion_hyperparameters(theta, *args):
	arg1 = args[0]
	arg2 = args[1:]

	lp = lnprior(theta, arg1)
	if not np.isfinite(lp):
		# print 'problem'
		return -np.inf
	# print 'some good steps'
	return lp + lnlike_RotationAndDispersion_hyperparameters(theta, arg2) #In logarithmic space, the multiplication becomes sum.

def lnlike_RotationAndDispersion_sims(theta, *args):
	try:
		log_I_Bulge, Re_Bulge, ellipticity_bulge, n, BulgeRotationScale, Max_vel_bulge, CentralBulgeDispersion, alpha_Bulge, log_I_Disc, \
		Re_Disc, ellipticity_disc, DiscRotationScale, Max_vel_disc, CentralDiscDispersion, alpha_Disc, AzimuthVariationParameterBulge, AzimuthVariationParameterDisc = theta
		tmpInputArgs = args[0]
		(X, Y, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed) = tmpInputArgs
		PA = 90*np.pi/180
		phi=(PA-np.pi/2.0) # accounting for the different 0 PA convention in astronomy to mathematics

		SimulatedData, SimulatedUncertainties, ModelData = MockKinematicsModel(X, Y, Vel_Observed, VelErr_Observed, \
			VelDisp_Observed, VelDispErr_Observed, phi, ellipticity_bulge, ellipticity_disc, log_I_Bulge, Re_Bulge, n, log_I_Disc, Re_Disc, \
			Max_vel_disc, DiscRotationScale, AzimuthVariationParameterDisc, Max_vel_bulge, BulgeRotationScale, AzimuthVariationParameterBulge, \
			CentralBulgeDispersion, CentralDiscDispersion, alpha_Bulge, alpha_Disc)
	
		Chi2Total = np.sum(((SimulatedData - ModelData)/SimulatedUncertainties)**2.)
	except:
		Chi2Total = np.inf
	return -Chi2Total/2

def lnprob_RotationAndDispersion_sims(theta, *args):
	arg1 = args[0]
	arg2 = args[1:]

	lp = lnprior(theta, arg1)
	if not np.isfinite(lp):
	    return -np.inf
	return lp + lnlike_RotationAndDispersion_sims(theta, arg2) #In logarithmic space, the multiplication becomes sum.

def lnlike_RotationAndDispersion_sims_TwoDatasets(theta, *args):
	try:
		log_I_Bulge, Re_Bulge, ellipticity_bulge, n, BulgeRotationScale, Max_vel_bulge, CentralBulgeDispersion, alpha_Bulge, log_I_Disc, \
		Re_Disc, ellipticity_disc, DiscRotationScale, Max_vel_disc, CentralDiscDispersion, alpha_Disc, AzimuthVariationParameterBulge, AzimuthVariationParameterDisc, \
		sluggsWeight, atlasWeight = theta
		tmpInputArgs = args[0]
		(X, Y, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed, \
		    X_2, Y_2, Vel_Observed_2, VelErr_Observed_2, VelDisp_Observed_2, VelDispErr_Observed_2) = tmpInputArgs
		PA = 90*np.pi/180
		phi=(PA-np.pi/2.0) # accounting for the different 0 PA convention in astronomy to mathematics

		# print 'test 1'
	
		Data, DataUncertainties, ModelData = MockKinematicsModel(X, Y, Vel_Observed, VelErr_Observed, \
			VelDisp_Observed, VelDispErr_Observed, phi, ellipticity_bulge, ellipticity_disc, log_I_Bulge, Re_Bulge, n, log_I_Disc, Re_Disc, \
			Max_vel_disc, DiscRotationScale, AzimuthVariationParameterDisc, Max_vel_bulge, BulgeRotationScale, AzimuthVariationParameterBulge, \
			CentralBulgeDispersion, CentralDiscDispersion, alpha_Bulge, alpha_Disc)

		Data_2, DataUncertainties_2, ModelData_2 = MockKinematicsModel(X_2, Y_2, Vel_Observed_2, VelErr_Observed_2, \
			VelDisp_Observed_2, VelDispErr_Observed_2, phi, ellipticity_bulge, ellipticity_disc, log_I_Bulge, Re_Bulge, n, log_I_Disc, Re_Disc, \
			Max_vel_disc, DiscRotationScale, AzimuthVariationParameterDisc, Max_vel_bulge, BulgeRotationScale, AzimuthVariationParameterBulge, \
			CentralBulgeDispersion, CentralDiscDispersion, alpha_Bulge, alpha_Disc)
	
		realChi2_sluggs = np.sum(np.log(2*np.pi*DataUncertainties**2/sluggsWeight) + (sluggsWeight*(Data - ModelData)/DataUncertainties)**2.)
		
		realChi2_atlas = np.sum(np.log(2*np.pi*DataUncertainties_2**2/atlasWeight) + (atlasWeight*(Data_2 - ModelData_2)/DataUncertainties_2)**2.)
			
		Chi2Total = realChi2_sluggs + realChi2_atlas # now each of the datasets is appropriately weighted, and it is the likelihoods themselves that are being multiplied
			
	except:
		Chi2Total = np.inf
	return -Chi2Total/2

def lnprob_RotationAndDispersion_sims_TwoDatasets(theta, *args):
	arg1 = args[0]
	arg2 = args[1:]

	lp = lnprior(theta, arg1)
	if not np.isfinite(lp):
	    return -np.inf
	return lp + lnlike_RotationAndDispersion_sims_TwoDatasets(theta, arg2) #In logarithmic space, the multiplication becomes sum.


def KinematicProducer_mainCall(pathName, MagneticumPathName, GalName, ExistingPhotometry = True, TwoDatasets = True, KrigingInput = True, Magneticum = False):
	# instead of sampling a given range, I now sample the same pixels as given for an observed galaxy. 
	# ObservedGalaxyInput_Path = os.path.abspath(pathName)
	import pickle

	if KrigingInput:
		X, Y, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed = krigingFileReadAll(pathName, GalName)
	elif Magneticum:
		pathMajor = MagneticumPathName+GalName+'_vor.dat'
		X, Y, Vel_Observed, VelErr_Observed = np.loadtxt(pathMajor, skiprows = 1, unpack = True, usecols = [0, 1, 2, 3])
		VelDisp_Observed, VelDispErr_Observed = 2 * np.ones(len(X)), 2 * np.ones(len(X)) # There are no uncertainties associated with the maps, at least not currently. 
																						 # Hence currently I am using a low "dummy" value. 
																						 # Need to clarify with Felix at some point whether as estimate has been made
																						 # as to the uncertainties that arise in the projected kinematics from numerical
																						 # resolution origin. 
	else:
		InputDataFilename = pathName+str(GalName)+'/MockKinematics_Input_'+str(GalName)+'_SLUGGS.txt'
		X, Y, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed = np.loadtxt(InputDataFilename, unpack = True, comments = '#')
	
	if TwoDatasets:
		InputDataFilename = pathName+str(GalName)+'/MockKinematics_Input_'+str(GalName)+'_ATLAS3D.txt'
		X_2, Y_2, Vel_Observed_2, VelErr_Observed_2, VelDisp_Observed_2, VelDispErr_Observed_2 = np.loadtxt(InputDataFilename, unpack = True, comments = '#')
	import time
	t0 = time.time() # record the time to output how long the total run took. 


	if ExistingPhotometry:
	
		if TwoDatasets:
			ndim, nwalkers = 15, 1000 # NUMBER OF WALKERS
									  # if both datasets are being used, then two hyperparameters are required, which increases
									  # the number of free parameters. 
		else:
			ndim, nwalkers = 13, 1000 # NUMBER OF WALKERS
		
		
		# setting the upper and lower bounds on the prior ranges of each parameter
		ellipticity_bulge_lower, ellipticity_bulge_upper = 0.0, 0.6 # we want this to be fairly round...
		BulgeRotationScale_lower, BulgeRotationScale_upper = 1, 50
		Max_vel_bulge_lower, Max_vel_bulge_upper = -100, 100 # don't expect the bulge component to be rotating
		CentralBulgeDispersion_lower, CentralBulgeDispersion_upper = 0, 300 # dispersion at R_e/2
		alpha_Bulge_lower, alpha_Bulge_upper = 0, 0.2 # power law slope
		
		
		log_I_Disc_lower, log_I_Disc_upper = np.log10(1e-8), np.log10(1e2)
		Re_Disc_lower, Re_Disc_upper = 0, 100
		DiscRotationScale_lower, DiscRotationScale_upper = 1, 50
		Max_vel_disc_lower, Max_vel_disc_upper = -400, 400
		CentralDiscDispersion_lower, CentralDiscDispersion_upper = 0, 300 # dispersion at R_e/2
		alpha_Disc_lower, alpha_Disc_upper = 0, 0.5 # power law slope
		
		AzimuthVariationParameterBulge_lower, AzimuthVariationParameterBulge_upper = -1.0, 1.0
		AzimuthVariationParameterDisc_lower, AzimuthVariationParameterDisc_upper = -1.0, 1.0
	
		if TwoDatasets:
			# need to define the two hyperparameters for each dataset
			sluggsWeight_lower, sluggsWeight_upper = 0, 10
			atlasWeight_lower, atlasWeight_upper = 0, 10
			sluggsWeightPrior_lower, sluggsWeightPrior_upper = np.exp(-sluggsWeight_upper), np.exp(sluggsWeight_lower)
			atlasWeightPrior_lower, atlasWeightPrior_upper = np.exp(-atlasWeight_upper), np.exp(atlasWeight_lower)
		
		
		# defining the initial position of the walkers
		pos_RotationAndDispersion = []
		for ii in np.arange(nwalkers):  
			ellipticity_bulge_init = np.random.uniform(low=ellipticity_bulge_lower, high=ellipticity_bulge_upper) 
			BulgeRotationScale_init = np.random.uniform(low=BulgeRotationScale_lower, high=BulgeRotationScale_upper) 
			Max_vel_bulge_init = np.random.uniform(low=Max_vel_bulge_lower, high=Max_vel_bulge_upper) 
			CentralBulgeDispersion_init = np.random.uniform(low=CentralBulgeDispersion_lower, high=CentralBulgeDispersion_upper) 
			alpha_Bulge_init = np.random.uniform(low=alpha_Bulge_lower, high=alpha_Bulge_upper) 
		
			log_I_Disc_init = np.random.uniform(low=log_I_Disc_lower, high=log_I_Disc_upper) # sample in log space
			Re_Disc_init = np.random.uniform(low=Re_Disc_lower, high=Re_Disc_upper) 
			DiscRotationScale_init = np.random.uniform(low=DiscRotationScale_lower, high=DiscRotationScale_upper) 
			Max_vel_disc_init = np.random.uniform(low=Max_vel_disc_lower, high=Max_vel_disc_upper) 
			CentralDiscDispersion_init = np.random.uniform(low=CentralDiscDispersion_lower, high=CentralDiscDispersion_upper) 
			alpha_Disc_init = np.random.uniform(low=alpha_Disc_lower, high=alpha_Disc_upper) 
		
			AzimuthVariationParameterBulge_init = np.random.uniform(low=AzimuthVariationParameterBulge_lower, high=AzimuthVariationParameterBulge_upper) 
			AzimuthVariationParameterDisc_init = np.random.uniform(low=AzimuthVariationParameterDisc_lower, high=AzimuthVariationParameterDisc_upper) 
	
			if TwoDatasets:
				sluggsWeightPrior = (np.random.uniform(low=sluggsWeightPrior_lower, high=sluggsWeightPrior_upper))   # uniform sampling in expoential space
				atlasWeightPrior = (np.random.uniform(low=atlasWeightPrior_lower, high=atlasWeightPrior_upper)) # uniform sampling in exponential space  
				sluggsWeight = -np.log(sluggsWeightPrior)
				atlasWeight = -np.log(atlasWeightPrior)
	
				pos_RotationAndDispersion.append([ellipticity_bulge_init, \
					BulgeRotationScale_init, \
					Max_vel_bulge_init, CentralBulgeDispersion_init, alpha_Bulge_init,  \
					log_I_Disc_init, Re_Disc_init, DiscRotationScale_init, Max_vel_disc_init, CentralDiscDispersion_init, \
					alpha_Disc_init, AzimuthVariationParameterBulge_init, AzimuthVariationParameterDisc_init, \
					sluggsWeight, atlasWeight])
		
			else:
				pos_RotationAndDispersion.append([ellipticity_bulge_init, \
					BulgeRotationScale_init, \
					Max_vel_bulge_init, CentralBulgeDispersion_init, alpha_Bulge_init,  \
					log_I_Disc_init, Re_Disc_init, DiscRotationScale_init, Max_vel_disc_init, CentralDiscDispersion_init, \
					alpha_Disc_init, AzimuthVariationParameterBulge_init, AzimuthVariationParameterDisc_init])
		
		# print pos_RotationAndDispersion
		if TwoDatasets:
			boundaries = [ellipticity_bulge_lower, ellipticity_bulge_upper, \
				BulgeRotationScale_lower, BulgeRotationScale_upper, Max_vel_bulge_lower, Max_vel_bulge_upper, \
				CentralBulgeDispersion_lower, CentralBulgeDispersion_upper, alpha_Bulge_lower, alpha_Bulge_upper, \
				log_I_Disc_lower, log_I_Disc_upper, Re_Disc_lower, Re_Disc_upper, DiscRotationScale_lower, DiscRotationScale_upper, \
				Max_vel_disc_lower, Max_vel_disc_upper, CentralDiscDispersion_lower, CentralDiscDispersion_upper, \
				alpha_Disc_lower, alpha_Disc_upper, \
				AzimuthVariationParameterBulge_lower, AzimuthVariationParameterBulge_upper, AzimuthVariationParameterDisc_lower, AzimuthVariationParameterDisc_upper, \
				sluggsWeight_lower, sluggsWeight_upper, atlasWeight_lower, atlasWeight_upper]
		else:
			boundaries = [ellipticity_bulge_lower, ellipticity_bulge_upper, \
				BulgeRotationScale_lower, BulgeRotationScale_upper, Max_vel_bulge_lower, Max_vel_bulge_upper, \
				CentralBulgeDispersion_lower, CentralBulgeDispersion_upper, alpha_Bulge_lower, alpha_Bulge_upper, \
				log_I_Disc_lower, log_I_Disc_upper, Re_Disc_lower, Re_Disc_upper, DiscRotationScale_lower, DiscRotationScale_upper, \
				Max_vel_disc_lower, Max_vel_disc_upper, CentralDiscDispersion_lower, CentralDiscDispersion_upper, \
				alpha_Disc_lower, alpha_Disc_upper, \
				AzimuthVariationParameterBulge_lower, AzimuthVariationParameterBulge_upper, AzimuthVariationParameterDisc_lower, AzimuthVariationParameterDisc_upper]
		
		EffectiveRadius = Reff_Spitzer[GalName]
		ObservedEllipticity = 1 - b_a[GalName]
		n_bulge = SersicIndex_Bulge[GalName]
		Re_Bulge = EffectiveRadius_Bulge[GalName]
		mag_Bulge = MagnitudeRe_Bulge[GalName]
		Spitzer_Radius, Spitzer_Mag, Spitzer_MagErr = SpitzerMagProfileFinder(GalName, 'SpitzerProfiles/') # I haven't yet done a thorough check on the impact of using a Spitzer lumiosity
																					   # profile, but using an optical magnitude for a previous photometric decomposition. 
																					   # My assumption here is that the impact is negligible. 
		
		# Setup MCMC sampler
		import emcee
		if TwoDatasets:
			sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_RotationAndDispersion_hyperparameters,
			                  args=(boundaries, X, Y, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed, \
			                  	X_2, Y_2, Vel_Observed_2, VelErr_Observed_2, VelDisp_Observed_2, VelDispErr_Observed_2, \
			                  	EffectiveRadius, ObservedEllipticity, n_bulge, Re_Bulge, mag_Bulge, Spitzer_Radius, Spitzer_Mag, Spitzer_MagErr),
			                  threads=16) #Threads gives the number of processors to use
		else:
			sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_RotationAndDispersion,
			                  args=(boundaries, X, Y, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed, \
			                  	EffectiveRadius, ObservedEllipticity, n_bulge, Re_Bulge, mag_Bulge, Spitzer_Radius, Spitzer_Mag, Spitzer_MagErr),
			                  threads=16) #Threads gives the number of processors to use
		
		############ implementing a burn-in period ###########
		burnSteps = 1000
		pos, prob, state = sampler.run_mcmc(pos_RotationAndDispersion, burnSteps)
		sampler.reset()
		
		stepNumber = 4000
		outputMCMC = sampler.run_mcmc(pos, stepNumber) # uses the final position of the burn-in period as the starting point. 
		######################################################
		
		
		t = time.time() - t0
		Date=time.strftime('%Y-%m-%d')
		Time = time.asctime().split()[3]
		print '########################################'
		if float(t)/3600 < 1:
			print 'time elapsed:', float(t)/60, 'minutes'
		else:
			print 'time elapsed:', float(t)/3600, 'hours'
		print '########################################'
		print 'Mean acceptance fraction: ', (np.mean(sampler.acceptance_fraction))
		if TwoDatasets:
			OutputFilename = pathName+str(GalName)+'/'+str(GalName)+'_MCMCOutput_TwoDatasets.dat'
		else:
			OutputFilename = pathName+str(GalName)+'/'+str(GalName)+'_MCMCOutput.dat'
		print 'output filename: ', OutputFilename
		
		fileOut = open(OutputFilename, 'wb')
		pickle.dump([sampler.chain, sampler.flatchain, sampler.lnprobability, sampler.flatlnprobability], fileOut)
		fileOut.close()
	
	# here implement an entirely independent version of the code for the purpose of fitting to simulated data. 
	else:
		if TwoDatasets:
			ndim, nwalkers = 19, 2000 # NUMBER OF WALKERS
		else:
			ndim, nwalkers = 17, 2000 # NUMBER OF WALKERS
		
		
		# setting the upper and lower bounds on the prior ranges of each parameter
		log_I_bulge_lower, log_I_bulge_upper = np.log10(1e-3), np.log10(1e7)
		Re_bulge_lower, Re_bulge_upper = 0, 100
		ellipticity_bulge_lower, ellipticity_bulge_upper = 0.0, 0.6 # we want this to be fairly round...
		n_lower, n_upper = 1, 8
		BulgeRotationScale_lower, BulgeRotationScale_upper = 1, 50
		Max_vel_bulge_lower, Max_vel_bulge_upper = -100, 100 # don't expect the bulge component to be rotating
		CentralBulgeDispersion_lower, CentralBulgeDispersion_upper = 0, 300 # dispersion at R_e/2
		alpha_Bulge_lower, alpha_Bulge_upper = 0, 0.2 # power law slope
		
		
		log_I_Disc_lower, log_I_Disc_upper = np.log10(1e-8), np.log10(1e2)
		Re_Disc_lower, Re_Disc_upper = 0, 100
		ellipticity_disc_lower, ellipticity_disc_upper = 0.0, 1.0 
		DiscRotationScale_lower, DiscRotationScale_upper = 1, 50
		Max_vel_disc_lower, Max_vel_disc_upper = -400, 400
		CentralDiscDispersion_lower, CentralDiscDispersion_upper = 0, 300 # dispersion at R_e/2
		alpha_Disc_lower, alpha_Disc_upper = 0, 0.5 # power law slope
		
		AzimuthVariationParameterBulge_lower, AzimuthVariationParameterBulge_upper = -1.0, 1.0
		AzimuthVariationParameterDisc_lower, AzimuthVariationParameterDisc_upper = -1.0, 1.0
	
		if TwoDatasets:
			# need to define the two hyperparameters for each dataset
			sluggsWeight_lower, sluggsWeight_upper = 0, 10
			atlasWeight_lower, atlasWeight_upper = 0, 10
			sluggsWeightPrior_lower, sluggsWeightPrior_upper = np.exp(-sluggsWeight_upper), np.exp(sluggsWeight_lower)
			atlasWeightPrior_lower, atlasWeightPrior_upper = np.exp(-atlasWeight_upper), np.exp(atlasWeight_lower)
		
		
		# defining the initial position of the walkers
		pos_RotationAndDispersion = []
		for ii in np.arange(nwalkers): 
			log_I_bulge_init = np.random.uniform(low=log_I_bulge_lower, high=log_I_bulge_upper)
			Re_bulge_init =  np.random.uniform(low=Re_bulge_lower, high=Re_bulge_upper)
			ellipticity_bulge_init = np.random.uniform(low=ellipticity_bulge_lower, high=ellipticity_bulge_upper) 
			n_init = np.random.uniform(low=n_lower, high=n_upper) 
			BulgeRotationScale_init = np.random.uniform(low=BulgeRotationScale_lower, high=BulgeRotationScale_upper) 
			Max_vel_bulge_init = np.random.uniform(low=Max_vel_bulge_lower, high=Max_vel_bulge_upper) 
			CentralBulgeDispersion_init = np.random.uniform(low=CentralBulgeDispersion_lower, high=CentralBulgeDispersion_upper) 
			alpha_Bulge_init = np.random.uniform(low=alpha_Bulge_lower, high=alpha_Bulge_upper) 
		
			log_I_Disc_init = np.random.uniform(low=log_I_Disc_lower, high=log_I_Disc_upper) # sample in log space
			Re_Disc_init = np.random.uniform(low=Re_Disc_lower, high=Re_Disc_upper) 
			ellipticity_disc_init = np.random.uniform(low=ellipticity_disc_lower, high=ellipticity_disc_upper) 
			DiscRotationScale_init = np.random.uniform(low=DiscRotationScale_lower, high=DiscRotationScale_upper) 
			Max_vel_disc_init = np.random.uniform(low=Max_vel_disc_lower, high=Max_vel_disc_upper) 
			CentralDiscDispersion_init = np.random.uniform(low=CentralDiscDispersion_lower, high=CentralDiscDispersion_upper) 
			alpha_Disc_init = np.random.uniform(low=alpha_Disc_lower, high=alpha_Disc_upper) 
		
			AzimuthVariationParameterBulge_init = np.random.uniform(low=AzimuthVariationParameterBulge_lower, high=AzimuthVariationParameterBulge_upper) 
			AzimuthVariationParameterDisc_init = np.random.uniform(low=AzimuthVariationParameterDisc_lower, high=AzimuthVariationParameterDisc_upper) 
		
			if TwoDatasets:
				sluggsWeightPrior = (np.random.uniform(low=sluggsWeightPrior_lower, high=sluggsWeightPrior_upper))   # uniform sampling in expoential space
				atlasWeightPrior = (np.random.uniform(low=atlasWeightPrior_lower, high=atlasWeightPrior_upper)) # uniform sampling in exponential space  
				sluggsWeight = -np.log(sluggsWeightPrior)
				atlasWeight = -np.log(atlasWeightPrior)
	
				pos_RotationAndDispersion.append([log_I_bulge_init, Re_bulge_init, ellipticity_bulge_init, n_init, \
					BulgeRotationScale_init, \
					Max_vel_bulge_init, CentralBulgeDispersion_init, alpha_Bulge_init,  \
					log_I_Disc_init, Re_Disc_init, ellipticity_disc_init, DiscRotationScale_init, Max_vel_disc_init, CentralDiscDispersion_init, \
					alpha_Disc_init, AzimuthVariationParameterBulge_init, AzimuthVariationParameterDisc_init, \
					sluggsWeight, atlasWeight])
		
			else:
				pos_RotationAndDispersion.append([log_I_bulge_init, Re_bulge_init, ellipticity_bulge_init, n_init, \
					BulgeRotationScale_init, \
					Max_vel_bulge_init, CentralBulgeDispersion_init, alpha_Bulge_init,  \
					log_I_Disc_init, Re_Disc_init, ellipticity_disc_init, DiscRotationScale_init, Max_vel_disc_init, CentralDiscDispersion_init, \
					alpha_Disc_init, AzimuthVariationParameterBulge_init, AzimuthVariationParameterDisc_init])
		
		# print pos_RotationAndDispersion
		if TwoDatasets:
			boundaries = [log_I_bulge_lower, log_I_bulge_upper, Re_bulge_lower, Re_bulge_upper, ellipticity_bulge_lower, ellipticity_bulge_upper, \
				n_lower, n_upper, BulgeRotationScale_lower, BulgeRotationScale_upper, Max_vel_bulge_lower, Max_vel_bulge_upper, \
				CentralBulgeDispersion_lower, CentralBulgeDispersion_upper, alpha_Bulge_lower, alpha_Bulge_upper, \
				log_I_Disc_lower, log_I_Disc_upper, Re_Disc_lower, Re_Disc_upper, ellipticity_disc_lower, ellipticity_disc_upper, \
				DiscRotationScale_lower, DiscRotationScale_upper, \
				Max_vel_disc_lower, Max_vel_disc_upper, CentralDiscDispersion_lower, CentralDiscDispersion_upper, \
				alpha_Disc_lower, alpha_Disc_upper, \
				AzimuthVariationParameterBulge_lower, AzimuthVariationParameterBulge_upper, AzimuthVariationParameterDisc_lower, AzimuthVariationParameterDisc_upper, \
				sluggsWeight_lower, sluggsWeight_upper, atlasWeight_lower, atlasWeight_upper]
		else:
			boundaries = [log_I_bulge_lower, log_I_bulge_upper, Re_bulge_lower, Re_bulge_upper, ellipticity_bulge_lower, ellipticity_bulge_upper, \
				n_lower, n_upper, BulgeRotationScale_lower, BulgeRotationScale_upper, Max_vel_bulge_lower, Max_vel_bulge_upper, \
				CentralBulgeDispersion_lower, CentralBulgeDispersion_upper, alpha_Bulge_lower, alpha_Bulge_upper, \
				log_I_Disc_lower, log_I_Disc_upper, Re_Disc_lower, Re_Disc_upper, ellipticity_disc_lower, ellipticity_disc_upper, \
				DiscRotationScale_lower, DiscRotationScale_upper, \
				Max_vel_disc_lower, Max_vel_disc_upper, CentralDiscDispersion_lower, CentralDiscDispersion_upper, \
				alpha_Disc_lower, alpha_Disc_upper, \
				AzimuthVariationParameterBulge_lower, AzimuthVariationParameterBulge_upper, AzimuthVariationParameterDisc_lower, AzimuthVariationParameterDisc_upper]
		
		# Setup MCMC sampler
		import emcee
		if TwoDatasets:
			sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_RotationAndDispersion_sims_TwoDatasets,
			                  args=(boundaries, X, Y, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed, \
			                  	X_2, Y_2, Vel_Observed_2, VelErr_Observed_2, VelDisp_Observed_2, VelDispErr_Observed_2),
			                  threads=16) #Threads gives the number of processors to use
		else:
			sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_RotationAndDispersion_sims,
			                  args=(boundaries, X, Y, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed),
			                  threads=16) #Threads gives the number of processors to use
		
		############ implementing a burn-in period ###########
		burnSteps = 4000
		pos, prob, state = sampler.run_mcmc(pos_RotationAndDispersion, burnSteps)
		sampler.reset()
		
		stepNumber = 2000
		outputMCMC = sampler.run_mcmc(pos, stepNumber) # uses the final position of the burn-in period as the starting point. 
		######################################################
		
		
		t = time.time() - t0
		Date=time.strftime('%Y-%m-%d')
		Time = time.asctime().split()[3]
		print '########################################'
		if float(t)/3600 < 1:
			print 'time elapsed:', float(t)/60, 'minutes'
		else:
			print 'time elapsed:', float(t)/3600, 'hours'
		print '########################################'
		print 'Mean acceptance fraction: ', (np.mean(sampler.acceptance_fraction))
		if TwoDatasets:
			OutputFilename = pathName+str(GalName)+'/'+str(GalName)+'_MCMCOutput_simsversion_TwoDatasets.dat'
		else:
			OutputFilename = pathName+str(GalName)+'/'+str(GalName)+'_MCMCOutput_simsversion.dat'
		print 'output filename: ', OutputFilename
		
		fileOut = open(OutputFilename, 'wb')
		pickle.dump([sampler.chain, sampler.flatchain, sampler.lnprobability, sampler.flatlnprobability], fileOut)
		fileOut.close()	