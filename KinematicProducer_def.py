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
	ObservedSel = np.where(np.isfinite(Vel_Observed))

	# needing to combine the velocity and velocity dispersion fits as though they were a single dataset

	ObservedData = np.append(Vel_Observed[ObservedSel], VelDisp_Observed[ObservedSel])
	ObservedUncertainties = np.append(VelErr_Observed[ObservedSel], VelDispErr_Observed[ObservedSel])
	ModelData = np.append(TotalRotation[ObservedSel], TotalDispersion[ObservedSel])

	return(ObservedData, ObservedUncertainties, ModelData)

# def AnglesFunction(X, Y):
# 	Angles=np.arctan(-radian(-X), -radian(-Y))
# 	sel = np.where(Angles < 0)
# 	Angles[sel] = Angles[sel] + 2*np.pi
# 	return (degree(Angles))

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
		Chi2Lum = np.sum( (ModelMagProfile - Spitzer_Mag)**2/(Spitzer_MagErr**2) ) 
		if (ellipticity_disc > 1) | (ellipticity_disc < 0) | (np.isfinite(ellipticity_disc) == False):
			Chi2Total = np.inf
		else:		
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
			PeakIntensity = BulgeIntensity[np.where((X == 0) & (Y == 0))] + DiscIntensity[np.where((X == 0) & (Y == 0))]
			
			
			
			# calculating the chi-squared of the match of the rotation field and dispersion field to the observed galaxy
			ObservedSel = np.where(np.isfinite(Vel_Observed))
			Chi2Rot = np.sum( (Vel_Observed[ObservedSel] - TotalRotation[ObservedSel])**2/(VelErr_Observed[ObservedSel]**2) ) 
			Chi2Disp = np.sum( (VelDisp_Observed[ObservedSel] - TotalDispersion[ObservedSel])**2/(VelDispErr_Observed[ObservedSel])**2 )
			
			Chi2Total = Chi2Rot + Chi2Disp + Chi2Lum
			# print Chi2Rot, Chi2Disp
	except:
		Chi2Total = np.inf
	return -Chi2Total/2


def lnprior_RotationAndDispersion(theta, *args): #p(m, b, f)
    ellipticity_bulge, BulgeRotationScale, Max_vel_bulge, CentralBulgeDispersion, alpha_Bulge,\
    log_I_Disc, Re_Disc, DiscRotationScale, Max_vel_disc, CentralDiscDispersion, alpha_Disc, AzimuthVariationParameterBulge, AzimuthVariationParameterDisc = theta

    tmpInputArgs = args[0] 
    (ellipticity_bulge_lower, ellipticity_bulge_upper, \
    	BulgeRotationScale_lower, BulgeRotationScale_upper, \
    	Max_vel_bulge_lower, Max_vel_bulge_upper, CentralBulgeDispersion_lower, CentralBulgeDispersion_upper, \
    	alpha_Bulge_lower, alpha_Bulge_upper,  \
    	log_I_Disc_lower, log_I_Disc_upper, Re_Disc_lower, Re_Disc_upper, \
    	DiscRotationScale_lower, DiscRotationScale_upper, Max_vel_disc_lower, Max_vel_disc_upper, \
    	CentralDiscDispersion_lower, CentralDiscDispersion_upper, alpha_Disc_lower, alpha_Disc_upper,  \
    	AzimuthVariationParameterBulge_lower, AzimuthVariationParameterBulge_upper, \
    	AzimuthVariationParameterDisc_lower, AzimuthVariationParameterDisc_upper \
    	) = tmpInputArgs

    if ((ellipticity_bulge_lower <= ellipticity_bulge <= ellipticity_bulge_upper) and \
    	(BulgeRotationScale_lower <= BulgeRotationScale <= BulgeRotationScale_upper) and \
    	(Max_vel_bulge_lower <= Max_vel_bulge <= Max_vel_bulge_upper) and \
    	(CentralBulgeDispersion_lower <= CentralBulgeDispersion <= CentralBulgeDispersion_upper) and \
    	(alpha_Bulge_lower <= alpha_Bulge <= alpha_Bulge_upper) and  \
    	(log_I_Disc_lower <= log_I_Disc <= log_I_Disc_upper) and \
    	(Re_Disc_lower <= Re_Disc <= Re_Disc_upper) and (DiscRotationScale_lower <= DiscRotationScale <= DiscRotationScale_upper) and \
    	(Max_vel_disc_lower <= Max_vel_disc <= Max_vel_disc_upper) and \
    	(CentralDiscDispersion_lower <= CentralDiscDispersion <= CentralDiscDispersion_upper) and \
    	(alpha_Disc_lower <= alpha_Disc <= alpha_Disc_upper) and \
    	(AzimuthVariationParameterBulge_lower <= AzimuthVariationParameterBulge <= AzimuthVariationParameterBulge_upper) and \
    	(AzimuthVariationParameterDisc_lower <= AzimuthVariationParameterDisc <= AzimuthVariationParameterDisc_upper)):
        return 0.
    else:
        return -np.inf

def lnprob_RotationAndDispersion(theta, *args):
	arg1 = args[0]
	arg2 = args[1:]

	lp = lnprior_RotationAndDispersion(theta, arg1)
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
		# print len(Spitzer_Radius), Spitzer_MagErr
		
		# Chi2Lum = np.sum( (ModelMagProfile - Spitzer_Mag)**2/(Spitzer_MagErr**2) ) 
		realChi2_lum = np.sum(np.log(2*np.pi*Spitzer_MagErr**2) + ((Spitzer_Mag - ModelMagProfile)/Spitzer_MagErr)**2.)
		if (ellipticity_disc > 1) | (ellipticity_disc < 0) | (np.isfinite(ellipticity_disc) == False):
			Chi2Total = np.inf
		else:		
			ObservedData_sluggs, ObservedUncertainties_sluggs, ModelData_sluggs = MockKinematicsModel(X, Y, Vel_Observed, VelErr_Observed, \
				VelDisp_Observed, VelDispErr_Observed, phi, ellipticity_bulge, ellipticity_disc, log_I_Bulge, Re_Bulge, n, log_I_Disc, Re_Disc, \
				Max_vel_disc, DiscRotationScale, AzimuthVariationParameterDisc, Max_vel_bulge, BulgeRotationScale, AzimuthVariationParameterBulge, \
				CentralBulgeDispersion, CentralDiscDispersion, alpha_Bulge, alpha_Disc)
		
			# Chi2Rot = np.sum( (Vel_Observed[ObservedSel] - TotalRotation[ObservedSel])**2/(VelErr_Observed[ObservedSel]**2) ) 
			# Chi2Disp = np.sum( (VelDisp_Observed[ObservedSel] - TotalDispersion[ObservedSel])**2/(VelDispErr_Observed[ObservedSel])**2 )
		
			realChi2_sluggs = np.sum(np.log(2*np.pi*ObservedUncertainties_sluggs**2/sluggsWeight) + (sluggsWeight*(ObservedData_sluggs - ModelData_sluggs)/ObservedUncertainties_sluggs)**2.)
		
			ObservedData_atlas, ObservedUncertainties_atlas, ModelData_atlas = MockKinematicsModel(X_2, Y_2, Vel_Observed_2, VelErr_Observed_2, \
				VelDisp_Observed_2, VelDispErr_Observed_2, phi, ellipticity_bulge, ellipticity_disc, log_I_Bulge, Re_Bulge, n, log_I_Disc, Re_Disc, \
				Max_vel_disc, DiscRotationScale, AzimuthVariationParameterDisc, Max_vel_bulge, BulgeRotationScale, AzimuthVariationParameterBulge, \
				CentralBulgeDispersion, CentralDiscDispersion, alpha_Bulge, alpha_Disc)
		
			realChi2_atlas = np.sum(np.log(2*np.pi*ObservedUncertainties_atlas**2/atlasWeight) + (atlasWeight*(ObservedData_atlas - ModelData_atlas)/ObservedUncertainties_atlas)**2.)
			
			Chi2Total = realChi2_sluggs + realChi2_atlas + realChi2_lum # now each of the datasets is appropriately weighted, and it is the likelihoods themselves that are being multiplied
			# print 'accepted'
	except:
		# print 'rejected'
		Chi2Total = np.inf
	return -Chi2Total/2


def lnprior_RotationAndDispersion_hyperparameters(theta, *args): #p(m, b, f)
    ellipticity_bulge, BulgeRotationScale, Max_vel_bulge, CentralBulgeDispersion, alpha_Bulge,\
    log_I_Disc, Re_Disc, DiscRotationScale, Max_vel_disc, CentralDiscDispersion, alpha_Disc, \
    AzimuthVariationParameterBulge, AzimuthVariationParameterDisc, sluggsWeight, atlasWeight = theta

    tmpInputArgs = args[0] 
    (ellipticity_bulge_lower, ellipticity_bulge_upper, \
    	BulgeRotationScale_lower, BulgeRotationScale_upper, \
    	Max_vel_bulge_lower, Max_vel_bulge_upper, CentralBulgeDispersion_lower, CentralBulgeDispersion_upper, \
    	alpha_Bulge_lower, alpha_Bulge_upper,  \
    	log_I_Disc_lower, log_I_Disc_upper, Re_Disc_lower, Re_Disc_upper, \
    	DiscRotationScale_lower, DiscRotationScale_upper, Max_vel_disc_lower, Max_vel_disc_upper, \
    	CentralDiscDispersion_lower, CentralDiscDispersion_upper, alpha_Disc_lower, alpha_Disc_upper,  \
    	AzimuthVariationParameterBulge_lower, AzimuthVariationParameterBulge_upper, \
    	AzimuthVariationParameterDisc_lower, AzimuthVariationParameterDisc_upper, \
    	sluggsWeight_lower, sluggsWeight_upper, \
		atlasWeight_lower, atlasWeight_upper) = tmpInputArgs

    if ((ellipticity_bulge_lower <= ellipticity_bulge <= ellipticity_bulge_upper) and \
		(BulgeRotationScale_lower <= BulgeRotationScale <= BulgeRotationScale_upper) and \
		(Max_vel_bulge_lower <= Max_vel_bulge <= Max_vel_bulge_upper) and \
		(CentralBulgeDispersion_lower <= CentralBulgeDispersion <= CentralBulgeDispersion_upper) and \
		(alpha_Bulge_lower <= alpha_Bulge <= alpha_Bulge_upper) and  \
		(log_I_Disc_lower <= log_I_Disc <= log_I_Disc_upper) and \
		(Re_Disc_lower <= Re_Disc <= Re_Disc_upper) and (DiscRotationScale_lower <= DiscRotationScale <= DiscRotationScale_upper) and \
		(Max_vel_disc_lower <= Max_vel_disc <= Max_vel_disc_upper) and \
		(CentralDiscDispersion_lower <= CentralDiscDispersion <= CentralDiscDispersion_upper) and \
		(alpha_Disc_lower <= alpha_Disc <= alpha_Disc_upper) and \
		(AzimuthVariationParameterBulge_lower <= AzimuthVariationParameterBulge <= AzimuthVariationParameterBulge_upper) and \
		(AzimuthVariationParameterDisc_lower <= AzimuthVariationParameterDisc <= AzimuthVariationParameterDisc_upper) and \
		(sluggsWeight_lower <= sluggsWeight <= sluggsWeight_upper) and (atlasWeight_lower <= atlasWeight <= atlasWeight_upper)):
		# print 'all good'
		return 0.
    else:
		# print ellipticity_bulge_lower, ellipticity_bulge, ellipticity_bulge_upper
		# print BulgeRotationScale_lower, BulgeRotationScale, BulgeRotationScale_upper
		# print Max_vel_bulge_lower, Max_vel_bulge, Max_vel_bulge_upper
		# print CentralBulgeDispersion_lower, CentralBulgeDispersion, CentralBulgeDispersion_upper
		# print alpha_Bulge_lower, alpha_Bulge, alpha_Bulge_upper
		# print log_I_Disc_lower, log_I_Disc, log_I_Disc_upper
		# print Re_Disc_lower, Re_Disc, Re_Disc_uppe
		# print DiscRotationScale_lower, DiscRotationScale, DiscRotationScale_upper
		# print Max_vel_disc_lower, Max_vel_disc, Max_vel_disc_upper
		# print CentralDiscDispersion_lower, CentralDiscDispersion, CentralDiscDispersion_upper
		# print alpha_Disc_lower, alpha_Disc, alpha_Disc_upper
		# print AzimuthVariationParameterBulge_lower, AzimuthVariationParameterBulge, AzimuthVariationParameterBulge_upper
		# print AzimuthVariationParameterDisc_lower, AzimuthVariationParameterDisc, AzimuthVariationParameterDisc_upper
		# print sluggsWeight_lower, sluggsWeight, sluggsWeight_upper
		# print atlasWeight_lower, atlasWeight, atlasWeight_upper
		return -np.inf

def lnprob_RotationAndDispersion_hyperparameters(theta, *args):
	arg1 = args[0]
	arg2 = args[1:]

	lp = lnprior_RotationAndDispersion_hyperparameters(theta, arg1)
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
		log_I_Bulge = -mag_Bulge / 2.5

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
		PeakIntensity = BulgeIntensity[np.where((X == 0) & (Y == 0))] + DiscIntensity[np.where((X == 0) & (Y == 0))]
		
		
		
		# calculating the chi-squared of the match of the rotation field and dispersion field to the observed galaxy
		ObservedSel = np.where(np.isfinite(Vel_Observed))
		Chi2Rot = np.sum( (Vel_Observed[ObservedSel] - TotalRotation[ObservedSel])**2/(VelErr_Observed[ObservedSel]**2) ) 
		Chi2Disp = np.sum( (VelDisp_Observed[ObservedSel] - TotalDispersion[ObservedSel])**2/(VelDispErr_Observed[ObservedSel])**2 )
		
		Chi2Total = Chi2Rot + Chi2Disp
		# print Chi2Rot, Chi2Disp
	except:
		Chi2Total = np.inf
	return -Chi2Total/2


def lnprior_RotationAndDispersion_sims(theta, *args): #p(m, b, f)
    log_I_Bulge, Re_Bulge, ellipticity_bulge, n, BulgeRotationScale, Max_vel_bulge, CentralBulgeDispersion, alpha_Bulge,\
    log_I_Disc, Re_Disc, ellipticity_disc, DiscRotationScale, Max_vel_disc, CentralDiscDispersion, alpha_Disc, \
    AzimuthVariationParameterBulge, AzimuthVariationParameterDisc = theta

    tmpInputArgs = args[0] 
    (log_I_Bulge_lower, log_I_Bulge_upper, Re_Bulge_lower, Re_Bulge_upper, ellipticity_bulge_lower, ellipticity_bulge_upper, \
    	n_lower, n_upper, BulgeRotationScale_lower, BulgeRotationScale_upper, \
    	Max_vel_bulge_lower, Max_vel_bulge_upper, CentralBulgeDispersion_lower, CentralBulgeDispersion_upper, \
    	alpha_Bulge_lower, alpha_Bulge_upper,  \
    	log_I_Disc_lower, log_I_Disc_upper, Re_Disc_lower, Re_Disc_upper, ellipticity_disc_lower, ellipticity_disc_upper, \
    	DiscRotationScale_lower, DiscRotationScale_upper, Max_vel_disc_lower, Max_vel_disc_upper, \
    	CentralDiscDispersion_lower, CentralDiscDispersion_upper, alpha_Disc_lower, alpha_Disc_upper,  \
    	AzimuthVariationParameterBulge_lower, AzimuthVariationParameterBulge_upper, \
    	AzimuthVariationParameterDisc_lower, AzimuthVariationParameterDisc_upper \
    	) = tmpInputArgs

    if ((log_I_Bulge_lower <= log_I_Bulge <= log_I_Bulge_upper) and \
    	(Re_Bulge_lower <= Re_Bulge <= Re_Bulge_upper) and \
    	(ellipticity_bulge_lower <= ellipticity_bulge <= ellipticity_bulge_upper) and \
    	(n_lower <= n <= n_upper) and \
    	(BulgeRotationScale_lower <= BulgeRotationScale <= BulgeRotationScale_upper) and \
    	(Max_vel_bulge_lower <= Max_vel_bulge <= Max_vel_bulge_upper) and \
    	(CentralBulgeDispersion_lower <= CentralBulgeDispersion <= CentralBulgeDispersion_upper) and \
    	(alpha_Bulge_lower <= alpha_Bulge <= alpha_Bulge_upper) and  \
    	(log_I_Disc_lower <= log_I_Disc <= log_I_Disc_upper) and \
    	(Re_Disc_lower <= Re_Disc <= Re_Disc_upper) and (ellipticity_disc_lower <= ellipticity_disc <= ellipticity_disc_upper) and \
    	(DiscRotationScale_lower <= DiscRotationScale <= DiscRotationScale_upper) and \
    	(Max_vel_disc_lower <= Max_vel_disc <= Max_vel_disc_upper) and \
    	(CentralDiscDispersion_lower <= CentralDiscDispersion <= CentralDiscDispersion_upper) and \
    	(alpha_Disc_lower <= alpha_Disc <= alpha_Disc_upper) and \
    	(AzimuthVariationParameterBulge_lower <= AzimuthVariationParameterBulge <= AzimuthVariationParameterBulge_upper) and \
    	(AzimuthVariationParameterDisc_lower <= AzimuthVariationParameterDisc <= AzimuthVariationParameterDisc_upper)):
        return 0.
    else:
        return -np.inf

def lnprob_RotationAndDispersion_sims(theta, *args):
	arg1 = args[0]
	arg2 = args[1:]

	lp = lnprior_RotationAndDispersion_sims(theta, arg1)
	if not np.isfinite(lp):
	    return -np.inf
	return lp + lnlike_RotationAndDispersion_sims(theta, arg2) #In logarithmic space, the multiplication becomes sum.