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
	return (10**log_I_Bulge) * np.exp(-k * ((Radius_Bulge/Re_Bulge)**(1./n)))

def DiscIntensityFunction(log_I_Disc, Radius_Disc, Re_Disc):
	h = Re_Disc/1.678
	return (10**log_I_Disc) * np.exp(-Radius_Disc/h)

def ComponentRadiusFunction(X, Y, phi, ellipticity_bulge, ellipticity_disc):
	Radius_Bulge = radiusArray(X, Y, phi, ellipticity_bulge)
	Radius_Disc = radiusArray(X, Y, phi, ellipticity_disc)
	return (Radius_Bulge, Radius_Disc)

def AnglesFunction(X, Y):
	Angles = []
	for ii in range(len(X)):
		Angles.append(positionAngle(X[ii], Y[ii], 0, 0))
	Angles = np.array(Angles)
	return Angles


def IntensityPlottingFunction(X, Y, BulgeIntensity, DiscIntensity, TotalIntensity, sizeMapx, sizeMapy, phi, Ellipticity_measured, HalfLightRadius, Linewidth_parameter = 0.8, filename = 'BulgeDiscTest.pdf'):
	fig=plt.figure(figsize=(11, 3))
	ax1=fig.add_subplot(131, aspect = 'equal')
	ax2=fig.add_subplot(132, aspect = 'equal')
	ax3=fig.add_subplot(133, aspect = 'equal')
	ax1.pcolor(X, Y, np.log10(BulgeIntensity), cmap = 'jet', vmin=np.min(np.log10(TotalIntensity)), vmax=np.max(np.log10(TotalIntensity)))
	ax1.contour(X, Y, np.log10(BulgeIntensity), colors='k', linewidths = Linewidth_parameter)
	ax1.set_title('Bulge')
	ax2.pcolor(X, Y, np.log10(DiscIntensity), cmap = 'jet', vmin=np.min(np.log10(TotalIntensity)), vmax=np.max(np.log10(TotalIntensity)))
	ax2.contour(X, Y, np.log10(DiscIntensity), colors='k', linewidths = Linewidth_parameter)
	ax2.set_title('Disc')
	ax3.pcolor(X, Y, np.log10(TotalIntensity), cmap = 'jet', vmin=np.min(np.log10(TotalIntensity)), vmax=np.max(np.log10(TotalIntensity)))
	cs = ax3.contour(X, Y, np.log10(TotalIntensity), colors='k', linewidths = Linewidth_parameter)
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

def RotationPlottingFunction(X, Y, BulgeRotationField, DiscRotationField, TotalRotation, ObservedRotation, MinimumRotation, MaximumRotation, \
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

	fig=plt.figure(figsize=(5, 3))
	ax1=fig.add_subplot(111, aspect = 'equal')
	ax1.pcolor(X, Y, ObservedRotation-TotalRotation, cmap = 'coolwarm', vmin=-50, vmax=50)
	CS1 = ax1.contour(X, Y, ObservedRotation-TotalRotation, colors='k', linewidths = Linewidth_parameter)
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

def DispersionPlottingFunction(X, Y, BulgeDispersion, DiscDispersion, TotalDispersion, ObservedDispersion, MinimumDispersion, MaximumDispersion, \
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

	fig=plt.figure(figsize=(5, 3))
	ax1=fig.add_subplot(111, aspect = 'equal')
	ax1.pcolor(X, Y, ObservedDispersion-TotalDispersion, cmap = 'coolwarm', vmin=-50, vmax=50)
	CS1 = ax1.contour(X, Y, ObservedDispersion-TotalDispersion, colors='k', linewidths = Linewidth_parameter)
	ax1.clabel(CS1, fontsize=7, inline=1)
	ax1.set_title('Dispersion Residual')

	radii = [1, 2, 3, 4]
	ellipses = [Ellipse(xy=[0,0], width=(2.*jj*HalfLightRadius/np.sqrt(AxialRatio)), 
		height=(2.*jj*HalfLightRadius*np.sqrt(AxialRatio)), angle=0, 
		edgecolor = 'white', facecolor = 'none', fill = False, linestyle = 'dashed', linewidth = 2) for jj in radii]
	for ee in ellipses:
		ax1.add_artist(ee)
	plt.subplots_adjust(left = 0.05, right = 0.95, hspace = 0., wspace = 0.01)
	plt.savefig(filename.split('.pdf')[0]+'_Residual.pdf')
	plt.close()

def lnlike_RotationAndDispersion(theta, *args):
	try:
		ellipticity_bulge, log_I_Bulge, BulgeRotationScale, Max_vel_bulge, CentralBulgeDispersion, alpha_Bulge,  \
		log_I_Disc, Re_Disc, DiscRotationScale, Max_vel_disc, CentralDiscDispersion, alpha_Disc = theta
		tmpInputArgs = args[0]
		(X, Y, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed, \
			EffectiveRadius, ObservedEllipticity, n, Re_Bulge) = tmpInputArgs
		PA = 90*np.pi/180
		phi=(PA-np.pi/2.0) # accounting for the different 0 PA convention in astronomy to mathematics

		'''
		in an attempt to reduce the number of free parameters, and also to use the fact that we know the projected 
		ellipticity of the modelled galaxy, here we reduce the info about the ellipticity of one component
		'''

		BulgeIntensity_ellipticityTest = BulgeIntensityFunction(log_I_Bulge, EffectiveRadius, Re_Bulge, n)
		DiscIntensity_ellipticityTest = DiscIntensityFunction(log_I_Disc, EffectiveRadius, Re_Disc)

		BulgeFraction_ellipticityTest = BulgeIntensity_ellipticityTest / (BulgeIntensity_ellipticityTest + DiscIntensity_ellipticityTest)
		DiscFraction_ellipticityTest = DiscIntensity_ellipticityTest / (BulgeIntensity_ellipticityTest + DiscIntensity_ellipticityTest)

		ellipticity_disc = (ObservedEllipticity - BulgeFraction_ellipticityTest*ellipticity_bulge) / DiscFraction_ellipticityTest
		# print 'bulge ellipticity:', ellipticity_bulge, 'disc ellipticity:', ellipticity_disc
		if (ellipticity_disc > 1) | (ellipticity_disc < 0) | (np.isfinite(ellipticity_disc) == False):
			Chi2Total = np.inf
			# print 'run exited'
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
			
			DiscRotation = (Max_vel_disc* Radius_Disc / (DiscRotationScale + Radius_Disc) ) * AngularTerm
			BulgeRotation = (Max_vel_bulge* Radius_Bulge / (BulgeRotationScale + Radius_Bulge) ) * AngularTerm
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


def lnprior_RotationAndDispersion(theta, *args): #p(m, b, f)
    ellipticity_bulge, log_I_Bulge, BulgeRotationScale, Max_vel_bulge, CentralBulgeDispersion, alpha_Bulge,\
    log_I_Disc, Re_Disc, DiscRotationScale, Max_vel_disc, CentralDiscDispersion, alpha_Disc = theta

    tmpInputArgs = args[0] 
    (ellipticity_bulge_lower, ellipticity_bulge_upper, log_I_Bulge_lower, log_I_Bulge_upper, BulgeRotationScale_lower, \
    	BulgeRotationScale_upper, Max_vel_bulge_lower, Max_vel_bulge_upper, CentralBulgeDispersion_lower, CentralBulgeDispersion_upper, \
    	alpha_Bulge_lower, alpha_Bulge_upper,  \
    	# ellipticity_disc_lower, ellipticity_disc_upper, \
    	log_I_Disc_lower, log_I_Disc_upper, Re_Disc_lower, Re_Disc_upper, \
    	DiscRotationScale_lower, DiscRotationScale_upper, Max_vel_disc_lower, Max_vel_disc_upper, \
    	CentralDiscDispersion_lower, CentralDiscDispersion_upper, alpha_Disc_lower, alpha_Disc_upper,  \
    	# gamma_Disc_lower, gamma_Disc_upper\
    	) = tmpInputArgs

    if ((ellipticity_bulge_lower <= ellipticity_bulge <= ellipticity_bulge_upper) and \
    	(log_I_Bulge_lower <= log_I_Bulge <= log_I_Bulge_upper) and (BulgeRotationScale_lower <= BulgeRotationScale <= BulgeRotationScale_upper) and \
    	(Max_vel_bulge_lower <= Max_vel_bulge <= Max_vel_bulge_upper) and \
    	(CentralBulgeDispersion_lower <= CentralBulgeDispersion <= CentralBulgeDispersion_upper) and \
    	(alpha_Bulge_lower <= alpha_Bulge <= alpha_Bulge_upper) and  \
    	# (gamma_Bulge_lower <= gamma_Bulge <= gamma_Bulge_upper) and
    	# (ellipticity_disc_lower <= ellipticity_disc <= ellipticity_disc_upper) and \
    	(log_I_Disc_lower <= log_I_Disc <= log_I_Disc_upper) and \
    	(Re_Disc_lower <= Re_Disc <= Re_Disc_upper) and (DiscRotationScale_lower <= DiscRotationScale <= DiscRotationScale_upper) and \
    	(Max_vel_disc_lower <= Max_vel_disc <= Max_vel_disc_upper) and \
    	(CentralDiscDispersion_lower <= CentralDiscDispersion <= CentralDiscDispersion_upper) and \
    	(alpha_Disc_lower <= alpha_Disc <= alpha_Disc_upper)):
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