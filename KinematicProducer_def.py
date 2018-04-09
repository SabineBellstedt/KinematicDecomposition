# retrieve dictionaries
import os, sys, random
import numpy as np

DropboxDirectory = os.getcwd().split('Dropbox')[0]
lib_path = os.path.abspath(DropboxDirectory+'Dropbox/PhD_Analysis/Library') 
sys.path.append(lib_path)
from galaxyParametersDictionary_v9 import *
from Sabine_Define import *


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
	
	# set up the phyical distribution of points from a bulge component with a sersic profile
	BulgeIntensity = BulgeIntensityFunction(log_I_Bulge, Radius_Bulge, Re_Bulge, n)
	
	# set up the phyical distribution of points from a disc component with an exponential profile
	DiscIntensity = DiscIntensityFunction(log_I_Disc, Radius_Disc, Re_Disc)
	
	# building mock rotational maps. 
	Angles = positionAngle(X, Y, 0, 0)		
	
	DiscRotation = (Max_vel_disc* Radius_Disc / (DiscRotationScale + Radius_Disc) ) * AngularVariationEpsilon(Angles, AzimuthVariationParameterDisc)
	BulgeRotation = (Max_vel_bulge* Radius_Bulge / (BulgeRotationScale + Radius_Bulge) ) * AngularVariationEpsilon(Angles, AzimuthVariationParameterBulge)
	BulgeFraction = BulgeIntensity/(BulgeIntensity+DiscIntensity)
	DiscFraction = DiscIntensity/(BulgeIntensity+DiscIntensity)
	TotalRotation = (BulgeFraction * BulgeRotation + DiscFraction * DiscRotation)
	
	
	# Calculating the velocity dispersions
	# using the power law equation as given by Cappellari et al (2006)
	BulgeDispersion = CentralBulgeDispersion * Radius_Bulge**(-alpha_Bulge)
	DiscDispersion = CentralDiscDispersion * Radius_Disc**(-alpha_Disc)
	
	# check that the dispersion profiles never go below 0
	BulgeDispersion[np.where(BulgeDispersion < 0)] = 0
	DiscDispersion[np.where(DiscDispersion < 0)] = 0
	
	TotalDispersion = BulgeDispersion * BulgeFraction + DiscDispersion * DiscFraction
	
	# calculating the chi-squared of the match of the rotation field and dispersion field to the observed galaxy
	ObservedSel = np.where(np.isfinite(Vel_Observed) & np.isfinite(TotalDispersion)) # avoid the values for which the model gives infinite 
																					 # velocity dispersion values at the centre of the bulge. 

	# needing to combine the velocity and velocity dispersion fits as though they were a single dataset

	ObservedData = np.append(Vel_Observed[ObservedSel], VelDisp_Observed[ObservedSel])
	ObservedUncertainties = np.append(VelErr_Observed[ObservedSel], VelDispErr_Observed[ObservedSel])
	ModelData = np.append(TotalRotation[ObservedSel], TotalDispersion[ObservedSel])

	return(ObservedData, ObservedUncertainties, ModelData)

def MockKinematicsModelGC(X, Y, Vel_Observed, VelErr_Observed, phi, ellipticity_bulge, ellipticity_disc, log_I_Bulge, Re_Bulge, n, log_I_Disc, Re_Disc, \
	Max_vel_disc, DiscRotationScale, AzimuthVariationParameterDisc, Max_vel_bulge, BulgeRotationScale, AzimuthVariationParameterBulge):
	Radius_Bulge, Radius_Disc = ComponentRadiusFunction(X, Y, phi, ellipticity_bulge, ellipticity_disc)
	
	# set up the phyical distribution of points from a bulge component with a sersic profile
	BulgeIntensity = BulgeIntensityFunction(log_I_Bulge, Radius_Bulge, Re_Bulge, n)
	
	# set up the phyical distribution of points from a disc component with an exponential profile
	DiscIntensity = DiscIntensityFunction(log_I_Disc, Radius_Disc, Re_Disc)
	
	# building mock rotational maps. 
	Angles = positionAngle(X, Y, 0, 0)		
	
	DiscRotation = (Max_vel_disc* Radius_Disc / (DiscRotationScale + Radius_Disc) ) * AngularVariationEpsilon(Angles, AzimuthVariationParameterDisc)
	BulgeRotation = (Max_vel_bulge* Radius_Bulge / (BulgeRotationScale + Radius_Bulge) ) * AngularVariationEpsilon(Angles, AzimuthVariationParameterBulge)
	BulgeFraction = BulgeIntensity/(BulgeIntensity+DiscIntensity)
	DiscFraction = DiscIntensity/(BulgeIntensity+DiscIntensity)
	TotalRotation = (BulgeFraction * BulgeRotation + DiscFraction * DiscRotation)
	

	# calculating the chi-squared of the match of the rotation field and dispersion field to the observed galaxy
	ObservedSel = np.where(np.isfinite(Vel_Observed))  # avoid the values for which the model gives infinite 
													   # velocity values at the centre of the bulge. 

	# needing to combine the velocity and velocity dispersion fits as though they were a single dataset

	ObservedData = Vel_Observed[ObservedSel]
	ObservedUncertainties = VelErr_Observed[ObservedSel]
	ModelData = TotalRotation[ObservedSel]

	return(ObservedData, ObservedUncertainties, ModelData)


def MockKinematicsModelAnalysis(X, Y, phi, ellipticity_bulge, ellipticity_disc, log_I_Bulge, Re_Bulge, n, log_I_Disc, Re_Disc, \
	Max_vel_disc, DiscRotationScale, AzimuthVariationParameterDisc, Max_vel_bulge, BulgeRotationScale, AzimuthVariationParameterBulge, \
	CentralBulgeDispersion, CentralDiscDispersion, alpha_Bulge, alpha_Disc, gridSizex,gridSizey): # simplified mock kinematics production for the sake of the 
																			 # analysis code. 
	Radius_Bulge, Radius_Disc = ComponentRadiusFunction(X, Y, phi, ellipticity_bulge, ellipticity_disc)
	
	# set up the phyical distribution of points from a bulge component with a sersic profile	
	BulgeIntensity = BulgeIntensityFunction(log_I_Bulge, Radius_Bulge, Re_Bulge, n)
	
	# set up the phyical distribution of points from a disc component with an exponential profile
	DiscIntensity = DiscIntensityFunction(log_I_Disc, Radius_Disc, Re_Disc)
	
	# building mock rotational maps. 
	# transforming the Angular term
	Angles = positionAngle(X, Y, 0, 0)
	
	DiscRotation = (Max_vel_disc* Radius_Disc / (DiscRotationScale+ Radius_Disc) ) * AngularVariationEpsilon(Angles, AzimuthVariationParameterDisc)
	BulgeRotation = (Max_vel_bulge* Radius_Bulge / (BulgeRotationScale+ Radius_Bulge) ) * AngularVariationEpsilon(Angles, AzimuthVariationParameterBulge)
	BulgeFraction = BulgeIntensity/(BulgeIntensity+DiscIntensity)
	DiscFraction = DiscIntensity/(BulgeIntensity+DiscIntensity)
	TotalRotation = (BulgeFraction * BulgeRotation + DiscFraction * DiscRotation)	
	
	# Calculating the velocity dispersions
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
	
	TotalIntensity = BulgeIntensity + DiscIntensity
	
	BulgeRotationField = BulgeRotation
	DiscRotationField = DiscRotation
	
	MinimumRotation = np.min([np.min(BulgeRotationField), np.min(DiscRotationField), np.min(TotalRotation)])
	MaximumRotation = np.max([np.max(BulgeRotationField), np.max(DiscRotationField), np.max(TotalRotation)])
	
	MinimumDispersion = np.min([np.min(BulgeDispersion), np.min(DiscDispersion), np.min(TotalDispersion)])
	MaximumDispersion = np.max([np.max(BulgeDispersion), np.max(DiscDispersion), np.max(TotalDispersion)])

	return (X, Y, BulgeIntensity, DiscIntensity, DiscRotation, BulgeRotation, TotalRotation, DiscDispersion, BulgeDispersion, TotalDispersion, \
		TotalIntensity, BulgeRotationField, DiscRotationField, MinimumRotation, MaximumRotation, MinimumDispersion, MaximumDispersion)


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

def lnprob(theta, *args):
	boundaries, ParameterNames, Data, PhotometricParameters, PhotometricParameterNames, SpitzerProfile = args

	lp = lnprior(theta, boundaries)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta, ParameterNames, Data, PhotometricParameters, PhotometricParameterNames, SpitzerProfile) #In logarithmic space, the multiplication becomes sum.

def lnlike(theta, Parameters, Data, PhotometricParameters, PhotometricParameterNames, SpitzerProfile):

	Parameters = np.array(Parameters)
	PhotometricParameterNames = np.array(PhotometricParameterNames)

	# now using the actual names of the parameters to extract the right values. 
	if 'BulgeIntensity' in Parameters:
		log_I_Bulge = np.array(theta)[np.where(Parameters == 'BulgeIntensity')] 
	elif 'BulgeMagnitude' in PhotometricParameterNames:
		log_I_Bulge = - np.array(PhotometricParameters)[np.where(PhotometricParameterNames == 'BulgeMagnitude')] / 2.5
	else:
		raise ValueError("Observed bulge magnitude not provided, and bulge intensity has not been set as a free parameter. ")


	if 'BulgeSize' in Parameters:
		Re_Bulge = np.array(theta)[np.where(Parameters == 'BulgeSize')] 
	elif 'BulgeSize' in PhotometricParameterNames:
		Re_Bulge = np.array(PhotometricParameters)[np.where(PhotometricParameterNames == 'BulgeSize')] 
	else:
		raise ValueError("Observed bulge Re not provided, and has not been set as a free parameter. ")


	if 'BulgeSersicIndex' in Parameters:
		n = np.array(theta)[np.where(Parameters == 'BulgeSersicIndex')] 
	elif 'BulgeSersicIndex' in PhotometricParameterNames:
		n = np.array(PhotometricParameters)[np.where(PhotometricParameterNames == 'BulgeSersicIndex')] 
	else:
		raise ValueError("Observed bulge sersic index not provided, and has not been set as a free parameter. ")

	ellipticity_bulge = theta[np.where(Parameters == 'BulgeEllipticity')]

	if 'BulgeVelocity' in Parameters:
		BulgeRotationScale = np.array(theta)[np.where(Parameters == 'BulgeRotationScale')]
		Max_vel_bulge = np.array(theta)[np.where(Parameters == 'BulgeVelocity')]
	elif 'BulgeRotationScale' in Parameters:
		raise ValueError("bulge rotation scale given, but not bulge rotational velocity. Check which values are included. ")
	else:
		BulgeRotationScale = 0
		Max_vel_bulge = 0
	Re_Bulge = np.array(theta)[np.where(Parameters == 'BulgeSize')]
	CentralBulgeDispersion = np.array(theta)[np.where(Parameters == 'BulgeDispersion')]
	alpha_Bulge = np.array(theta)[np.where(Parameters == 'BulgeDispersionDropOff')]
	log_I_Disc = np.array(theta)[np.where(Parameters == 'DiscIntensity')]
	Re_Disc = np.array(theta)[np.where(Parameters == 'DiscSize')]
	DiscRotationScale = np.array(theta)[np.where(Parameters == 'DiscRotationScale')]
	Max_vel_disc = np.array(theta)[np.where(Parameters == 'DiscVelocity')]

	if 'DiscDispersion' in Parameters:
		CentralDiscDispersion = np.array(theta)[np.where(Parameters == 'DiscDispersion')]
		alpha_Disc = np.array(theta)[np.where(Parameters == 'DiscDispersionDropOff')]
	elif 'DiscDispersionDropOff' in Parameters:
		raise ValueError("Disc dispersion drop off value given, but not central disc dispersion. Check which values are included. ")
	else:
		# in this scenario, the disc dispersion has not been determined. 
		# This means that we assume the full dispersion contribution comes from the bulge component of the galaxy. 
		CentralDiscDispersion = 0
		alpha_Disc = 0

	AzimuthVariationParameterBulge = np.array(theta)[np.where(Parameters == 'BulgeAzimuthVariation')]
	AzimuthVariationParameterDisc = np.array(theta)[np.where(Parameters == 'DiscAzimuthVariation')]



	# now to check whether the user provided a disc ellipticity
	if 'DiscEllipticity' in Parameters:
		ellipticity_disc = np.array(theta)[np.where(Parameters == 'DiscEllipticity')]
	elif 'ObservedEllipticity' in PhotometricParameterNames:
		# in this case, we want to calculate what the disc ellipticity should be, given the bulge ellipticity, 
		# to match the observed ellipticity. 
		ObservedEllipticity = np.array(PhotometricParameters)[np.where(PhotometricParameterNames == 'ObservedEllipticity')]
		EffectiveRadius = np.array(PhotometricParameters)[np.where(PhotometricParameterNames == 'ObservedEllipticity')]

		BulgeIntensity_ellipticityTest = BulgeIntensityFunction(log_I_Bulge, EffectiveRadius, Re_Bulge, n)
		DiscIntensity_ellipticityTest = DiscIntensityFunction(log_I_Disc, EffectiveRadius, Re_Disc)
		
		BulgeFraction_ellipticityTest = BulgeIntensity_ellipticityTest / (BulgeIntensity_ellipticityTest + DiscIntensity_ellipticityTest)
		DiscFraction_ellipticityTest = DiscIntensity_ellipticityTest / (BulgeIntensity_ellipticityTest + DiscIntensity_ellipticityTest)
		
		ellipticity_disc = (ObservedEllipticity - BulgeFraction_ellipticityTest*ellipticity_bulge) / DiscFraction_ellipticityTest

		if (ellipticity_disc > 1) | (ellipticity_disc < 0) | (np.isfinite(ellipticity_disc) == False):
			return -np.inf # this setup is unphysical, therefore return a 0 likelihood. 
	else:
		raise ValueError("Observed ellipticity not provided, and disc ellipticity has not been set as a free parameter. ")


	if len(SpitzerProfile) > 0:
		if not 'LuminosityWeight' in Parameters:
			raise ValueError("Luminosity hyperparameter MUST be selected if the data is being contrained to a luminosity profile.")
		else:
			LuminosityWeight = np.array(theta)[np.where(Parameters == 'LuminosityWeight')]
			# the Spitzer luminosity profile has been provided, and hence we have an extra constraint of the model. 
			# print PhotometricParameters
			Spitzer_Radius = np.array(SpitzerProfile[0])
			Spitzer_Mag = np.array(SpitzerProfile[1])
			Spitzer_MagErr = np.array(SpitzerProfile[2])
		
			ModelMagProfile = -2.5*np.log10(BulgeIntensityFunction(log_I_Bulge, Spitzer_Radius, Re_Bulge, n) + \
				DiscIntensityFunction(log_I_Disc, Spitzer_Radius, Re_Disc))
			realChi2_lum = np.sum(np.log(2*np.pi*Spitzer_MagErr**2/LuminosityWeight) + (LuminosityWeight*(Spitzer_Mag - ModelMagProfile)/Spitzer_MagErr)**2.) 
	
			photometricContraint = True

	else:
		photometricContraint = False  



	# identifying whether or not to implement the hyperparameter regime. 
	if len(Data) == 6:
		if not 'KinematicsWeight' in Parameters:
			raise ValueError("Kinematics hyperparameter MUST be selected if only a single dataset is used.")
		else:
			KinematicsWeight = np.array(theta)[np.where(Parameters == 'KinematicsWeight')]
			X, Y, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed = Data
			DatasetNumber = 1
			GCKinematics = False
	elif len(Data) == 10: # here, only one dataset is being used internally, and the GC velocities are being used
		if not 'GCWeight' in Parameters:
			raise ValueError("GC hyperparameter MUST be selected when fitting to the GC dataset.")
		if not 'KinematicsWeight' in Parameters:
			raise ValueError("Kinematics hyperparameter MUST be selected if only a single dataset is used.")
		else:
			GCWeight = np.array(theta)[np.where(Parameters == 'GCWeight')]
			KinematicsWeight = np.array(theta)[np.where(Parameters == 'KinematicsWeight')]
			X, Y, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed, \
				X_GC, Y_GC, Vel_GC, VelErr_GC = Data
			DatasetNumber = 1 # still only using one central kinematic dataset
			GCKinematics = True
	elif len(Data) == 12: # at this point the user is fitting to two datasets and no GCs, and hence we require the use of hyperparameters
		if not 'SluggsWeight' in Parameters:
			raise ValueError("Hyperparameters MUST be selected when fitting to two datasets.")
		else:
			sluggsWeight = np.array(theta)[np.where(Parameters == 'SluggsWeight')]
			atlasWeight = np.array(theta)[np.where(Parameters == 'AtlasWeight')]
			X, Y, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed, \
			X_2, Y_2, Vel_Observed_2, VelErr_Observed_2, VelDisp_Observed_2, VelDispErr_Observed_2 = Data
			DatasetNumber = 2
			GCKinematics = False
	elif len(Data) == 16: # will now be using two central datasets, plus the GC dataset
		if not 'SluggsWeight' in Parameters:
			raise ValueError("Hyperparameters MUST be selected when fitting to two datasets.")
		if not 'GCWeight' in Parameters:
			raise ValueError("GC hyperparameter MUST be selected when fitting to the GC dataset.")
		else:
			GCWeight = np.array(theta)[np.where(Parameters == 'GCWeight')]
			sluggsWeight = np.array(theta)[np.where(Parameters == 'SluggsWeight')]
			atlasWeight = np.array(theta)[np.where(Parameters == 'AtlasWeight')]
			X, Y, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed, \
				X_2, Y_2, Vel_Observed_2, VelErr_Observed_2, VelDisp_Observed_2, VelDispErr_Observed_2, \
				X_GC, Y_GC, Vel_GC, VelErr_GC = Data
			DatasetNumber = 2
			GCKinematics = True
	else:
		raise ValueError("Unexpected input dataset configuration. \n For one dataset, 6 arrays expected, \
			and for two datasets, 12. If GC radial velocities are desired as an additional constraint, \
			then an additional 4 arrays are expected. {0} provided in total.".format(len(Data)))


	PA = 90*np.pi/180
	phi=(PA-np.pi/2.0) # accounting for the different 0 PA convention in astronomy to mathematics
			


	if DatasetNumber == 1:

		ObservedData, ObservedUncertainties, ModelData = MockKinematicsModel(X, Y, Vel_Observed, VelErr_Observed, \
			VelDisp_Observed, VelDispErr_Observed, phi, ellipticity_bulge, ellipticity_disc, log_I_Bulge, Re_Bulge, n, log_I_Disc, Re_Disc, \
			Max_vel_disc, DiscRotationScale, AzimuthVariationParameterDisc, Max_vel_bulge, BulgeRotationScale, AzimuthVariationParameterBulge, \
			CentralBulgeDispersion, CentralDiscDispersion, alpha_Bulge, alpha_Disc)
		
		realChi2_kin = np.sum(np.log(2*np.pi*ObservedUncertainties**2/KinematicsWeight) + (KinematicsWeight*(ObservedData - ModelData)/ObservedUncertainties)**2.)

	elif DatasetNumber == 2:
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
			
		realChi2_kin = 	realChi2_sluggs + realChi2_atlas

	# depending on whether the GC radial velocity dataset is included, the final realChi2_kin value will change:
	if GCKinematics:
		ObservedData_GC, ObservedUncertainties_GC, ModelData_GC = MockKinematicsModelGC(X_GC, Y_GC, Vel_GC, VelErr_GC, phi, ellipticity_bulge, ellipticity_disc, \
			log_I_Bulge, Re_Bulge, n, log_I_Disc, Re_Disc, Max_vel_disc, DiscRotationScale, AzimuthVariationParameterDisc, Max_vel_bulge, BulgeRotationScale, \
			AzimuthVariationParameterBulge)

		realChi2_GC = np.sum(np.log(2*np.pi*ObservedUncertainties_GC**2/GCWeight) + (GCWeight*(ObservedData_GC - ModelData_GC)/ObservedUncertainties_GC)**2.)

		realChi2_kin += realChi2_GC # adding the contribution of the GC kinematics

	if photometricContraint:	
		Chi2Total = realChi2_kin + realChi2_lum # now each of the datasets is appropriately weighted, and it is the likelihoods themselves that are being multiplied		
	else:
		Chi2Total = realChi2_kin

	return -Chi2Total/2

def InitialWalkerPosition(WalkerNumber, LowerBounds, UpperBounds, PriorType):
	# wiriting up a code that, for an arbitrary number of parameters and prior bounds, will set up the initial walker positions. 

	InitialPosition = []
	for ii in np.arange(WalkerNumber): 
		WalkerPosition = [] 
		for jj in range(len(LowerBounds)):
			if PriorType[jj] == 'uniform':
				WalkerPosition.append(np.random.uniform(low=LowerBounds[jj], high=UpperBounds[jj]) )
			elif PriorType[jj] == 'exponential':
				ExpInitPosition = np.random.uniform(low=np.exp(-UpperBounds[jj]), high=np.exp(LowerBounds[jj])) 
				WalkerPosition.append(-np.log(ExpInitPosition))
	
		InitialPosition.append(WalkerPosition)

	Boundaries = []
	for ii in range(len(LowerBounds)):
		Boundaries.append(LowerBounds[ii])
		Boundaries.append(UpperBounds[ii])

	return InitialPosition, Boundaries


def mainCall_modular(pathName, MagneticumPathName, GalName, \
	Photometry = False, KrigingInput = True, Magneticum = False, TwoDatasets = True, GlobularClusters = False, \
	BulgeIntensity = True, BulgeSize = True, BulgeEllipticity = True, BulgeSersicIndex = True, \
	BulgeRotationScale = True, BulgeVelocity = True, BulgeDispersion = True, BulgeDispersionDropOff = True, \
	DiscIntensity = True, DiscSize = True, DiscEllipticity = True, DiscRotationScale = True, DiscVelocity = True, \
	DiscDispersion = True, DiscDispersionDropOff = True, BulgeAzimuthVariation = True, DiscAzimuthVariation = True, \
	SluggsWeight = True, AtlasWeight = True, KinematicsWeight = True, GCWeight = False, LuminosityWeight = False, \
	nwalkers = 2000, burnSteps = 1000, stepNumber = 4000):

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
		Data = [X, Y, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed, X_2, Y_2, Vel_Observed_2, VelErr_Observed_2, VelDisp_Observed_2, VelDispErr_Observed_2]
	else:
		Data = [X, Y, Vel_Observed, VelErr_Observed, VelDisp_Observed, VelDispErr_Observed]

	if GlobularClusters:
		X_GC, Y_GC, Vel_GC, VelErr_GC = np.loadtxt(pathName+str(GalName)+'/GC_RVs_'+str(GalName)+'.txt')
		Data.append(X_GC)
		Data.append(Y_GC)
		Data.append(Vel_GC)
		Data.append(VelErr_GC)


	import time
	t0 = time.time() # record the time to output how long the total run took. 

	ndim = 0
	LowerBounds, UpperBounds, PriorType, ParameterNames, ParamSymbol = [], [], [], [], []
	# selecting all possible free parameters, with their prior bounds:
	if BulgeIntensity: # log bulge I
		ndim += 1
		LowerBounds.append(-3)
		UpperBounds.append(7)
		PriorType.append('uniform')
		ParameterNames.append('BulgeIntensity')
		ParamSymbol.append(r"$\log(I_b)$")
	if BulgeSize:
		ndim += 1
		LowerBounds.append(0)
		UpperBounds.append(100)
		PriorType.append('uniform')
		ParameterNames.append('BulgeSize')
		ParamSymbol.append(r"$R_{e,b}$")
	if BulgeEllipticity:
		ndim += 1
		LowerBounds.append(0.0) # we want this to be fairly round...
		UpperBounds.append(0.6)
		PriorType.append('uniform')
		ParameterNames.append('BulgeEllipticity')
		ParamSymbol.append(r"$\epsilon_b$")
	if BulgeSersicIndex:
		ndim += 1
		LowerBounds.append(1)
		UpperBounds.append(8)
		PriorType.append('uniform')
		ParameterNames.append('BulgeSersicIndex')
		ParamSymbol.append(r'n')
	if BulgeRotationScale:
		ndim += 1
		LowerBounds.append(1)
		UpperBounds.append(50)
		PriorType.append('uniform')
		ParameterNames.append('BulgeRotationScale')
		ParamSymbol.append(r"$s_b$")
	if BulgeVelocity:
		ndim += 1
		LowerBounds.append(-100)# don't expect the bulge component to be rotating
		UpperBounds.append(100)
		PriorType.append('uniform')
		ParameterNames.append('BulgeVelocity')
		ParamSymbol.append(r"$v_b$")
	if BulgeDispersion:
		ndim += 1
		LowerBounds.append(0)
		UpperBounds.append(300)
		PriorType.append('uniform')
		ParameterNames.append('BulgeDispersion')
		ParamSymbol.append(r"$\sigma_{c,b}$")
	if BulgeDispersionDropOff:
		ndim += 1
		LowerBounds.append(0)
		UpperBounds.append(0.2)
		PriorType.append('uniform')
		ParameterNames.append('BulgeDispersionDropOff')
		ParamSymbol.append(r"$\alpha_b$")
		
	if DiscIntensity: # log disc I
		ndim += 1
		LowerBounds.append(-8)
		UpperBounds.append(2)
		PriorType.append('uniform')
		ParameterNames.append('DiscIntensity')
		ParamSymbol.append(r"$\log(I_d)$")
	if DiscSize:
		ndim += 1
		LowerBounds.append(0)
		UpperBounds.append(100)
		PriorType.append('uniform')
		ParameterNames.append('DiscSize')
		ParamSymbol.append(r"$R_{e,d}$")
	if DiscEllipticity:
		ndim += 1		 	 			# the only reason why this would be round, is because a disc is fairly inclined. 
		LowerBounds.append(0.5)  		# it might therefore be worth thinking about whether this value and the maximum
		UpperBounds.append(0.95)  		# rotational velocity should be linked, such that if the disc is very round, then 
		PriorType.append('uniform') 	# one won't see it rotating...
		ParameterNames.append('DiscEllipticity')
		ParamSymbol.append(r"$\epsilon_d$")
	if DiscRotationScale:
		ndim += 1
		LowerBounds.append(1)
		UpperBounds.append(50)
		PriorType.append('uniform')
		ParameterNames.append('DiscRotationScale')
		ParamSymbol.append(r"$s_d$")
	if DiscVelocity:
		ndim += 1
		LowerBounds.append(-400)
		UpperBounds.append(400)
		PriorType.append('uniform')
		ParameterNames.append('DiscVelocity')
		ParamSymbol.append(r"$v_d$")
	if DiscDispersion:
		ndim += 1
		LowerBounds.append(0)
		UpperBounds.append(300)
		PriorType.append('uniform')
		ParameterNames.append('DiscDispersion')
		ParamSymbol.append(r"$\sigma_{c,d}$")
	if DiscDispersionDropOff:
		ndim += 1
		LowerBounds.append(0)
		UpperBounds.append(0.5)
		PriorType.append('uniform')
		ParameterNames.append('DiscDispersionDropOff')
		ParamSymbol.append(r"$\alpha_d$")
	if BulgeAzimuthVariation:
		ndim += 1
		LowerBounds.append(-1.0)
		UpperBounds.append(1.0)
		PriorType.append('uniform')
		ParameterNames.append('BulgeAzimuthVariation')
		ParamSymbol.append(r"$\theta_{\rm b}$")
	if DiscAzimuthVariation:
		ndim += 1
		LowerBounds.append(-1.0)
		UpperBounds.append(1.0)
		PriorType.append('uniform')
		ParameterNames.append('DiscAzimuthVariation')
		ParamSymbol.append(r"$\theta_{\rm d}$")
	
	if SluggsWeight:
		ndim += 1
		LowerBounds.append(0)
		UpperBounds.append(10)
		PriorType.append('exponential')
		ParameterNames.append('SluggsWeight')
		ParamSymbol.append(r"$\omega_{\rm SLUGGS}$")
	if AtlasWeight:
		ndim += 1
		LowerBounds.append(0)
		UpperBounds.append(10)
		PriorType.append('exponential')
		ParameterNames.append('AtlasWeight')
		ParamSymbol.append(r"$\omega_{\rm ATLAS}$")
	if KinematicsWeight:
		ndim += 1
		LowerBounds.append(0)
		UpperBounds.append(10)
		PriorType.append('exponential')
		ParameterNames.append('KinematicsWeight')
		ParamSymbol.append(r"$\omega_{\rm kin}$")
	if GCWeight:
		ndim += 1
		LowerBounds.append(0)
		UpperBounds.append(10)
		PriorType.append('exponential')
		ParameterNames.append('GCWeight')
		ParamSymbol.append(r"$\omega_{\rm GC}$")
	if LuminosityWeight:
		ndim += 1
		LowerBounds.append(0)
		UpperBounds.append(10)
		PriorType.append('exponential')
		ParameterNames.append('LuminosityWeight')
		ParamSymbol.append(r"$\omega_{\rm lum}$")
	# making a selection of which free parameters are required, in order to select the walker initial positions and define prior boundaries
	pos, boundaries = InitialWalkerPosition(nwalkers, LowerBounds, UpperBounds, PriorType)


	if Photometry:

		EffectiveRadius = Reff_Spitzer[GalName]
		ObservedEllipticity = 1 - b_a[GalName]
		n_bulge = SersicIndex_Bulge[GalName]
		Re_Bulge = EffectiveRadius_Bulge[GalName]
		mag_Bulge = MagnitudeRe_Bulge[GalName]
		Spitzer_Radius, Spitzer_Mag, Spitzer_MagErr = SpitzerMagProfileFinder(GalName, 'SpitzerProfiles/') # I haven't yet done a thorough check on the impact of using a Spitzer lumiosity
																						   # profile, but using an optical magnitude for a previous photometric decomposition. 
																						   # My assumption here is that the impact is negligible. 
		PhotometricParameters = [EffectiveRadius, ObservedEllipticity, n_bulge, Re_Bulge, mag_Bulge]
	
		PhotometricParameterNames = ['EffectiveRadius', 'ObservedEllipticity', 'BulgeSersicIndex', 'BulgeSize', 'BulgeMagnitude']

		SpitzerProfile = [Spitzer_Radius, Spitzer_Mag, Spitzer_MagErr]

	else:
		# just include empty arrays for each of the photometric arrays. 
		# empty arrays will trigger the relevant sections in the lnprob functions to indicate that photometry is not required. 
		PhotometricParameters = []
		PhotometricParameterNames = []
		SpitzerProfile = []

	import emcee
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
		  args=(boundaries, ParameterNames, Data, PhotometricParameters, PhotometricParameterNames, SpitzerProfile), 
		  threads=16) #Threads gives the number of processors to use

	############ implementing a burn-in period ###########
	pos_postBurn, prob, state = sampler.run_mcmc(pos, burnSteps)
	sampler.reset()
	
	outputMCMC = sampler.run_mcmc(pos_postBurn, stepNumber) # uses the final position of the burn-in period as the starting point. 
	######################################################


	t = time.time() - t0
	print '########################################'
	if float(t)/3600 < 1:
		print 'time elapsed:', float(t)/60, 'minutes'
	else:
		print 'time elapsed:', float(t)/3600, 'hours'
	print '########################################'
	print 'Mean acceptance fraction: ', (np.mean(sampler.acceptance_fraction))
	suffix = ''
	if TwoDatasets:
		suffix = suffix + 'TwoDatasets_'
	if Photometry:
		suffix = suffix + 'Photometry_'
	if Magneticum:
		suffix = suffix + 'Magneticum_'
	if GlobularClusters:
		suffix = suffix + 'GC_'
	suffix = suffix + 'FreeParam-'+str(ndim)

	if not os.path.exists(pathName+str(GalName)+'/'+suffix): 
		os.mkdir(pathName+str(GalName)+'/'+suffix)

	OutputFilename = pathName+str(GalName)+'/'+suffix+'/'+str(GalName)+'_MCMCOutput_'+suffix+'.dat'
	print 'output filename: ', OutputFilename
	
	import pickle
	fileOut = open(OutputFilename, 'wb')
	pickle.dump([sampler.chain, sampler.flatchain, sampler.lnprobability, sampler.flatlnprobability], fileOut)
	fileOut.close()	

	return OutputFilename, ParamSymbol, ParameterNames, Data, PhotometricParameters, PhotometricParameterNames, SpitzerProfile




