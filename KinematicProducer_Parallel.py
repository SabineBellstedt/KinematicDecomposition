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
import matplotlib
matplotlib.use('Agg') # stops the figure from being shown, which is what is needed to run on a supercomputer job. 

lib_path = os.path.abspath('/nfs/cluster/gals/sbellstedt/Library') 
sys.path.append(lib_path)
from galaxyParametersDictionary_v9 import *
from Sabine_Define import *
from KinematicProducer_def import *


GalName = str(sys.argv[1])
if str(sys.argv[2]) == 'False':
	KrigingInput = False # determines whether the SLUGGS dataset is given to MCMC in terms of the raw datapoints, or the kriging pixels. 
else:
	KrigingInput = True
if str(sys.argv[3]) == 'False':
	Photometry = False
else:
	Photometry = True
if str(sys.argv[4]) == 'False':
	Magneticum = False
else:
	Magneticum = True
if str(sys.argv[5]) == 'False':
	TwoDatasets = False 
else:
	TwoDatasets = True
if str(sys.argv[6]) == 'False':
	GlobularClusters = False
else:
	GlobularClusters = True


if str(sys.argv[7]) == 'False':
	BulgeIntensity = False
else:
	BulgeIntensity = True
if str(sys.argv[8]) == 'False':
	BulgeSize = False
else:
	BulgeSize = True
if str(sys.argv[9]) == 'False':
	BulgeEllipticity = False
else:
	BulgeEllipticity = True
if str(sys.argv[10]) == 'False':
	BulgeSersicIndex = False
else:
	BulgeSersicIndex = True
if str(sys.argv[11]) == 'False':
	BulgeRotationScale = False
else:
	BulgeRotationScale = True
if str(sys.argv[12]) == 'False':
	BulgeVelocity = False
else:
	BulgeVelocity = True
if str(sys.argv[13]) == 'False':
	BulgeDispersion = False
else:
	BulgeDispersion = True
if str(sys.argv[14]) == 'False':
	BulgeDispersionDropOff = False
else:
	BulgeDispersionDropOff = True
if str(sys.argv[15]) == 'False':
	DiscIntensity = False
else:
	DiscIntensity = True
if str(sys.argv[16]) == 'False':
	DiscSize = False
else:
	DiscSize = True
if str(sys.argv[17]) == 'False':
	DiscEllipticity = False
else:
	DiscEllipticity = True
if str(sys.argv[18]) == 'False':
	DiscRotationScale = False
else:
	DiscRotationScale = True
if str(sys.argv[19]) == 'False':
	DiscVelocity = False
else:
	DiscVelocity = True
if str(sys.argv[20]) == 'False':
	DiscDispersion = False
else:
	DiscDispersion = True
if str(sys.argv[21]) == 'False':
	DiscDispersionDropOff = False
else:
	DiscDispersionDropOff = True
if str(sys.argv[22]) == 'False':
	BulgeAzimuthVariation = False
else:
	BulgeAzimuthVariation = True
if str(sys.argv[23]) == 'False':
	DiscAzimuthVariation = False
else:
	DiscAzimuthVariation = True
if str(sys.argv[24]) == 'False':
	SluggsWeight = False
else:
	SluggsWeight = True
if str(sys.argv[25]) == 'False':
	AtlasWeight = False
else:
	AtlasWeight = True
if str(sys.argv[26]) == 'False':
	KinematicsWeight = False
else:
	KinematicsWeight = True
if str(sys.argv[27]) == 'False':
	GCWeight = False
else:
	GCWeight = True
if str(sys.argv[28]) == 'False':
	LuminosityWeight = False
else:
	LuminosityWeight = True


pathName = '/nfs/cluster/gals/sbellstedt/Analysis/Mock_Kinematics/'
MagneticumPathName = '/nfs/cluster/gals/sbellstedt/Analysis/Mock_Kinematics/MagneticumGalaxies/'


OutputFilename, ParamSymbol, ParameterNames, Data, PhotometricParameters, PhotometricParameterNames, SpitzerProfile = mainCall_modular(pathName, MagneticumPathName, GalName, \
	Photometry, KrigingInput, Magneticum, TwoDatasets, GlobularClusters, BulgeIntensity, BulgeSize, BulgeEllipticity, BulgeSersicIndex, \
	BulgeRotationScale, BulgeVelocity, BulgeDispersion, BulgeDispersionDropOff, DiscIntensity, DiscSize, DiscEllipticity, \
	DiscRotationScale, DiscVelocity, DiscDispersion, DiscDispersionDropOff, BulgeAzimuthVariation, DiscAzimuthVariation, \
	SluggsWeight, AtlasWeight, KinematicsWeight, GCWeight, LuminosityWeight, \
	nwalkers = 2000, burnSteps = 1000, stepNumber = 1000)

# now to directly analyse the output

from chainconsumer import ChainConsumer

fileIn = open(OutputFilename, 'rb')
chain, flatchain, lnprobability, flatlnprobability, = pickle.load(fileIn) 
fileIn.close()

triangleFilename = OutputFilename.split('MCMCOutput_')[0]+OutputFilename.split('MCMCOutput_')[1].split('.')[0]+'_triangle.pdf'
			
c = ChainConsumer().add_chain(flatchain, parameters=ParamSymbol)
try:
	c.configure(statistics='max_shortest', summary=True)
	fig = c.plotter.plot(figsize = 'PAGE', filename = triangleFilename)
	FinalValues, Lower, Upper = [], [], []
	for parameter in ParamSymbol:
		value, lower, upper = parameterExtractor(c.analysis.get_summary(), parameter)
		FinalValues.append(value)
		Lower.append(lower)
		Upper.append(upper)
except:
	try:
		c.configure(statistics='max', summary=True)
		fig = c.plotter.plot(figsize = 'PAGE', filename = triangleFilename)
		FinalValues, Lower, Upper = [], [], []
		for parameter in ParamSymbol:
			value, lower, upper = parameterExtractor(c.analysis.get_summary(), parameter)
			FinalValues.append(value)
			Lower.append(lower)
			Upper.append(upper)
	except:
		try:
			c.configure(statistics='max_symmetric', summary=True)
			fig = c.plotter.plot(figsize = 'PAGE', filename = triangleFilename)
			FinalValues, Lower, Upper = [], [], []
			for parameter in ParamSymbol:
				value, lower, upper = parameterExtractor(c.analysis.get_summary(), parameter)
				FinalValues.append(value)
				Lower.append(lower)
				Upper.append(upper)
		except:
			c.configure(statistics='max_central', summary=True)
			fig = c.plotter.plot(figsize = 'PAGE', filename = triangleFilename)
			FinalValues, Lower, Upper = [], [], []
			for parameter in ParamSymbol:
				value, lower, upper = parameterExtractor(c.analysis.get_summary(), parameter)
				FinalValues.append(value)
				Lower.append(lower)
				Upper.append(upper)



X = np.zeros(np.array(ParameterNames).size, dtype=[('params', 'U22'), ('values', float), ('lower', float), ('upper', float)])
X['params'] = np.array(ParameterNames)
X['values'] = np.array(FinalValues)
X['lower']  = np.array(Lower)
X['upper']  = np.array(Upper)

np.savetxt(OutputFilename.split('MCMCOutput_')[0]+OutputFilename.split('MCMCOutput_')[1].split('.')[0]+'_parameters.txt', \
	X, fmt="%22s %10.3f %10.3f %10.3f", header = 'Parameter, Value, LowerErr, UpperErr')