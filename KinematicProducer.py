'''
This code is designed to produce mock kinematics based on a number of bulge and disc parameters. 

'''

# retrieve dictionaries
import os, sys, random, pickle
import numpy as np

DropboxDirectory = os.getcwd().split('Dropbox')[0]
lib_path = os.path.abspath(DropboxDirectory+'Dropbox/PhD_Analysis/Library') 
sys.path.append(lib_path)
from galaxyParametersDictionary_v9 import *
from Sabine_Define import *
from KinematicProducer_def import *


GalName = 'NGC1023'
KrigingInput = True # determines whether the SLUGGS dataset is given to MCMC in terms of the raw datapoints, or the kriging pixels. 
Photometry = True
TwoDatasets = False # two weights must be selected when selecting two datasets. 
Magneticum = False

BulgeIntensity = True
BulgeSize = True
BulgeEllipticity = True
BulgeSersicIndex = True
BulgeRotationScale = True
BulgeVelocity = True
BulgeDispersion = True
BulgeDispersionDropOff = True
DiscIntensity = True
DiscSize = True
DiscEllipticity = True
DiscRotationScale = True
DiscVelocity = True
DiscDispersion = False
DiscDispersionDropOff = False
BulgeAzimuthVariation = True
DiscAzimuthVariation = True
SluggsWeight = False
AtlasWeight = False

if not os.path.exists(str(GalName)): 
	os.mkdir(str(GalName))

pathName = DropboxDirectory+'Dropbox/PhD_Analysis/Analysis/Angular Momentum/Mock_Kinematics/'
MagneticumPathName = DropboxDirectory+'Dropbox/PhD_Analysis/data/MagneticumGalaxies/'
# KinematicProducer_mainCall(pathName, MagneticumPathName, GalName, ExistingPhotometry = True, TwoDatasets = True, KrigingInput = True, Magneticum = False)

OutputFilename, ParamSymbol, ParameterNames, Data, PhotometricParameters, PhotometricParameterNames, SpitzerProfile = mainCall_modular(pathName, MagneticumPathName, GalName, \
	Photometry, KrigingInput, Magneticum, TwoDatasets, BulgeIntensity, BulgeSize, BulgeEllipticity, BulgeSersicIndex, \
	BulgeRotationScale, BulgeVelocity, BulgeDispersion, BulgeDispersionDropOff, DiscIntensity, DiscSize, DiscEllipticity, \
	DiscRotationScale, DiscVelocity, DiscDispersion, DiscDispersionDropOff, BulgeAzimuthVariation, DiscAzimuthVariation, \
	SluggsWeight, AtlasWeight, \
	nwalkers = 800, burnSteps = 500, stepNumber = 100)

# now to directly analyse the output

from chainconsumer import ChainConsumer

fileIn = open(OutputFilename, 'rb')
chain, flatchain, lnprobability, flatlnprobability, = pickle.load(fileIn) 
fileIn.close()

triangleFilename = OutputFilename.split('MCMCOutput_')[0]+OutputFilename.split('MCMCOutput_')[1].split('.')[0]+'_triangle.pdf'
			
c = ChainConsumer().add_chain(flatchain, parameters=ParamSymbol)
c.configure(statistics='max_shortest', summary=True)
fig = c.plotter.plot(figsize = 'PAGE', filename = triangleFilename)

def parameterExtractor(inputDict, name):
	value = inputDict[name][1]
	lower = inputDict[name][1] - inputDict[name][0]
	upper = inputDict[name][2] - inputDict[name][1]
	# print 'Parameter extraction successful for', name
	return value, lower, upper

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


