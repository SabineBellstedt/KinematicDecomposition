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
	try:
		CS1 = ax1.contour(X, Y, BulgeRotationField, colors='k', linewidths = Linewidth_parameter)
		ax1.clabel(CS1, fontsize=7, inline=1)
	except:
		print 'no bulge rotation contours plotted'
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
	try:
		CS1 = ax1.contour(X, Y, BulgeDispersion, colors='k', linewidths = Linewidth_parameter)
		ax1.clabel(CS1, fontsize=7, inline=1)
	except:
		print 'did not plot bulge dispersion contours'
	ax2.pcolor(X, Y, DiscDispersion, cmap = 'jet', vmin=MinimumDispersion, vmax=MaximumDispersion)
	ax2.set_title('Disc Dispersion')
	try:
		CS2 = ax2.contour(X, Y, DiscDispersion, colors='k', linewidths = Linewidth_parameter)
		ax2.clabel(CS2, fontsize=7, inline=1)
	except:
		print 'did not plot disc dispersion contours'
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