#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 15:35:45 2020

@author: claudriel
"""
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from holopy.scattering import Sphere
from holopy.scattering import calc_cross_sections
from scipy.signal import argrelextrema
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition


def matrix_R(N, pixel_size):   
    """
    Measurements of the matrix of the positions and of the angles
    -----------------------------------------------------
    N: int
        Shape of the hologram
    pixel_size: float
        Value of the pixel size (um)
    -------------------------------------------------------
    R: :class:`.Image` or :class:`.VectorGrid`
       Matrix of positions 
    A: :class:`.Image` or :class:`.VectorGrid`
       Matrix of angles (°) 
    """      
    
    onesvec = np.ones(N)
    inds  = (np.arange(N)+.5 - N/2.) /(N-1.)
    X = np.outer(onesvec, inds)
    Y = np.transpose(X)
    R  = np.sqrt(X**2. + Y**2.) * pixel_size *N
    A = -np.arctan2(X,Y)*180/np.pi+180
    return(R,A)

def media_angolare(R, holo, pixel_size, lim):
    """
    Measurements of the angular average of the hologram and the angular integration
    -----------------------------------------------------
    R: :class:`.Image` or :class:`.VectorGrid`
       Matrix of positions   
    holo: :class:`.Image` or :class:`.VectorGrid`
       Hologram in function of x,y
    pixel_size: float
        Value of the pixel size (um)
    lim: int 
        Center of the hologram
    
    Returns
    -------------------------------------------------------
    Integrale_array: float or list of floats
       Angular integration of the hologram 
    total_aver: float or list of floats
       Angular average of the hologram 
    """      
    
    total_aver = np.array([])
    restrict=np.array([])
    Integrale_array = np.array([])

    freq = R[int(lim)][int(lim)+1]-R[int(lim)][int(lim)]
    x = np.arange(freq, R[int(lim)][0]+freq,freq)
    
    for i in x:
       
        Integrale = np.sum(holo.values[R<=i] *pixel_size**2)#dati nei due cerchi
        Integrale_array = np.append(Integrale_array, Integrale)
        
        restrict = np.sum(holo.values[(R<i) & (R>=i-freq)])/len(holo.values[(R<i) & (R>=i-freq)])
        total_aver = np.append(total_aver, restrict)

    return (Integrale_array, total_aver)


def Cext_FIT(holo, pixel_size, z, fuoco, lim, k, x_fit_1, media, name_graph_3d, name_graph_2d, numero, j):
    
    """
    Measurements of the FIT 2d of the hologram.
    By the fit you can have 
    1) module of S(0)
    2) phase
    3) so by the Optical Theorem you can obtain the Cext
    -----------------------------------------------------
    
    holo: :class:`.Image` or :class:`.VectorGrid`
       Hologram in function of x,y
    pixel_size: float
        Value of the pixel size (um)
    z: float or list of floats
       Distance of propagation 
    fuoco: float
        Position of the focal point (um)
    lim: int 
        Center of the hologram
    k: float
        wavevector
    x_fit_1: float or list of floats
        x-array of the angular average
    media: float or list of floats
        angular average of the hologram 
    integral: str
       Path of the directory where save the data
    numero: int
        parameter of control for each hologram    
    j: str
        str name of the number of object in each image
    
    Returns
    -------------------------------------------------------
    c: float
        Value of the Cext
    err_c: float
        error of the Cext value
    residui: :class:`.Image` or :class:`.VectorGrid`
        residuals of the fit
    params : float
        params of the fit: S(0), phase, sigma (of the exp), A (constant), zeta 
    """      
    
    x_fit = np.arange(len(holo))*pixel_size
    y_fit =np.arange(len(holo))*pixel_size
    x_f, y_f = np.meshgrid(x_fit,y_fit)
    
    def func_hologram_2d(xy_mesh,S, P,sigma,A,zeta):
        (x, y) = xy_mesh
        g = (A+2*(S)/(k*zeta)* np.cos( (k/(2*zeta))*((x-xo)**2 + (y-yo)**2) + P  ))*np.exp(-((x-xo)**2+(y-yo)**2)/(2*sigma**2))
                                            
        return g.ravel()
         
    zeta =z[fuoco]
    S = 13000
    P = np.pi/2
    sigma =60
    xo = lim*pixel_size
    yo = lim*pixel_size
    A = 0.2
                                   
    srot_holo = np.array([])
    for i_img in np.arange(0,len(holo.values)):
        srot_holo =np.append(srot_holo, holo[i_img].values)
    
    try:
        params, params_covariance = optimize.curve_fit(func_hologram_2d,(x_f,y_f), srot_holo , p0=[S, P,sigma, A,zeta])
        data_fitted = func_hologram_2d((x_f,y_f), *params)
        perr = np.sqrt(np.diag(params_covariance))
        err_S = perr[0]
        err_P=perr[1]
                                        
        residui = np.abs(data_fitted.reshape(int(lim*2),int(lim*2))-srot_holo.reshape(int(lim*2),int(lim*2))) 
    
    
        fig, (ax, ax2, cax) = plt.subplots(ncols=3,figsize=(12,6), gridspec_kw={"width_ratios":[1,1, 0.01]})
        fig.subplots_adjust(wspace=0.5)

        im = ax.imshow(srot_holo.reshape(int(lim*2),int(lim*2)), cmap='viridis',alpha = 1, extent=[0,lim*2*pixel_size,0,lim*2*pixel_size],origin='bottom',vmin=-0.1, vmax=0.1)
        ax.set_xlabel("x ($\mu$m)",fontsize=20)
        ax.set_ylabel("y ($\mu$m)",fontsize=20)
        ax.tick_params(axis='both', which='both', labelsize=18)
                        
        im2 = ax2.imshow(data_fitted.reshape(int(lim*2),int(lim*2)),cmap='viridis', extent=[0,lim*2*pixel_size,0,lim*2*pixel_size], origin='bottom',vmin=-0.1, vmax=0.1)
        ax2.set_xlabel("x ($\mu$m)",fontsize=20)
        ax2.set_ylabel("y ($\mu$m)",fontsize=20)
        ax2.tick_params(axis='both', which='both', labelsize=18)
#        ax2.text(90,120, r'S(0) = '+str("{:.0f}".format(params[0])), {'color': 'w', 'fontsize': 18})
#        ax2.text(90,100, r'$\phi$ = '+str("{:.2f}".format(params[1])), {'color': 'w', 'fontsize': 18})
#    
        ip = InsetPosition(ax2, [1.05,0,0.05,1]) 
        cax.set_axes_locator(ip)
    
        fig.colorbar(im, cax=cax, ax=[ax,ax2])
        plt.savefig(name_graph_3d)
        plt.clf()
        plt.close()
    
                                   
        c = 4*np.pi/(k**2)*np.real(params[0])*np.cos(np.pi/2-params[1]) 
        err_c = ((4*np.pi/(k**2)*np.cos(np.pi/2-params[1]))**2*err_S**2+(4*np.pi/(k**2)*np.real(params[0])*np.sin(np.pi/2-params[1]))**2*err_P**2)**0.5
                                        
        plt.plot(x_fit_1, media, '-.',label = 'data')
#                                           plt.plot(holo[int(xo),int(xo):])
        plt.plot(x_fit[0:int(lim+1)],data_fitted.reshape(int(lim*2),int(lim*2))[int(lim),int(lim):], '-.', label = 'fit')
                                        
        plt.plot(x_fit[0:int(lim)], params[3]+np.e**(-(x_fit[0:int(lim)]**2)/(2* params[2]**2))*params[0]*2/(k*params[4]), '-k',alpha = 1,label = 'Gaussian Envelope')
        plt.title('Cext = {:.2f}'.format(c)+ ' +- = {:.2f}'.format(err_c))
        plt.xlabel('x($\mu$m)')
        plt.ylabel('Intensity a.u')
        plt.legend()
        plt.savefig(name_graph_2d)
        plt.clf()
        plt.close()
    
    except RuntimeError:
        print("Error - curve_fit failed")
        c = 0
        err_c = 0
        residui = 0
        params = np.array([0,0])
        
        
    return (c, err_c, residui, params)


def Integration_tw_square(holo, lim, pixel_size):
    """
    Integration of the hologram tw square.
    -----------------------------------------------------
    
    holo: :class:`.Image` or :class:`.VectorGrid`
       Hologram in function of x,y
    lim: int 
        Center of the hologram
    pixel_size: float
        Value of the pixel size (um)
    
    Returns
     -----------------------------------------------------
    Integrale_array: float or list of floats
       Integration of the hologram tw square 
    """      
    Integrale_array  = np.array([])
    for r in np.arange(0,int(lim),1):
        Integrale = np.sum(holo[int(lim-r):int(lim+r),int(lim-r):int(lim+r)] )*pixel_size**2
        Integrale_array = np.append(Integrale_array, Integrale)
    return (Integrale_array)
                                    
def Cext_tw_integration(Integrale_array, raggio, numero_linea, name_graph, dati):
    """
    Plot of the integration of the hologram. 
    By this you can have 
    1) Cext
    2) so by the Optical Theorem you can obtain the real part of S(0)
    -----------------------------------------------------
    
    Integrale_array: float or list of floats
       Integration of the hologram, can be tw circle or square 
    raggio: int 
        Ray of the sphere. It needs for obtain the expected value of cext
    numero:linea: int
        Only for a beautifiul graph (lenght of the exp value of cext line )
    graph name: str
        str name of the graph to save
    
    Returns
     -----------------------------------------------------
    y[0]: float
        Value of the Cext
    """      
    
    medium_index = 1.33
    illum_wavelen = 0.6328
    illum_polarization =(0,1)
    
    Integrale_array = -Integrale_array[:]
    x = np.arange(0,len(Integrale_array),1)
    
    if dati == 'poli':
        distant_sphere = Sphere(r=raggio, n=1.59)
        x_sec = calc_cross_sections(distant_sphere, medium_index, illum_wavelen, illum_polarization)
   # x1=np.arange(0,numero_linea,1)
    
    inviluppo_sup = argrelextrema(Integrale_array[:], np.greater)[0] 
    inviluppo_min = argrelextrema(Integrale_array[:], np.less)[0]
    
    inviluppo_min = inviluppo_min[inviluppo_min>50]
    inviluppo_sup = inviluppo_sup[inviluppo_sup>20]
    
    
    if len(inviluppo_sup)>0 and len(inviluppo_min)>0 :
        if len(inviluppo_sup)>len(inviluppo_min):
            x2 =inviluppo_sup[0:len(inviluppo_min)]
            y = ((Integrale_array[inviluppo_sup[0:len(inviluppo_min)]]-Integrale_array[inviluppo_min])/2+Integrale_array[inviluppo_min])
        if len(inviluppo_sup)<len(inviluppo_min):
            x2 =inviluppo_sup
            y = ((Integrale_array[inviluppo_sup]-Integrale_array[inviluppo_min[0:len(inviluppo_sup)]])/2+Integrale_array[inviluppo_min[0:len(inviluppo_sup)]])
        if len(inviluppo_sup)==len(inviluppo_min):
            x2 =inviluppo_sup
            y = ((Integrale_array[inviluppo_sup]-Integrale_array[inviluppo_min])/2+Integrale_array[inviluppo_min])
    
    
        x1=np.arange(5,inviluppo_sup[0]-10,1)
        x3 = np.arange(5,inviluppo_sup[0]-10,1)
        y2 = np.ones(len(x3))*((Integrale_array[inviluppo_sup[0]]-Integrale_array[inviluppo_min[0]])/2+Integrale_array[inviluppo_min[0]])
                     
        plt.figure(figsize=(12,8))
        plt.plot(x,Integrale_array,'b',linewidth=2,alpha =0.5)
        
        if dati == 'poli':
            plt.plot(x1,np.ones(len(x1))*x_sec[2].values, '<--g', label='Cext Expected')
        
        plt.plot(inviluppo_sup,Integrale_array[inviluppo_sup],'-.k',linewidth=1.5, label ='Envolepe')
        plt.plot(inviluppo_min,Integrale_array[inviluppo_min],'-.k',linewidth=1.5)
        plt.plot(x3,y2,'<--r', label=  'Cext Obtained')
        plt.plot(x2,y,'--r',linewidth=2)
        
        plt.title('Cext = {:.2f}'.format(y[0]))
        plt.xlabel('x (pixel)',fontsize = 20)
        plt.ylabel('Integration ($\mu m^2$)',fontsize = 20) 
        plt.legend(fontsize = 18)
        plt.tick_params(axis='both', which='both', labelsize=18)
        plt.savefig(name_graph)
        plt.clf()
        plt.close()

    else:
        y = np.array([0])
    
    return(y[0])
                                    