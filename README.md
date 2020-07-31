# Description

Holohram package - functions to analyse holograms

HoloSoft: Holography in Python
=========================================

HoloSoft is a python based tool for study mineral dust in ice core trough digital holography technique.
The code present in this repository is based on [Holopy](https://github.com/manoharan-lab/holopy) repository by Manoharan Laboratory. 

The project is born from the need to quantify the role of mineral dust in radiative forcing. Nowadays, the uncertainty in aerosol forcing is nearly as large as the forcing value itself, making their influence on climate the least understood of effects.

The impact of aerosols is related to their optical properties, which dictate their effectiveness in scattering and absorbing both solar and terrestrial radiation, as well as the morphological and chemical properties that determine their ability to induce the nucleation of water droplets, hence the formation of clouds and rainclouds.<br>
Among aerosols, mineral dust is responsible for the most significant contribution to the dry mass particle load in the troposphere. Deviations from ideal spheres have an appreciable impact on the radiative forcing component from aerosols.

To this end, particle-by-particle optical measurements contribute significantly to an all-round characterization of mineral dust, by giving direct access to their optical properties, especially if many parameters are measured simultaneously and independently.

* With Holopy you may load images with experimental metadata, reconstruct 3D volumes from digital holograms, do scattering calculations, and make precise measurements by fitting scattering models.

* With HoloSoft you may analyze melted ice core dust sample obtaining a direct measurement of the extinction cross-section of the particles and an image of their silhouette when crossing a visible laser beam.


# How to install 

## Clone the repository

`git https://github.com/OpticaLab/HoloSoft`

## Intall the requirements

`cd HoloSoft/` <br/>
`pip3 install -r requirements.txt`

## Run the script `setup.py`

If you want to install it locally you may set the prefix directory as follow. <br/>
`python3 setup.py install --prefix=~/.local`

## Export the `PATH`

In order to make available the script from command line you have to re-define the `PATH` variable appending the `~/.local/bin`.  <br/>
You may add this line in your `~/.bashrc`  <br/>
`export PATH=$PATH:~/.local/bin`


