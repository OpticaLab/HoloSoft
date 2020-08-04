from distutils.core import setup
from os import *

setup(
        name="HoloSoft",
        version="0.0.1",
        author="Claudia Ravasio",
        author_email="claudia.ravasio@unimi.it",
        descripton="Hologram Package Analysis", 
        packages=['HoloSoft'],
        package_dir={'HoloSoft':'src'},
        scripts=['analysis/ImgCorrect.py','analysis/main_polystyrene.py', 'analysis/main_dust.py', 'analysis/FindCenters.py'],
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approves :: GPL3",
            "Operative System :: OS Indipendent",
            ],
        python_requires=">=3.8",
)
