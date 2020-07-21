from distutils.core import setup

setup(
        name="holoPkg",
        version="0.0.1",
        author="Claudia Ravasio",
        author_email="claudia.ravasio@unimi.it",
        descripton="Hologram Package Analysis", 
        packages=['holoPkg'],
        package_dir={'holoPkg':'src'},
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approves :: GPL3",
            "Operative System :: OS Indipendent",
            ],
        python_requires=">=3.8",
)
