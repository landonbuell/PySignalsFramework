"""
PySignalsFramework
setup.py
Landon Buell
15 June 2021
"""

import setuptools

VERSION = '0.0.1'
DESCRIPTION = 'Python Package for Digital Signal Processing'

setuptools.setup(
		name			= "PySignalsFramework",
		version			= VERSION,
		author			= "Landon Buell",
		description		= DESCRIPTION,
		url				= "https://github.com/landonbuell/PySignalsFramework",
		package_dir		= {"": "PySignalsFramework"},
		packages		= setuptools.find_packages(where="PySignalsFramework"),
		python_requires	= ">=3.6",
	
	)