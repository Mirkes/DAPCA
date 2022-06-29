#! /usr/bin/env python
# FOR THE LICENSE, see LICENSE file

import codecs
import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join('DAPCA', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'DAPCA'
DESCRIPTION = 'Domain Adaptation Principal Component Analysis'
MAINTAINER = 'Andrei Zinovyev, Eugene Mirkes'
MAINTAINER_EMAIL = 'zinovyev@gmail.com'
URL = 'http://andreizinovyev.site'
LICENSE = 'GNU'
DOWNLOAD_URL = 'https://github.com/Mirkes/DAPCA/'
VERSION = __version__
INSTALL_REQUIRES = ['numpy', 'matplotlib', 'scikit-learn']
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.7',
               'Programming Language :: Python :: 3.8']

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages()
      install_requires=INSTALL_REQUIRES)
