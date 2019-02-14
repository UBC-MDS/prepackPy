from distutils.core import setup


DESCRIPTION         = "prepackPy: data pre-processing package"
LONG_DESCRIPTION    = open("README.md").read()
NAME                = "prepackPy"
PACKAGES            = ["prepackPy"]
AUTHOR              = "Jingyun Chen | Anthony Chiodo | Sarah Watts |"
URL                 = "https://github.com/UBC-MDS/prepackPy"
DOWNLOAD_URL        = "https://github.com/UBC-MDS/prepackPy"
LICENSE             = "MIT"
INSTALL_REQUIRES    = ["pandas", "numpy", "pytest",]
VERSION             = "2.0dev"


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    url=URL,
    download_url=DOWNLOAD_URL,
    license=LICENSE,
    packages=PACKAGES,
    install_requires=INSTALL_REQUIRES
)