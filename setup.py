import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dempster_algorithm",
    version="0.0.1",
    description="Dempster's covariance selection algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Apollon76/dempster_algorithm",
    install_requires=[
        'numpy',
        'pandas',
    ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
