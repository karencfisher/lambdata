from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="lambdata-karen", # the name that you will install via pip
    version="0.2.0",
    author="Karen Fisher",
    author_email="karen-fisher@lambdastudents.com",
    description="A buffet of statistical and other useful functions",
    long_description=long_description,
    long_description_content_type="text/markdown", # required if using a md file for long desc
    license="MIT",
    url="https://github.com/karencfisher/lambdata",
    #keywords="",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)