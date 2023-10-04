from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

long_description = """
This project is focused on a unique and challenging task: Given two fMRI scans, the objective is to predict whether these scans belong to twins. The rationale behind this endeavor is rooted in the understanding that twins may share similar neurological patterns and structures, which might be discernible in their fMRI scans.

To accomplish this classification task, we utilize PyTorch, a leading deep learning framework. PyTorch provides the flexibility and computational prowess required to handle and process complex fMRI data. By leveraging its extensive functionalities and powerful neural network modules, we aim to develop a robust model that can accurately discern the subtle nuances between twin and non-twin fMRI scans.

Whether you're a neuroscientist curious about the capabilities of deep learning in your domain, or a machine learning enthusiast eager to explore the intricacies of fMRI data, this project offers a fascinating intersection of the two fields.
"""

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='twin status prediction with fmri images.',
    author='Sven Maurice Morlock ',
    license='MIT',
    python_requires='>=3.7',
    url='https://github.com/mms1410/fmri_twins_prediction', 
    #long_description=long_description, 
    long_description=long_description,   
    classifiers=[
        "Development Status :: 3 - Alpha",  
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",

    ],
    author_email='sven.morlock@lmu.campus.de',
    keywords='fmri, twins, prediction, machine learning, neuroimaging',

)
