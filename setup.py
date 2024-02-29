from setuptools import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='hiermodelutils',
    version='0.1.0',    
    description='Data processing utilities for hierarchical modeling.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    # url='https://github.com/PredictiveScienceLab/orthojax',
    author='Wesley Holt',
    author_email='holtw@purdue.edu',
    license='MIT License',
    packages=['hiermodelutils'],
    install_requires=['pandas'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: OS Independent',        
        'Programming Language :: Python :: 3'
    ],
)