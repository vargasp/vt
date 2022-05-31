from setuptools import setup
from os import path

# read the contents of README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

DEPENDENCIES = ['numpy', 'matplotlib']

setup(name='vt',
      version='0.1',
      author='Phillip Vargas',
      author_email='vargasp@uchicago.edu',
      description='Wrapper functions for matplotlib',
      long_description=long_description,
      license='MIT',
      install_requires=DEPENDENCIES,
      url='https://github.com/vargasp/vt',
      keywords=['visulaization'],
      packages=['vt'],
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License'
      ]
      )