from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='probabilistic_forecast',
   version='0.1.0',
   description='deep probabilistic models applied to air quality forcasting',
   license="MIT",
   long_description=long_description,
   author='Abdulmajid Murad',
   author_email='abdulmajid.a.murad@ntnu.no',
   url="https://www.ntnu.edu/employees/abdulmajid.a.murad",
   packages=['probabilistic_forecast']
#    install_requires=['pytorch', 'pandas', 'numpy', 'scipy', 'matplotlib']
)
