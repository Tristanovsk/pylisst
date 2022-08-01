# import ez_setup
# ez_setup.use_setuptools()

from setuptools import setup, find_packages


__package__ = 'PyLisst'
__version__ = '1.0.0'

setup(
    name=__package__,
    version=__version__,
    packages=find_packages(exclude=['build']),
    package_data={
         '': ['*.nc','*.txt','*.csv','*.dat'],
        # 'aux': ['data/aux/*']
    },
    include_package_data=True,

    url='',
    license='MIT',
    author='T. Harmel',
    author_email='tristan.harmel@gmail.com',
    description='Scientific code to process LISST-VSF measurements',

    # Dependent packages (distributions)
    install_requires=['numpy','scipy','pandas','xarray','lmfit',
                      'matplotlib','datetime'],

    entry_points={
          'console_scripts': [
              'pylisst = TODO'
          ]}
)
