from setuptools import setup

setup(name='gmutils',
      version='0.2.0',
      description='Python 3 utils to simplify natural language machine learning code',
      url='https://github.com/grahammorehead/gmutils.git',
      author='grahammorehead',
      license='GPL',
      packages=['gmutils'],
      install_requires=['numpy', 'scipy', 'dill', 'requests', 'editdistance'],
      zip_safe=False)
