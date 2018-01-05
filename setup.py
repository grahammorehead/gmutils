from setuptools import setup

setup(name='gmutils',
      version='0.0.1',
      description='Python 3 utils to simplify natural language machine learning code',
      url='https://github.com/grahammorehead/gmutils.git',
      author='grahammorehead',
      license='GPL',
      packages=['gmutils'],
      install_requires=[
          'spacy', 'pandas', 'numpy', 'scipy', 'tensorflow', 'graphviz', 'pydot', 'matplotlib', 'keras', 'elasticsearch'
      ],
      zip_safe=False)
