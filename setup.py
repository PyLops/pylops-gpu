import os
from distutils.core import setup
from setuptools import find_packages

def src(pth):
    return os.path.join(os.path.dirname(__file__), pth)

# Project description
descr = """
        An extension to PyLops for linear operators on GPUs.
        """

# Setup
setup(
    name='pylops_gpu',
    description=descr,
    long_description=open(src('README.md')).read(),
    long_description_content_type='text/markdown',
    keywords=['algebra',
              'inverse problems',
              'large-scale optimization'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    author='mrava',
    author_email='mrava@equinor.com',
    install_requires=['numpy >= 1.15.0',
                      'torch >= 1.2.0',
                      'pytorch_complex_tensor',
                      'pylops'],
    packages=find_packages(exclude=['pytests']),
    use_scm_version=dict(root = '.',
                         relative_to = __file__,
                         write_to = src('pylops_gpu/version.py')),
    setup_requires=['pytest-runner', 'setuptools_scm'],
    test_suite='pytests',
    tests_require=['pytest'],
    zip_safe=True)
