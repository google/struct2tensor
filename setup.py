# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Setup for pip package."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install
from setuptools.dist import Distribution


# struct2tensor is not a purelib. However because of the extension module is not
# built by setuptools, it will be incorrectly treated as a purelib. The
# following works around that bug.
class _InstallPlatlib(install):

  def finalize_options(self):
    install.finalize_options(self)
    self.install_lib = self.install_platlib


class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

  def is_pure(self):
    return False

  def has_ext_modules(self):
    return True

# Get the long description from the README file.
with open('README.md') as fp:
  _LONG_DESCRIPTION = fp.read()

with open('struct2tensor/version.py') as fp:
  globals_dict = {}
  exec(fp.read(), globals_dict)  # pylint: disable=exec-used
__version__ = globals_dict['__version__']

setup(
    name='struct2tensor',
    version=__version__,
    description=('Struct2Tensor is a package for parsing and manipulating'
                 ' structured data for TensorFlow'),
    author='Google LLC',
    author_email='tensorflow-extended-dev@googlegroups.com',
    # Contained modules and scripts.
    packages=find_packages(),
    install_requires=[
        'protobuf>=3.8.0,<4',
        'tensorflow>=2.2.0,<2.3',
        'tensorflow_metadata>=0.21,<0.22',
        'pyarrow>=0.15,<1',
    ],
    # Add in any packaged data.
    include_package_data=True,
    package_data={'': ['*.lib', '*.so']},
    zip_safe=False,
    distclass=BinaryDistribution,
    # PyPI package information.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Apache 2.0',
    keywords='tensorflow protobuf machine learning',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    cmdclass={'install': _InstallPlatlib})
