import setuptools

setuptools.setup(
    name='object_states',
    version='0.0.1',
    description='Object state classification',
    long_description=open('README.md').read().strip(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    install_requires=[
        'torch', 
        'torchvision',
        'numpy',
        'opencv-python', 
        'fiftyone',
        'supervision',
        'tqdm',
        'pathtrees',
        'pandas',
    ],
    extras_require={})
