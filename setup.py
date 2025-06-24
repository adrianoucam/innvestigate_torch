from setuptools import setup, find_packages

setup(
    name='innvestigate_torch',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'matplotlib',
        'numpy',
        'Pillow'
    ],
    author='Adriano Lima e Souza',
    author_email='adrianoucam@gmail.com',
    description='Versão PyTorch da biblioteca iNNvestigate para métodos de explicabilidade',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/seuusuario/innvestigate_torch',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
