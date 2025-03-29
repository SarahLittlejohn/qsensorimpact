from setuptools import setup, find_packages

setup(
    name="qsensorimpact",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",  # Add any dependencies your package needs
        "scipy"
    ],
    author="S Littlejohn",
    author_email="sarahjanelittlejohn@gmail.com",
    description="A package for investigating how error signatures in large-scale qubit arrays can be leveraged to extract meaningful physical information.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/SarahLittlejohn/qsensorimpact",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
