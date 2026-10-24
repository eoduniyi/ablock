from setuptools import setup, find_packages

setup(
    name="ablock",
    version="1.0.0",
    description="A-Block: Clifford Neural Score Matching for Multi-Well Conformational Dynamics and Active Particle Sampling",
    author="Erick Oduniyi",
    author_email="erick000@media.mit.edu",
    url="https://github.com/eoduniyi/ablock",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.22.0",
        "scipy>=1.8.0",
        "matplotlib>=3.5.0",
        "pillow>=9.0.0",
    ],
)
