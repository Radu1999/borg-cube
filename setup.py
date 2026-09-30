from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="borg-cube",
    version="0.1.0",
    description="A complete NLP processing pipeline using DeBERTa-v3 and adapters",
    packages=find_packages(include=["borg", "src", "src.*", "eval"]),
    entry_points={
        "console_scripts": ["borg=borg.cli:main"],
    },
    install_requires=install_requires,
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
