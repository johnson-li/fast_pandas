import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fast-pandas",
    version="0.0.1",
    author="Johnson Li",
    author_email="johnsonli1993@gmail.com",
    description="Make your pandas run faster",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/johnson-li/fast-pandas",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
