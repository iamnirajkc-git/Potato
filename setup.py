# The setup.py file is primarily used for packaging and distributing Python packages through tools like pip or for defining dependencies and metadata 
# when you're distributing your code as a reusable Python package.
#For this project bwe may not need this file.

import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = "Potato"
AUTHOR_USER_NAME = "iamnirajkc-git"
SRC_REPO = "cnnClassifier"
AUTHOR_EMAIL = "kcniraj44@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for CNN app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)