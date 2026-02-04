from setuptools import setup, find_packages

setup(
    name="KG_RAG",
    version="0.1.0",
    package_dir={"": "src"},  # Add this line to tell setuptools to look in src/
    packages=find_packages(where="src"),  # Modify this line to look in src/
    install_requires=[],  # Add dependencies here if needed
)