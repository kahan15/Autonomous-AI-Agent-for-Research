from setuptools import setup, find_packages

setup(
    name="research_agent",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.3",
        "python-dotenv>=1.0.1",
        "feedparser>=6.0.11",
        "pyyaml>=6.0.1",
        "scholarly>=1.7.11"
    ],
    python_requires=">=3.8",
) 