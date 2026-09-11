from setuptools import find_packages, setup


setup(
    name="semantic-commerce-cs",
    version="0.1.0",
    description="Two-level semantic cache and hybrid RAG demo for e-commerce customer service.",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.111.0",
        "uvicorn[standard]>=0.30.0",
        "pydantic>=2.7.0",
        "pydantic-settings>=2.3.0",
        "redis>=5.0.0",
        "redisvl>=0.3.0",
        "numpy>=1.26.0",
        "rapidfuzz>=3.9.0",
        "httpx>=0.27.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.2.0",
            "ruff>=0.5.0",
        ],
    },
)
