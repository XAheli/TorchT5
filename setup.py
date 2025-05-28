from setuptools import setup, find_packages

setup(
    name="t5-from-scratch",
    version="0.1.0",
    description="T5 (Text-to-Text Transfer Transformer) implementation from scratch in PyTorch",
    author="Aheli Poddar",
    author_email="ahelipoddar2003@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.25.0",
        "sentencepiece>=0.1.97",
        "numpy>=1.21.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
    },
)
