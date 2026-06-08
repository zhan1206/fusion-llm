"""Fusion-LLM setup configuration."""
from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="fusion-llm",
    version="1.2.0",
    description="Fusion LLM — SBLA attention + Thinking Dial, open-source local training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="zhan1206",
    author_email="zhan1206@example.com",
    url="https://github.com/zhan1206/fusion-llm",
    license="Apache License 2.0",
    packages=find_packages(exclude=["tests*", "experiments*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0.0",
    ],
    extras_require={
        # Core inference
        "core": [
            "transformers>=4.36.0",
            "sentencepiece>=0.1.99",
        ],
        # Training
        "training": [
            "transformers>=4.36.0",
            "accelerate>=0.25.0",
            "datasets>=2.16.0",
            "peft>=0.7.0",
            "bitsandbytes>=0.41.0",
            "deepspeed>=0.12.0",
            "pandas>=2.0.0",
        ],
        # Inference deployment
        "inference": [
            "onnx>=1.14.0",
            "onnxruntime>=1.16.0",
        ],
        # Evaluation
        "eval": [
            "datasets>=2.16.0",
            "bert-score>=0.3.13",
            "rouge-score>=0.1.2",
            "moverscore>=1.0.0",
            "matplotlib>=3.7.0",
        ],
        # Development
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        # Full installation
        "all": [
            "transformers>=4.36.0",
            "sentencepiece>=0.1.99",
            "accelerate>=0.25.0",
            "datasets>=2.16.0",
            "peft>=0.7.0",
            "bitsandbytes>=0.41.0",
            "deepspeed>=0.12.0",
            "pandas>=2.0.0",
            "onnx>=1.14.0",
            "onnxruntime>=1.16.0",
            "bert-score>=0.3.13",
            "rouge-score>=0.1.2",
            "moverscore>=1.0.0",
            "matplotlib>=3.7.0",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fusion-train=train.train_mini:main",
            "fusion-eval=evaluation.metrics:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="llm transformer attention sbla thinking-dial local-training",
)
