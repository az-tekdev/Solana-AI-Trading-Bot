"""Setup script for Solana AI Trading Bot."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="solana-ai-trading-bot",
    version="1.0.0",
    description="AI-powered trading bot for Solana blockchain",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Solana AI Trading Bot Contributors",
    author_email="",
    url="https://github.com/yourusername/Solana-AI-Trading-Bot",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "solders>=0.18.0",
        "solana>=0.30.0",
        "requests>=2.31.0",
        "httpx>=0.24.0",
        "websocket-client>=1.6.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "colorlog>=6.7.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "aiosqlite>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "solana-ai-bot=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
