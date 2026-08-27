"""Compatibility shim for editable installs with pip versions before PEP 660."""

from setuptools import setup


setup(
    name="evidenceviewbench",
    version="0.1.2",
    description="Private development benchmark for agent-selected numerical evidence",
    packages=["evidenceviewbench", "evidenceviewbench.resources"],
    package_data={"evidenceviewbench.resources": ["*.jsonl", "*.jsonl.gz"]},
    include_package_data=True,
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "evidenceviewbench-validate=evidenceviewbench.validation:main",
        ]
    },
)
