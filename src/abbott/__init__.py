"""3D Multiplexed Image Analysis Workflows"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("abbott")
except PackageNotFoundError:
    __version__ = "uninstalled"
