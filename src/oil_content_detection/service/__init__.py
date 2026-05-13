"""Web service package for Huajiao oil-content demo application."""

from .app import create_app
from .settings import HuajiaoWebSettings

__all__ = ["HuajiaoWebSettings", "create_app"]
