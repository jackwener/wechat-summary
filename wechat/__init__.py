"""
wechat — Locate→Act framework for macOS WeChat automation.

All UI operations decompose into:
  Locate(element) → find position via layout detection or OCR
  Act(action)     → click, type, scroll, screenshot at that position
"""

from wechat.locator import Locator
from wechat import actions

__all__ = ["Locator", "actions"]
