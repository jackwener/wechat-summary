"""
wechat — Locate→Act framework for macOS WeChat automation.

All UI operations decompose into:
  Locate(element) → find position via layout detection or OCR
  Act(action)     → click, type, scroll, screenshot at that position

Element bridges Locate and Act — it carries position and supports .click().
Snapshot caches screenshot + OCR results for efficient multi-query access.
"""

from wechat.element import Element, Snapshot
from wechat.locator import Locator
from wechat import actions

__all__ = ["Element", "Snapshot", "Locator", "actions"]
