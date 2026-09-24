"""Baselines every learned model must beat, measured on the same splits."""

from spoiler_shield.baselines.base import SpoilerModel
from spoiler_shield.baselines.rules import KeywordRules
from spoiler_shield.baselines.tfidf import TfidfLogReg

__all__ = ["KeywordRules", "SpoilerModel", "TfidfLogReg"]
