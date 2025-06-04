"""
Research Agent package.

This package provides the core functionality for an autonomous research agent
that follows a Plan-Execute-Reflect workflow pattern.
"""

from .base import ResearchAgent, PlanStep, ExecutionResult
from .web_research_agent import WebResearchAgent

__all__ = ['ResearchAgent', 'WebResearchAgent', 'PlanStep', 'ExecutionResult']