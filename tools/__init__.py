"""Tools package: the deterministic execution layer.

The model (Qwen) never executes code directly. It selects a tool by name
and supplies arguments; the registry in registry.py validates and
executes the corresponding function. No tool in this package contains
any LLM reasoning.
"""
