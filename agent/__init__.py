"""Agent package: model interface, agent state, and the control loop.

This package intentionally contains NO dataset-specific logic. It only
knows how to:
  1. Ask the model (Qwen) for the next action, given the current state.
  2. Hand that action to the tool registry for controlled execution.
  3. Record the observation and decide whether to continue or stop.
"""
