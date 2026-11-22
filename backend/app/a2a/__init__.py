"""Google A2A (Agent-to-Agent) Protocol server, Agent Card discovery, and anti-confused-deputy RBAC."""

from app.a2a.server import a2a_manager, build_agent_card

__all__ = ["a2a_manager", "build_agent_card"]
