"""TDD Integrity Guard for Spec-Kit Repository Documentation (.specify/ and specs/001-staff-rag-platform/).

Enforces zero documentation drift:
1. All required Spec-Kit artifacts exist (Constitution, Spec, Plan, Data Model, API/Protocol Contracts, Ops Runbook).
2. Every FastAPI HTTP route in main.app is documented in contracts/api-and-protocols.md.
3. Every backend/app/<package> directory is documented in plan.md.
4. Every MCP tool and Google A2A skill is documented in contracts/api-and-protocols.md.
5. All Spec-Kit docs are free of OWASP / prompt-injection / tool-poisoning signatures.
"""

import re
from pathlib import Path

from main import app
from mcp_server import MCP_TOOL_MANIFEST
from app.a2a.server import build_agent_card


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_ROOT = REPO_ROOT / "specs" / "001-staff-rag-platform"
CONSTITUTION_PATH = REPO_ROOT / ".specify" / "memory" / "constitution.md"

POISONING_SIGNATURES = [
    re.compile(r"<\s*IMPORTANT\s*>", re.IGNORECASE),
    re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"exfiltrate|curl\s+http|wget\s+http|powershell\s+-enc", re.IGNORECASE),
    re.compile(r"read\s+~/\.ssh|/etc/passwd|id_rsa", re.IGNORECASE),
]


def test_spec_kit_core_documents_exist_and_are_non_empty():
    required_files = [
        CONSTITUTION_PATH,
        SPEC_ROOT / "spec.md",
        SPEC_ROOT / "plan.md",
        SPEC_ROOT / "data-model.md",
        SPEC_ROOT / "contracts" / "api-and-protocols.md",
        SPEC_ROOT / "operations-runbook.md",
    ]
    for path in required_files:
        assert path.exists(), f"Missing Spec-Kit document: {path}"
        content = path.read_text(encoding="utf-8")
        assert len(content.strip()) >= 500, f"Spec-Kit document unexpectedly short: {path}"


def test_every_fastapi_route_mcp_tool_and_a2a_skill_documented_in_contracts():
    contracts_text = (SPEC_ROOT / "contracts" / "api-and-protocols.md").read_text(encoding="utf-8")

    skip_paths = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
    missing_routes = []
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = sorted(m for m in (getattr(route, "methods", None) or set()) if m not in {"HEAD", "OPTIONS"})
        if path and path not in skip_paths and methods:
            if path not in contracts_text:
                missing_routes.append(f"{','.join(methods)} {path}")
    assert not missing_routes, f"Undocumented FastAPI routes in contracts/api-and-protocols.md: {missing_routes}"

    for tool in MCP_TOOL_MANIFEST:
        assert tool["name"] in contracts_text, f"Undocumented MCP tool: {tool['name']}"

    card = build_agent_card("http://localhost:8000")
    for skill in card["skills"]:
        assert skill["id"] in contracts_text, f"Undocumented A2A skill: {skill['id']}"


def test_every_backend_app_package_documented_in_plan():
    plan_text = (SPEC_ROOT / "plan.md").read_text(encoding="utf-8")
    app_dir = REPO_ROOT / "backend" / "app"
    packages = sorted(
        p.name for p in app_dir.iterdir()
        if p.is_dir() and not p.name.startswith("__")
    )
    missing_pkgs = [pkg for pkg in packages if f"`app/{pkg}" not in plan_text and f"backend/app/{pkg}" not in plan_text]
    assert not missing_pkgs, f"Undocumented backend/app packages in plan.md: {missing_pkgs}"


def test_spec_kit_docs_free_of_injection_and_poisoning_signatures():
    md_files = list((REPO_ROOT / ".specify").rglob("*.md")) + list((REPO_ROOT / "specs").rglob("*.md"))
    assert len(md_files) >= 6, "Expected at least 6 Spec-Kit markdown files"
    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8")
        for sig in POISONING_SIGNATURES:
            assert not sig.search(content), f"Security signature {sig.pattern} found in {md_file}"
