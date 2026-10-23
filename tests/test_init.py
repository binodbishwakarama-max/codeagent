"""Unit tests for prism init repository inspection and standards scaffolding."""

import json
from pathlib import Path

import pytest

from prism.models import StandardsFile
from prism.review.cli import handle_init_command
from prism.review.init import (
    detect_tech_stack,
    scaffold_prism,
)


def test_detect_tech_stack_python_fastapi(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """[project]
name = "demo-api"
dependencies = [
    "fastapi>=0.111.0",
    "sqlalchemy>=2.0.0",
    "pydantic>=2.7.0",
]
""",
        encoding="utf-8",
    )
    res = detect_tech_stack(tmp_path)
    assert "Python" in res["languages"]
    assert "FastAPI" in res["frameworks"]
    assert "SQLAlchemy" in res["frameworks"]
    assert "Pydantic" in res["frameworks"]


def test_detect_tech_stack_typescript_next(tmp_path: Path) -> None:
    pkg = tmp_path / "package.json"
    pkg.write_text(
        json.dumps({
            "name": "next-app",
            "dependencies": {
                "next": "^14.0.0",
                "react": "^18.2.0",
                "@prisma/client": "^5.0.0",
            },
        }),
        encoding="utf-8",
    )
    (tmp_path / "tsconfig.json").write_text("{}", encoding="utf-8")

    res = detect_tech_stack(tmp_path)
    assert "TypeScript" in res["languages"]
    assert "Next.js" in res["frameworks"]
    assert "React" in res["frameworks"]
    assert "Prisma" in res["frameworks"]


def test_detect_tech_stack_go_gin(tmp_path: Path) -> None:
    go_mod = tmp_path / "go.mod"
    go_mod.write_text(
        """module github.com/acme/service

go 1.22

require (
    github.com/gin-gonic/gin v1.9.1
    gorm.io/gorm v1.25.0
)
""",
        encoding="utf-8",
    )
    res = detect_tech_stack(tmp_path)
    assert "Go" in res["languages"]
    assert "Gin" in res["frameworks"]
    assert "GORM" in res["frameworks"]


def test_detect_tech_stack_rust_axum(tmp_path: Path) -> None:
    cargo = tmp_path / "Cargo.toml"
    cargo.write_text(
        """[package]
name = "api-service"
version = "0.1.0"

[dependencies]
axum = "0.7"
tokio = { version = "1.0", features = ["full"] }
""",
        encoding="utf-8",
    )
    res = detect_tech_stack(tmp_path)
    assert "Rust" in res["languages"]
    assert "Axum" in res["frameworks"]
    assert "Tokio" in res["frameworks"]


def test_scaffold_prism_creates_valid_standards(tmp_path: Path) -> None:
    # Set up dummy Python repo
    (tmp_path / "requirements.txt").write_text("fastapi\nhttpx\n", encoding="utf-8")

    standards_path, workflow_path, langs, frameworks, count = scaffold_prism(
        target_dir=tmp_path,
        force=False,
        include_workflow=True,
    )

    assert standards_path.exists()
    assert workflow_path is not None
    assert workflow_path.exists()
    assert "Python" in langs
    assert "FastAPI" in frameworks
    assert count >= 4

    # Validate that generated YAML is parseable and valid
    parsed = StandardsFile.from_file(standards_path)
    assert len(parsed.standards) == count
    rule_ids = [s.id for s in parsed.standards]
    assert "STD-SEC-01" in rule_ids
    assert "STD-FASTAPI-ASYNC-01" in rule_ids


def test_scaffold_prism_file_exists_error_without_force(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
    scaffold_prism(target_dir=tmp_path, force=False)

    # Calling again without force should raise FileExistsError
    with pytest.raises(FileExistsError):
        scaffold_prism(target_dir=tmp_path, force=False)

    # Calling with force should succeed
    scaffold_prism(target_dir=tmp_path, force=True)


def test_handle_init_command_cli(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module demo\ngo 1.22\n", encoding="utf-8")
    exit_code = handle_init_command([str(tmp_path), "--workflow", "--force"])
    assert exit_code == 0
    assert (tmp_path / ".prism" / "standards.yml").exists()
    assert (tmp_path / ".github" / "workflows" / "prism.yml").exists()
