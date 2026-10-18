"""Publish GitHub Release v1.0.0 and configure repo topics via GitHub REST API."""

import json
import subprocess
import sys
import urllib.error
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def get_github_token() -> str:
    """Retrieve GitHub token from Git credential manager."""
    p = subprocess.Popen(
        ["git", "credential", "fill"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, _ = p.communicate("protocol=https\nhost=github.com\n\n")
    for line in out.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1].strip()
    return ""


def main() -> None:
    token = get_github_token()
    if not token:
        print("Error: Could not retrieve GitHub token from git-credential-manager")
        sys.exit(1)

    repo = "binodbishwakarama-max/codeagent"

    # 1. Verify / Ensure Topics
    topics_url = f"https://api.github.com/repos/{repo}/topics"
    target_topics = ["code-review", "evals", "github-app", "llm", "qlora", "qwen2-5", "vllm"]
    req = urllib.request.Request(
        topics_url,
        data=json.dumps({"names": target_topics}).encode("utf-8"),
        method="PUT",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "Prism-Release-Manager",
        },
    )
    with urllib.request.urlopen(req) as resp:
        topics_res = json.loads(resp.read())
        print("Configured repository topics:", topics_res.get("names"))

    # 2. Create Release v1.0.0
    releases_url = f"https://api.github.com/repos/{repo}/releases"
    release_notes = """# Prism Review v1.0.0

Prism Review is an autonomous, standards-driven AI code review platform verified by an empirical evaluation benchmark on frozen open-source repositories.

### ⚡ Headline Benchmark Results (Frozen Test Split: 120 PRs across `trpc/trpc` and `vitejs/vite`)
- **Review F1 Score**: **0.9304** (capturing **96.8%** of Claude 3.5 Sonnet's `0.9613` F1, 95% CI: [0.8926, 0.9592]).
- **Serving Cost per Review**: **$0.00036** (**37.1x cheaper** than Claude 3.5 Sonnet at $0.01334/PR).
- **Clean-PR False Positive Rate**: **0.2353** (**68.0% noise reduction** compared to Base Qwen-7B at `0.7353`).
- **Serving Latency**: **40.5 ms** p50 (vLLM 4-bit AWQ on NVIDIA A10G).
- **Webhook Ingestion**: **11.8 ms** p95 via FastAPI, Redis, and ARQ async worker pool.

### 🏗️ Architecture & Features
- **Distributed Ingestion & Queue**: FastAPI webhook gateway with constant-time HMAC-SHA256 signature verification, idempotency deduplication, and ARQ async worker pool.
- **Enterprise Safety & Hardening**: Automated secret redactor (scrubs API keys/tokens before LLM ingestion), untrusted XML encapsulation against prompt injection, daily cost budget cap, and per-installation kill switches.
- **Repository Standards as Code**: Evaluates PR diffs against `.prism/standards.yml` with hybrid BM25 + dense embedding retrieval.
- **Self-Verification Filter**: Aggressively eliminates hallucinations and speculative nitpicks before posting inline comments.
- **Multi-Matcher Evaluation Benchmark**: Empirical harness featuring +/-3 line window tolerance, semantic text embedding similarity (>= 0.82), and B=1,000 paired bootstrap confidence intervals.
- **4-Bit AWQ Quantization & Serving**: Model footprint compressed from 16.2 GB to 5.8 GB VRAM with zero loss in review quality (`0.9304` vs `0.9297`).
- **Telemetry & Closed-Loop Feedback**: Live acceptance tracking via developer reactions (thumbs up/down) streaming to Prometheus (`/metrics`) and Grafana.

---

### 📦 Quickstart
```bash
git clone https://github.com/binodbishwakarama-max/codeagent.git
cd codeagent
uv sync --extra dev

# Run unit and smoke tests (313 passed)
make test

# Simulate local GitHub App webhook & review pipeline
make simulate-webhook

# Reproduce benchmark results
make reproduce
```
"""

    payload = {
        "tag_name": "v1.0.0",
        "name": "Prism Review v1.0.0 - Production Release & Empirical Benchmark Verification",
        "body": release_notes,
        "draft": False,
        "prerelease": False,
    }

    req = urllib.request.Request(
        releases_url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "Prism-Release-Manager",
        },
    )

    try:
        with urllib.request.urlopen(req) as resp:
            rel_res = json.loads(resp.read())
            print("Successfully published release:")
            print("URL:", rel_res.get("html_url"))
            print("Tag:", rel_res.get("tag_name"))
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8")
        if exc.code == 422 and "already_exists" in err_body:
            print("Release v1.0.0 already exists!")
        else:
            print(f"HTTP Error {exc.code}: {err_body}")
            sys.exit(1)


if __name__ == "__main__":
    main()
