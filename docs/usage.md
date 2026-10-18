# Prism Review: Installation & Usage Guide

Prism Review is an autonomous GitHub App that conducts high-signal code reviews on pull requests against custom repository engineering standards.

---

## 1. GitHub App Registration

To deploy your own instance of Prism Review, create a GitHub App in your GitHub account or organization:

1. Navigate to **GitHub Settings** → **Developer Settings** → **GitHub Apps** → **New GitHub App**.
2. Fill in the application details:
   - **GitHub App name**: `prism-code-reviewer`
   - **Homepage URL**: `https://prism-review.fly.dev`
   - **Webhook URL**: `https://prism-review.fly.dev/webhook/github`
   - **Webhook secret**: Generate a secure random token (e.g. `openssl rand -hex 32`) and save as `GITHUB_WEBHOOK_SECRET`.
3. Set the required **Repository Permissions**:
   | Permission | Access | Purpose |
   |---|---|---|
   | **Pull Requests** | Read & Write | Fetch PR diffs, post inline review comments |
   | **Checks** | Read & Write | Post and update check run conclusions (`in_progress`, `success`, `neutral`) |
   | **Issues** | Read & Write | Monitor developer comment reactions and thread resolution |
   | **Repository Contents** | Read-only | Fetch repository standards (`.prism/standards.yml`) |
   | **Metadata** | Read-only | Mandatory default permission for repo metadata |
4. Subscribe to the following **Webhook Events**:
   - `Pull request` (`opened`, `synchronize`, `reopened`)
   - `Pull request review comment` (`created`, `edited`, `deleted`)
   - `Pull request review thread` (`resolved`, `unresolved`)
5. Click **Create GitHub App**.
6. Under **Private keys**, click **Generate a private key** and download the `.pem` file. Save its content as `GITHUB_PRIVATE_KEY`.
7. Note down the **App ID** and save as `GITHUB_APP_ID`.

---

## 2. Installing Prism on 3 Repositories

Prism is installed across 3 distinct target repositories:

### Target Repositories:
1. **Primary Backend Repository**: `binod/codeagent` (Owner repository)
2. **Frontend Web Repository**: `binod/prism-demo-app` (Owner secondary repository)
3. **Open-Source Collaborator Repository**: `friend-collab/shared-microservice` (Partner repository with explicit permission)

### Installation Steps:
1. In your GitHub App page, go to **Install App**.
2. Select your account or target organization.
3. Choose **Only select repositories** and pick:
   - `binod/codeagent`
   - `binod/prism-demo-app`
   - `friend-collab/shared-microservice`
4. Click **Install**.
5. Note the generated numeric **Installation ID** (visible in URL: `https://github.com/settings/installations/<INSTALLATION_ID>`).

---

## 3. Configuring Repository Standards (`.prism/standards.yml`)

Add a `.prism/standards.yml` file to the root of each repository to declare your team's engineering rules.

```yaml
version: "1.0"
description: "Core engineering standards and architectural invariants for Prism review."

standards:
  - id: "STD-SEC-01"
    title: "No Hardcoded Secrets or Tokens"
    category: "SECURITY"
    severity: "HIGH"
    description: "Never commit API keys, private keys, database passwords, or auth tokens. Load all secrets strictly from environment variables or runtime secret managers."
    good_example: |
      api_key = os.environ.get("OPENAI_API_KEY")
      if not api_key:
          raise ValueError("OPENAI_API_KEY environment variable required")
    bad_example: |
      api_key = "sk-proj-9382109482019482019"

  - id: "STD-ERR-01"
    title: "Explicit Async Exception & Timeout Handling"
    category: "ERROR_HANDLING"
    severity: "HIGH"
    description: "All outbound network, HTTP, and database requests must declare an explicit timeout deadline and handle timeout/network exceptions gracefully without crashing workers."
    good_example: |
      try:
          response = await client.get(url, timeout=10.0)
          response.raise_for_status()
      except httpx.TimeoutException as exc:
          logger.warning("Upstream call timed out: %s", exc)
          raise ServiceUnavailableError() from exc
    bad_example: |
      response = await client.get(url)  # Unbounded timeout, swallowed crash

  - id: "STD-SQL-01"
    title: "Parameterized Database Queries"
    category: "SECURITY"
    severity: "HIGH"
    description: "Always use parameterized query placeholders or ORM bind parameters. Never interpolate user-supplied strings directly into SQL queries via format strings."
    good_example: |
      await db.execute(text("SELECT * FROM users WHERE email = :email"), {"email": email})
    bad_example: |
      await db.execute(f"SELECT * FROM users WHERE email = '{email}'")

  - id: "STD-PERF-01"
    title: "Avoid N+1 Database Queries in Loops"
    category: "PERFORMANCE"
    severity: "MEDIUM"
    description: "Avoid issuing individual database queries inside loops. Batch fetch records using SQL IN clauses, joined queries, or selectinload."
    good_example: |
      users = await session.execute(select(User).filter(User.id.in_(user_ids)))
    bad_example: |
      for user_id in user_ids:
          user = await session.get(User, user_id)

  - id: "STD-TYP-01"
    title: "Strict Type Annotations on Public Interfaces"
    category: "READABILITY"
    severity: "LOW"
    description: "All public methods, endpoints, and domain models must declare complete parameter and return type annotations."
    good_example: |
      async def compute_summary(repo: str, pr_number: int) -> dict[str, float]: ...
    bad_example: |
      def compute_summary(repo, pr_number): ...
```

---

## 4. Operational Safety Controls

### 🔴 Per-Installation Kill Switch
To instantly disable Prism on a specific repository or organization without revoking the GitHub App:
```bash
# In Fly.io secrets or environment variables:
fly secrets set BLOCKED_INSTALLATIONS="123456,789012"
```
When an installation ID is listed in `BLOCKED_INSTALLATIONS`:
- Webhook payloads are acknowledged immediately with 200 OK.
- No background review jobs are enqueued.
- Zero LLM calls or tokens are consumed.

### 💰 Daily Cost Budget Cap
To prevent runaway billing or unexpected PR flood costs:
```bash
fly secrets set DAILY_COST_BUDGET_USD="15.0" BUDGET_ENFORCED="true"
```
- Tracks cumulative daily review costs across all LLM providers in UTC days.
- When today's spend exceeds the budget threshold, LLM calls are disabled, check runs are marked neutral, and no comments are posted.

---

## 5. Live Review Workflow

Once installed, simply open a pull request on any registered repository:
1. Prism detects the `pull_request.opened` or `synchronize` webhook event.
2. An initial GitHub Check Run (`prism-review`) is created in status `in_progress`.
3. Prism parses the diff, scrubs secrets, retrieves relevant standards, and queries the LLM engine.
4. Inline comments are posted directly on lines violating standards.
5. If developers resolve comment threads or react with 👍, Prism updates its live acceptance rate.
