# GitHub App Setup Guide

This guide walks you through registering the **prism-review** GitHub App, configuring the correct permissions and webhook events, and wiring the credentials into your environment.

---

## Prerequisites

- A GitHub account with access to the organisation or personal namespace where you want to install the app
- A publicly reachable HTTPS URL for your webhook endpoint (e.g., via [ngrok](https://ngrok.com) locally, or your production domain)
- `openssl` on your local machine (for key conversion, if needed)

---

## Step 1 - Create the GitHub App

1. Go to **GitHub Settings -> Developer settings -> GitHub Apps**.
   - For an organisation: `https://github.com/organizations/<ORG>/settings/apps`
   - For a personal account: `https://github.com/settings/apps`
2. Click **New GitHub App**.

### App Metadata

| Field | Value |
| :--- | :--- |
| **GitHub App name** | `prism-review` (must be globally unique; append your org if needed) |
| **Homepage URL** | `https://github.com/your-org/prism-review` |
| **Description** | `Automated PR reviews against custom engineering standards` |

### Webhook

| Field | Value |
| :--- | :--- |
| **Webhook URL** | `https://<your-domain>/webhook/github` |
| **Webhook secret** | A random secret you generate: `openssl rand -hex 32` |
| **SSL verification** | Enabled (required for production) |

---

## Step 2 - Configure Permissions

Under **Repository permissions**, set the following:

| Permission | Access |
| :--- | :--- |
| **Contents** | Read-only |
| **Metadata** | Read-only (automatically set) |
| **Pull requests** | Read and write |

> **IMPORTANT:** `Pull requests: Read and write` is required for posting inline review comments and dismissing old reviews.  
> `Contents: Read` is required to fetch file contents for context.

Leave all **Organization permissions** and **Account permissions** at their defaults (No access).

---

## Step 3 - Subscribe to Webhook Events

Under **Subscribe to events**, check **only**:

- [x] `Pull request`

All other events can remain unchecked to minimise noise.

---

## Step 4 - Installation Settings

| Field | Value |
| :--- | :--- |
| **Where can this GitHub App be installed?** | `Only on this account` (for private use) or `Any account` (public) |

Click **Create GitHub App**.

---

## Step 5 - Generate and Download a Private Key

1. After creation, scroll to **Private keys** on your app's settings page.
2. Click **Generate a private key** - GitHub downloads a `.pem` file automatically.
3. Save it securely. You'll need it for the `GITHUB_PRIVATE_KEY` environment variable.

> **CAUTION:** Never commit the `.pem` file to version control. Add `*.pem` to `.gitignore`.

### Verify key format

The downloaded key is in PKCS#1 format which `pyjwt` with `cryptography` accepts directly. No conversion is normally needed.

To verify the key is readable:

```bash
openssl rsa -in your-app.2024-01-01.private-key.pem -check
```

---

## Step 6 - Install the App on a Repository

1. From the app's settings page, click **Install App** in the left sidebar.
2. Choose the organisation or user account.
3. Select **Only select repositories** and choose the repository (or repositories) you want Prism to review.
4. Click **Install**.

Note the **Installation ID** - you can find it in the URL after installation:
`https://github.com/settings/installations/<INSTALLATION_ID>`

---

## Step 7 - Configure Environment Variables

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

| Variable | Where to find it |
| :--- | :--- |
| `GITHUB_APP_ID` | App settings page -> **App ID** field |
| `GITHUB_PRIVATE_KEY` | Path to the downloaded `.pem` file, **or** the raw PEM contents (newlines as `\n`) |
| `GITHUB_WEBHOOK_SECRET` | The secret you generated in Step 1 |

Example `.env` entries:

```dotenv
# GitHub App credentials
GITHUB_APP_ID=123456
GITHUB_PRIVATE_KEY=/run/secrets/github_private_key.pem
GITHUB_WEBHOOK_SECRET=a4b3c2d1e0...

# Alternatively, inline the PEM (useful for container secrets):
# GITHUB_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----\nMIIEo...\n-----END RSA PRIVATE KEY-----"
```

> **TIP:** In production, inject `GITHUB_PRIVATE_KEY` via a secrets manager (e.g., AWS Secrets Manager, Vault, or Docker secrets) rather than a plain `.env` file.

---

## Step 8 - Validate the Setup

Start the server and expose the webhook endpoint:

```bash
# Start local dev server
uvicorn prism.api:app --reload --port 8000

# In another terminal, expose locally via ngrok
ngrok http 8000
```

Update the **Webhook URL** in the GitHub App settings to the ngrok HTTPS URL:
`https://<ngrok-id>.ngrok.io/webhook/github`

Then open a test pull request in the installed repository. You should see:

1. A `200 OK` delivery in GitHub App -> **Advanced -> Recent Deliveries**
2. A `202 Accepted` response body: `{"status": "accepted", "repo": "...", "pr": ..., "sha": "..."}`
3. A review comment posted by the app bot account on the PR (once the worker processes the job)

---

## Appendix: Required Permissions Summary

| Scope | Level | Reason |
| :--- | :--- | :--- |
| `contents` | read | Fetch file contents for additional review context |
| `metadata` | read | Required by all GitHub Apps (auto-granted) |
| `pull_requests` | write | Post reviews, inline comments, and dismiss old reviews |

---

## Appendix: Webhook Payload Verification

Prism verifies every webhook using HMAC-SHA256:

```
X-Hub-Signature-256: sha256=<HMAC-SHA256(GITHUB_WEBHOOK_SECRET, request_body)>
```

Invalid signatures return `401 Unauthorized`. This prevents spoofed webhook deliveries.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| :--- | :--- | :--- |
| `401` on webhook delivery | Mismatched `GITHUB_WEBHOOK_SECRET` | Regenerate and sync secret in both GitHub and `.env` |
| `422` on token exchange | Wrong `GITHUB_APP_ID` or stale/rotated key | Re-download private key and update `GITHUB_PRIVATE_KEY` |
| App posts no comment | PR is a draft, bot-authored, or untrusted fork | Intended - see filtering rules in `src/prism/github/webhook.py` |
| `Redis connection refused` | Redis not running | Start Redis: `docker compose up redis -d` |
| Worker not processing jobs | Arq worker not started | Run: `uv run arq prism.queue.worker.WorkerSettings` |
