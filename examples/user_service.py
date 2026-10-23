"""Example microservice demonstrating pull request standards checking."""

import hashlib
from typing import Any


def hash_token(raw_token: str) -> str:
    """Safely hash an access token with SHA-256."""
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


class UserService:
    """Production user service with database querying."""

    def __init__(self, db_conn: Any) -> None:
        self.db = db_conn

    def get_active_user_by_tenant(self, tenant_id: str, user_id: str) -> dict[str, Any] | None:
        """Fetch user profile with tenant isolation securely."""
        cursor = self.db.cursor()
        # Secure: parameterized query protects against SQL injection (STD-PY-SQL-01)
        query = "SELECT id, email, role FROM users WHERE tenant_id = %s AND id = %s"
        cursor.execute(query, (tenant_id, user_id))
        return cursor.fetchone()
