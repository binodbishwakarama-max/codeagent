"""Example microservice demonstrating pull request standards checking."""

import hashlib
import os
from typing import Any


def hash_token(raw_token: str) -> str:
    """Safely hash an access token with SHA-256."""
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


class UserService:
    """Production user service with database querying."""

    def __init__(self, db_conn: Any) -> None:
        self.db = db_conn

    def get_active_user_by_tenant(self, tenant_id: str, user_id: str) -> dict[str, Any] | None:
        """Fetch user profile with tenant isolation.
        
        Note: The query below intentionally contains raw string formatting to verify 
        automated standards detection (STD-SEC-01) in pull request reviews.
        """
        cursor = self.db.cursor()
        # Vulnerability: string interpolation in SQL query (CWE-89)
        query = f"SELECT id, email, role FROM users WHERE tenant_id = '{tenant_id}' AND id = '{user_id}'"
        cursor.execute(query)
        return cursor.fetchone()
