"""Bootstrap a test session for browser testing.

Run inside the container:
  docker exec causaltraceai-proxy-1 python /app/proxy/get_test_token.py
"""
import sys
import os

# Must match what the container was started with
os.environ["ACCESS_SIGNING_SECRET"] = "test-secret-for-local-dev-only"
os.environ["ACCESS_STORE"] = "memory"
os.environ["APP_URL"] = "http://localhost:8080"

sys.path.insert(0, "/app")
from proxy import access  # noqa: E402

email = "test@test.com"

# Approve user (record was created by the POST /auth/login call from the host)
record = access.set_status(email, "approved")
if record is None:
    print("ERROR: no record found — did you POST /auth/login first?")
    sys.exit(1)

print(f"STATUS: {record.get('status')}")

# Issue nonce and build magic link
nonce = access.issue_login_nonce(email)
link = access.login_link(email, nonce)
print(f"LOGIN_LINK={link}")
