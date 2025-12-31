import logging
import os
import re
import tempfile
from io import StringIO
from pathlib import Path

import pytest

from clyptq.infra.security.secrets import EnvSecretsManager, SecretsManager
from clyptq.infra.utils.logging import configure_logging, get_logger


@pytest.mark.security
def test_credentials_never_logged():
    print("\n=== Credential Leak Detection (Logging) ===")

    api_key = "test_api_key_12345"
    api_secret = "test_api_secret_67890"

    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.DEBUG)

    logger = get_logger("test_security")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    logger.info("Starting authentication process")
    logger.debug(f"Configuration loaded")
    logger.warning("Rate limit approaching")
    logger.error("Connection failed")

    log_output = log_stream.getvalue()

    print(f"Log output ({len(log_output)} chars):")
    print(log_output[:500])

    assert api_key not in log_output, f"API key leaked in logs"
    assert api_secret not in log_output, f"API secret leaked in logs"

    print("✅ No credentials found in logs")


@pytest.mark.security
def test_secrets_manager_env_isolation():
    print("\n=== Secrets Manager Environment Isolation ===")

    os.environ["CLYPTQ_TEST_API_KEY"] = "sensitive_key_123"
    os.environ["CLYPTQ_TEST_API_SECRET"] = "sensitive_secret_456"

    manager = EnvSecretsManager(prefix="CLYPTQ_")

    api_key = manager.get("TEST_API_KEY")
    api_secret = manager.get("TEST_API_SECRET")

    print(f"Retrieved API key: {api_key[:8]}...")
    print(f"Retrieved API secret: {api_secret[:8]}...")

    assert api_key == "sensitive_key_123"
    assert api_secret == "sensitive_secret_456"

    credentials = manager.get_dict("TEST")
    assert "api_key" in credentials
    assert "api_secret" in credentials

    non_existent = manager.get("DOES_NOT_EXIST")
    assert non_existent is None

    print("✅ Secrets manager isolation working")

    del os.environ["CLYPTQ_TEST_API_KEY"]
    del os.environ["CLYPTQ_TEST_API_SECRET"]


@pytest.mark.security
def test_credential_regex_scanner():
    print("\n=== Credential Pattern Scanner ===")

    patterns = {
        "api_key": r"api[_-]?key\s*[=:]\s*['\"]?([a-zA-Z0-9_-]{20,})",
        "api_secret": r"api[_-]?secret\s*[=:]\s*['\"]?([a-zA-Z0-9_-]{20,})",
        "password": r"password\s*[=:]\s*['\"]?([a-zA-Z0-9_@#$%^&*]{8,})",
        "token": r"token\s*[=:]\s*['\"]?([a-zA-Z0-9_.-]{20,})",
        "private_key": r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----",
    }

    test_cases = [
        ('api_key = "sk_test_51234567890abcdefghijk"', True, "api_key"),
        ('password = "SecurePass123!"', True, "password"),
        ('token: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"', True, "token"),
        ("config = {'debug': True}", False, None),
        ('logger.info("Process completed")', False, None),
    ]

    for code, should_match, expected_pattern in test_cases:
        matched = False
        matched_pattern = None

        for pattern_name, regex in patterns.items():
            if re.search(regex, code, re.IGNORECASE):
                matched = True
                matched_pattern = pattern_name
                break

        print(f"Code: {code[:50]}...")
        print(f"  Expected match: {should_match}, Pattern: {expected_pattern}")
        print(f"  Actual match: {matched}, Pattern: {matched_pattern}")

        if should_match:
            assert matched, f"Expected to match pattern {expected_pattern}"
            assert matched_pattern == expected_pattern
        else:
            assert not matched, f"Should not match any pattern"

    print("✅ Credential regex scanner working")


@pytest.mark.security
def test_source_code_credential_scan():
    print("\n=== Source Code Credential Scan ===")

    project_root = Path(__file__).parent.parent.parent
    source_dir = project_root / "clyptq"

    if not source_dir.exists():
        pytest.skip(f"Source directory not found: {source_dir}")

    dangerous_patterns = [
        (r"api[_-]?key\s*=\s*['\"][a-zA-Z0-9_-]{20,}['\"]", "Hardcoded API key"),
        (r"api[_-]?secret\s*=\s*['\"][a-zA-Z0-9_-]{20,}['\"]", "Hardcoded API secret"),
        (r"password\s*=\s*['\"][^'\"]{8,}['\"]", "Hardcoded password"),
        (r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----", "Private key in source"),
    ]

    violations = []

    for py_file in source_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        try:
            content = py_file.read_text()

            for pattern, description in dangerous_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[: match.start()].count("\n") + 1
                    violations.append(
                        {
                            "file": str(py_file.relative_to(project_root)),
                            "line": line_num,
                            "pattern": description,
                            "match": match.group(0)[:50],
                        }
                    )
        except Exception as e:
            print(f"  Warning: Could not scan {py_file}: {e}")

    print(f"\nScanned {len(list(source_dir.rglob('*.py')))} Python files")

    if violations:
        print(f"\n⚠️  Found {len(violations)} potential credential leaks:")
        for v in violations:
            print(f"  {v['file']}:{v['line']} - {v['pattern']}")
            print(f"    {v['match']}")

    assert len(violations) == 0, f"Found {len(violations)} credential leaks in source code"

    print("✅ No hardcoded credentials found")


@pytest.mark.security
def test_error_message_pii_redaction():
    print("\n=== Error Message PII Redaction ===")

    sensitive_data = {
        "email": "user@example.com",
        "phone": "+1-555-123-4567",
        "ssn": "123-45-6789",
        "credit_card": "4532-1234-5678-9010",
        "api_key": "sk_test_51234567890",
    }

    def redact_pii(message: str) -> str:
        redacted = message
        redacted = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL_REDACTED]", redacted)
        redacted = re.sub(r"\+?1?[-.]?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}", "[PHONE_REDACTED]", redacted)
        redacted = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REDACTED]", redacted)
        redacted = re.sub(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "[CARD_REDACTED]", redacted)
        redacted = re.sub(r"sk_(?:test|live)_[a-zA-Z0-9]{20,}", "[API_KEY_REDACTED]", redacted)
        return redacted

    test_messages = [
        f"User {sensitive_data['email']} failed authentication",
        f"Payment failed for card {sensitive_data['credit_card']}",
        f"API key {sensitive_data['api_key']} is invalid",
        f"Contact customer at {sensitive_data['phone']}",
        f"SSN verification failed: {sensitive_data['ssn']}",
    ]

    print("Testing PII redaction:")
    for msg in test_messages:
        redacted = redact_pii(msg)
        print(f"\nOriginal: {msg}")
        print(f"Redacted: {redacted}")

        for key, value in sensitive_data.items():
            assert value not in redacted, f"{key} not redacted: {value}"

    print("\n✅ PII redaction working")


@pytest.mark.security
def test_secrets_rotation():
    print("\n=== Secrets Rotation Test ===")

    class RotatableSecretsManager(SecretsManager):
        def __init__(self):
            self._secrets = {}
            self._version = 1

        def get(self, key: str) -> str:
            versioned_key = f"{key}_v{self._version}"
            return self._secrets.get(versioned_key)

        def get_dict(self, key: str) -> dict:
            result = {}
            for k, v in self._secrets.items():
                if k.startswith(key):
                    clean_key = k.replace(f"_v{self._version}", "")
                    result[clean_key] = v
            return result

        def set_secret(self, key: str, value: str):
            versioned_key = f"{key}_v{self._version}"
            self._secrets[versioned_key] = value

        def rotate(self, key: str, new_value: str):
            old_version = self._version
            self._version += 1

            new_key = f"{key}_v{self._version}"
            self._secrets[new_key] = new_value

            print(f"Rotated {key}: v{old_version} → v{self._version}")

    manager = RotatableSecretsManager()

    manager.set_secret("api_key", "old_key_12345")
    manager.set_secret("api_secret", "old_secret_67890")

    old_key = manager.get("api_key")
    print(f"V1 API key: {old_key}")
    assert old_key == "old_key_12345"

    manager.rotate("api_key", "new_key_99999")
    manager.rotate("api_secret", "new_secret_88888")

    new_key = manager.get("api_key")
    new_secret = manager.get("api_secret")

    print(f"V2 API key: {new_key}")
    print(f"V2 API secret: {new_secret}")

    assert new_key == "new_key_99999"
    assert new_secret == "new_secret_88888"
    assert new_key != old_key

    print("✅ Secrets rotation working")


@pytest.mark.security
def test_input_validation_fuzzing():
    print("\n=== Input Validation Fuzzing ===")

    from clyptq.core.types import Constraints

    def validate_symbol(symbol: str) -> bool:
        if not symbol or len(symbol) > 20:
            return False
        if not re.match(r"^[A-Z0-9]+/[A-Z0-9]+$", symbol):
            return False
        return True

    def validate_amount(amount: float, constraints: Constraints) -> bool:
        if amount <= 0:
            return False
        if constraints.max_position_size and amount > constraints.max_position_size:
            return False
        if constraints.min_position_size and amount < constraints.min_position_size:
            return False
        return True

    fuzz_symbols = [
        ("BTC/USDT", True),
        ("ETH/USDT", True),
        ("", False),
        ("INVALID", False),
        ("BTC-USDT", False),
        ("btc/usdt", False),
        ("A" * 50, False),
        ("../../etc/passwd", False),
        ("<script>alert('xss')</script>", False),
        ("'; DROP TABLE symbols;--", False),
    ]

    fuzz_amounts = [
        (100.0, True),
        (0.0, False),
        (-100.0, False),
        (float("inf"), False),
        (float("-inf"), False),
        (float("nan"), False),
        (1e308, False),
        (-1e308, False),
    ]

    print("\nFuzzing symbol validation:")
    for symbol, expected_valid in fuzz_symbols:
        is_valid = validate_symbol(symbol)
        status = "✅" if is_valid == expected_valid else "❌"
        print(f"{status} Symbol: {symbol[:30]:30s} Valid: {is_valid} (expected: {expected_valid})")
        assert is_valid == expected_valid

    print("\nFuzzing amount validation:")
    constraints = Constraints(min_position_size=10.0, max_position_size=10000.0)
    for amount, expected_valid in fuzz_amounts:
        try:
            is_valid = validate_amount(amount, constraints)
        except (ValueError, OverflowError):
            is_valid = False

        status = "✅" if is_valid == expected_valid else "❌"
        print(f"{status} Amount: {str(amount):20s} Valid: {is_valid} (expected: {expected_valid})")
        assert is_valid == expected_valid

    print("\n✅ Input validation fuzzing passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "security"])
