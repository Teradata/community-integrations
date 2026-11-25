"""Unit tests for the credential and key handling in
dagster_teradata.ttu.utils.encryption_utils.

Everything runs under pytest's tmp_path; ~/.ssh is never touched.
"""

import json
import os
from types import SimpleNamespace
from unittest import mock

import pytest
from cryptography.fernet import Fernet, InvalidToken
from dagster import DagsterError

from dagster_teradata.ttu.utils import encryption_utils as eu
from dagster_teradata.ttu.utils.encryption_utils import (
    SecureCredentialManager,
    decrypt_remote_file,
    decrypt_remote_file_to_string,
    get_stored_credentials,
    store_credentials,
)


@pytest.fixture
def key_file(tmp_path):
    return str(tmp_path / "keys" / "bteq_cred_key")


@pytest.fixture(autouse=True)
def cred_file(tmp_path, monkeypatch):
    """Sandbox every test: redirect ~/.ssh credential and key paths into tmp_path."""
    ssh_dir = tmp_path / "ssh"
    targets = {
        "~/.ssh/bteq_credentials.json": str(ssh_dir / "bteq_credentials.json"),
        "~/.ssh/bteq_cred_key": str(ssh_dir / "bteq_cred_key"),
    }
    real_expanduser = os.path.expanduser

    def fake_expanduser(path):
        return targets.get(path, real_expanduser(path))

    monkeypatch.setattr(os.path, "expanduser", fake_expanduser)
    return ssh_dir / "bteq_credentials.json"


def _owner(manager=None):
    return SimpleNamespace(
        log=mock.MagicMock(),
        cred_manager=manager or SecureCredentialManager(key_file=None),
    )


# ---------------------------------------------------------------------------
# SecureCredentialManager
# ---------------------------------------------------------------------------


class TestSecureCredentialManager:
    def test_generates_and_persists_key_when_missing(self, key_file):
        mgr = SecureCredentialManager(key_file=key_file)
        assert os.path.exists(key_file)
        with open(key_file, "rb") as fh:
            assert fh.read() == mgr.key
        Fernet(mgr.key)  # raises if not a valid Fernet key

    def test_reuses_existing_key(self, key_file):
        os.makedirs(os.path.dirname(key_file), exist_ok=True)
        existing = Fernet.generate_key()
        with open(key_file, "wb") as fh:
            fh.write(existing)
        assert SecureCredentialManager(key_file=key_file).key == existing

    def test_encrypt_decrypt_roundtrip(self, key_file):
        mgr = SecureCredentialManager(key_file=key_file)
        token = mgr.encrypt("s3cret!")
        assert token != "s3cret!"
        assert mgr.decrypt(token) == "s3cret!"

    def test_tokens_from_other_key_are_rejected(self, tmp_path):
        a = SecureCredentialManager(key_file=str(tmp_path / "a.key"))
        b = SecureCredentialManager(key_file=str(tmp_path / "b.key"))
        with pytest.raises(InvalidToken):
            b.decrypt(a.encrypt("data"))

    def test_encrypt_without_key_raises(self, key_file):
        mgr = SecureCredentialManager(key_file=key_file)
        mgr.key = None
        with pytest.raises(ValueError, match="not initialized"):
            mgr.encrypt("x")

    def test_decrypt_without_key_raises(self, key_file):
        mgr = SecureCredentialManager(key_file=key_file)
        mgr.key = None
        with pytest.raises(ValueError, match="not initialized"):
            mgr.decrypt("x")

    def test_unreadable_key_path_is_wrapped(self, tmp_path):
        # A directory where the key file should be cannot be read as a file.
        key_dir_as_file = tmp_path / "is_a_dir"
        key_dir_as_file.mkdir()
        with pytest.raises(DagsterError, match="Encryption key handling failed"):
            SecureCredentialManager(key_file=str(key_dir_as_file))


# ---------------------------------------------------------------------------
# generate_random_password
# ---------------------------------------------------------------------------


def test_random_password_default_length_is_twelve():
    assert len(eu.generate_random_password()) == 12


def test_random_passwords_differ():
    assert eu.generate_random_password(32) != eu.generate_random_password(32)


# ---------------------------------------------------------------------------
# decrypt_remote_file / decrypt_remote_file_to_string (SSH)
# ---------------------------------------------------------------------------


def _ssh_with_exit(exit_status, out=b"", err=b""):
    ssh = mock.MagicMock()
    stdout = mock.MagicMock()
    stdout.channel.recv_exit_status.return_value = exit_status
    stdout.read.return_value = out
    stderr = mock.MagicMock()
    stderr.read.return_value = err
    ssh.exec_command.return_value = (mock.MagicMock(), stdout, stderr)
    return ssh


class TestRemoteDecrypt:
    def test_decrypt_remote_file_success_quotes_password(self):
        ssh = _ssh_with_exit(0)
        logger = mock.MagicMock()
        rc = decrypt_remote_file(ssh, "/r/a.enc", "/r/a.sql", "it's", logger)
        assert rc == 0
        cmd = ssh.exec_command.call_args.args[0]
        assert "-pass pass:'it'\\''s'" in cmd
        assert "-in /r/a.enc -out /r/a.sql" in cmd
        logger.info.assert_called_once()

    def test_decrypt_remote_file_failure_raises(self):
        ssh = _ssh_with_exit(1)
        with pytest.raises(DagsterError, match="Decryption failed with exit status 1"):
            decrypt_remote_file(ssh, "/r/a.enc", "/r/a.sql", "pw", mock.MagicMock())

    @pytest.mark.xfail(
        strict=True,
        raises=AttributeError,
        reason=(
            "Known bug: decrypt_remote_file defaults logger to None but calls "
            "logger.info unconditionally on success."
        ),
    )
    def test_decrypt_remote_file_success_without_logger(self):
        ssh = _ssh_with_exit(0)
        assert decrypt_remote_file(ssh, "/r/a.enc", "/r/a.sql", "pw") == 0

    def test_decrypt_to_string_returns_status_and_streams(self):
        ssh = _ssh_with_exit(0, out=b"SELECT 1;\n", err=b"")
        rc, out, err = decrypt_remote_file_to_string(ssh, "/r/a.enc", "pw", "bteq")
        assert (rc, out, err) == (0, "SELECT 1;\n", "")
        assert ssh.exec_command.call_args.args[0].endswith("| bteq")

    def test_decrypt_to_string_propagates_nonzero_status(self):
        ssh = _ssh_with_exit(2, out=b"", err=b"bad key")
        rc, _, err = decrypt_remote_file_to_string(ssh, "/r/a.enc", "pw", "bteq")
        assert rc == 2
        assert err == "bad key"


# ---------------------------------------------------------------------------
# store_credentials / get_stored_credentials
# ---------------------------------------------------------------------------


class TestStoredCredentials:
    def test_store_then_read_roundtrip(self, cred_file):
        owner = _owner()
        assert store_credentials(owner, "db.example.com", "alice", "hunter2") is True
        entry = get_stored_credentials(owner, "db.example.com", "alice")
        assert entry is not None
        assert entry["username"] == "alice"
        assert entry["password"] != "hunter2"
        assert owner.cred_manager.decrypt(entry["password"]) == "hunter2"
        assert "timestamp" in entry

    def test_multiple_hosts_are_kept(self, cred_file):
        owner = _owner()
        store_credentials(owner, "h1", "u", "p1")
        store_credentials(owner, "h2", "u", "p2")
        data = json.loads(cred_file.read_text())
        assert set(data) == {"u@h1", "u@h2"}

    def test_overwrites_existing_entry_for_same_host_and_user(self, cred_file):
        owner = _owner()
        store_credentials(owner, "h", "u", "old")
        store_credentials(owner, "h", "u", "new")
        entry = get_stored_credentials(owner, "h", "u")
        assert entry is not None
        assert owner.cred_manager.decrypt(entry["password"]) == "new"

    def test_corrupt_file_is_logged_and_replaced(self, cred_file):
        cred_file.parent.mkdir(parents=True, exist_ok=True)
        cred_file.write_text("{not json")
        owner = _owner()
        assert store_credentials(owner, "h", "u", "p") is True
        owner.log.error.assert_called()
        assert "u@h" in json.loads(cred_file.read_text())

    def test_write_failure_returns_false(self, cred_file, monkeypatch):
        owner = _owner()

        def boom(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", boom)
        assert store_credentials(owner, "h", "u", "p") is False
        owner.log.error.assert_called_once()

    def test_get_returns_none_when_no_file(self, cred_file):
        assert get_stored_credentials(_owner(), "h", "u") is None

    def test_get_returns_none_for_unknown_key(self, cred_file):
        owner = _owner()
        store_credentials(owner, "h1", "u", "p")
        assert get_stored_credentials(owner, "other", "u") is None

    def test_get_corrupt_file_logs_warning_and_returns_none(self, cred_file):
        cred_file.parent.mkdir(parents=True, exist_ok=True)
        cred_file.write_text("{oops")
        owner = _owner()
        assert get_stored_credentials(owner, "h", "u") is None
        owner.log.warning.assert_called_once()
