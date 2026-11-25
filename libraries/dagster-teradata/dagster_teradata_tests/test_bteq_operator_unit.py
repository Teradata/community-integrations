"""Unit tests for dagster_teradata.ttu.bteq.Bteq.

Subprocesses, SSH, SFTP, encryption and BTEQ-script helpers are patched at the
bteq module, so no BTEQ binary, host, or home directory is needed.
"""

import os
import socket
import subprocess
import sys
import tempfile
from unittest import mock

import pytest
from dagster import DagsterError
from paramiko.ssh_exception import SSHException

from dagster_teradata.ttu.bteq import Bteq

MODULE = "dagster_teradata.ttu.bteq"


@pytest.fixture(autouse=True)
def _sandbox_home(tmp_path, monkeypatch):
    """Bteq() builds SecureCredentialManager() and stored-credential paths under ~/.ssh."""
    real_expanduser = os.path.expanduser
    targets = {
        "~/.ssh/bteq_cred_key": str(tmp_path / "ssh" / "bteq_cred_key"),
        "~/.ssh/bteq_credentials.json": str(tmp_path / "ssh" / "bteq_credentials.json"),
    }
    monkeypatch.setattr(
        os.path, "expanduser", lambda p: targets.get(p, real_expanduser(p))
    )


def _bteq():
    return Bteq(mock.MagicMock(), mock.MagicMock(), mock.MagicMock())


# ---------------------------------------------------------------------------
# bteq_operator: local dispatch
# ---------------------------------------------------------------------------


class TestBteqOperatorLocal:
    def test_sql_runs_local_script(self):
        op = _bteq()
        with (
            mock.patch(
                f"{MODULE}.prepare_bteq_script_for_local_execution",
                return_value="SCRIPT",
            ),
            mock.patch.object(op, "execute_bteq_script", return_value=0) as run,
        ):
            assert op.bteq_operator(sql="SELECT 1;") == 0
        assert run.call_args.args[0] == "SCRIPT"

    def test_invalid_file_path_rejected(self):
        op = _bteq()
        with mock.patch(f"{MODULE}.is_valid_file", return_value=False):
            with pytest.raises(ValueError, match="is invalid or does not exist"):
                op.bteq_operator(file_path="/missing.sql")

    def test_encoding_mismatch_names_configured_encoding(self):
        op = _bteq()
        err = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad byte")
        with (
            mock.patch(f"{MODULE}.is_valid_file", return_value=True),
            mock.patch(f"{MODULE}.is_valid_encoding", side_effect=err),
        ):
            with pytest.raises(ValueError, match="specified BTEQ I/O encoding 'utf-8'"):
                op.bteq_operator(file_path="/f.sql", bteq_script_encoding="utf-8")

    def test_encoding_mismatch_without_script_encoding_uses_default_message(self):
        op = _bteq()
        err = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad byte")
        with (
            mock.patch(f"{MODULE}.is_valid_file", return_value=True),
            mock.patch(f"{MODULE}.is_valid_encoding", side_effect=err),
        ):
            with pytest.raises(ValueError, match="BTEQ I/O encoding i.e.'UTF-8'"):
                op.bteq_operator(file_path="/f.sql", bteq_script_encoding=None)

    def test_valid_file_is_handed_to_local_file_handler(self):
        op = _bteq()
        with (
            mock.patch(f"{MODULE}.is_valid_file", return_value=True),
            mock.patch(f"{MODULE}.is_valid_encoding"),
            mock.patch.object(op, "_handle_local_bteq_file", return_value=5) as handler,
        ):
            assert op.bteq_operator(file_path="/f.sql") == 5
        handler.assert_called_once_with(file_path="/f.sql")

    def test_no_sql_and_no_file_returns_none(self):
        assert _bteq().bteq_operator() is None


# ---------------------------------------------------------------------------
# bteq_operator: remote dispatch
# ---------------------------------------------------------------------------


class TestBteqOperatorRemote:
    def test_sql_runs_remote_script(self):
        op = _bteq()
        with (
            mock.patch(
                f"{MODULE}.prepare_bteq_script_for_remote_execution", return_value="S"
            ),
            mock.patch.object(op, "execute_bteq_script", return_value=3) as run,
        ):
            assert (
                op.bteq_operator(sql="SELECT 1;", remote_host="h", remote_user="u") == 3
            )
        assert run.call_args.args[0] == "S"

    def test_remote_file_uses_ssh_and_remote_handler(self):
        op = _bteq()

        def fake_setup(**kwargs):
            op.ssh_client = mock.MagicMock()
            return True

        with (
            mock.patch.object(op, "_setup_ssh_connection", side_effect=fake_setup),
            mock.patch(f"{MODULE}.is_valid_remote_bteq_script_file", return_value=True),
            mock.patch.object(
                op, "_handle_remote_bteq_file", return_value=0
            ) as handler,
        ):
            assert (
                op.bteq_operator(file_path="/r/f.sql", remote_host="h", remote_user="u")
                == 0
            )
        handler.assert_called_once()

    def test_failed_ssh_setup_raises(self):
        op = _bteq()
        with mock.patch.object(op, "_setup_ssh_connection", return_value=False):
            with pytest.raises(
                DagsterError, match="Failed to establish SSH connection"
            ):
                op.bteq_operator(file_path="/r/f.sql", remote_host="h", remote_user="u")

    def test_missing_remote_file_rejected(self):
        op = _bteq()

        def fake_setup(**kwargs):
            op.ssh_client = mock.MagicMock()
            return True

        with (
            mock.patch.object(op, "_setup_ssh_connection", side_effect=fake_setup),
            mock.patch(
                f"{MODULE}.is_valid_remote_bteq_script_file", return_value=False
            ),
        ):
            with pytest.raises(
                ValueError, match="invalid or file does not exist on remote machine"
            ):
                op.bteq_operator(file_path="/r/f.sql", remote_host="h", remote_user="u")

    def test_remote_without_sql_or_file_rejected(self):
        op = _bteq()
        with pytest.raises(ValueError, match="Both are missing"):
            op.bteq_operator(remote_host="h", remote_user="u")


# ---------------------------------------------------------------------------
# execute_bteq_script dispatch and remote temp-file staging
# ---------------------------------------------------------------------------


class TestExecuteDispatch:
    def test_remote_host_goes_to_remote_executor(self):
        op = _bteq()
        with (
            mock.patch.object(
                op, "execute_bteq_script_at_remote", return_value=1
            ) as remote,
            mock.patch.object(op, "execute_bteq_script_at_local") as local,
        ):
            rc = op.execute_bteq_script(
                "S", "/tmp", "utf-8", 60, None, "ASCII", 0, "UTF-8", remote_host="h"
            )
        assert rc == 1
        remote.assert_called_once()
        local.assert_not_called()

    def test_no_remote_host_goes_to_local_executor(self):
        op = _bteq()
        with (
            mock.patch.object(
                op, "execute_bteq_script_at_local", return_value=0
            ) as local,
            mock.patch.object(op, "execute_bteq_script_at_remote") as remote,
        ):
            assert (
                op.execute_bteq_script(
                    "S", "/tmp", "utf-8", 60, None, "ASCII", 0, "UTF-8"
                )
                == 0
            )
        local.assert_called_once()
        remote.assert_not_called()

    def test_remote_executor_stages_script_in_temp_file(self):
        op = _bteq()
        seen = {}

        def capture(file_path, *args, **kwargs):
            with open(file_path, encoding="utf-8") as fh:
                seen["content"] = fh.read()
            seen["name"] = os.path.basename(file_path)
            return 0

        with mock.patch.object(
            op, "_transfer_to_and_execute_bteq_on_remote", side_effect=capture
        ):
            rc = op.execute_bteq_script_at_remote(
                "SELECT 42;",
                "/r",
                "utf-8",
                60,
                None,
                "ASCII",
                0,
                "UTF-8",
                "h",
                "u",
                "pw",
                None,
                22,
            )
        assert rc == 0
        assert seen == {"content": "SELECT 42;", "name": "bteq_script.txt"}


# ---------------------------------------------------------------------------
# _transfer_to_and_execute_bteq_on_remote
# ---------------------------------------------------------------------------


def _remote_patches(op, exit_status=0, stdout="", stderr=""):
    """Start every remote helper; returns (patchers, mocks) for cleanup.

    _setup_ssh_connection succeeds and installs a fake ssh_client by default.
    """
    patchers = {
        n: mock.patch(f"{MODULE}.{n}")
        for n in (
            "verify_bteq_installed_remote",
            "generate_random_password",
            "generate_encrypted_file_with_openssl",
            "transfer_file_sftp",
            "prepare_bteq_command_for_remote_execution",
            "decrypt_remote_file_to_string",
        )
    }
    patchers["_setup_ssh_connection"] = mock.patch.object(op, "_setup_ssh_connection")
    mocks = {n: p.start() for n, p in patchers.items()}
    mocks["generate_random_password"].return_value = "pw"
    mocks["prepare_bteq_command_for_remote_execution"].return_value = "CMD"
    mocks["decrypt_remote_file_to_string"].return_value = (exit_status, stdout, stderr)

    def set_client(**kwargs):
        op.ssh_client = mock.MagicMock()
        return True

    mocks["_setup_ssh_connection"].side_effect = set_client
    return patchers, mocks


def _stop(patchers):
    for p in patchers.values():
        p.stop()


def _transfer(op, **overrides):
    # Use a fresh temp directory so the executor's real local cleanup can only touch
    # files this test created, never anything at a fixed path on the machine.
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = dict(
            file_path=os.path.join(tmp_dir, "bteq_script.txt"),
            remote_working_dir="/r",
            bteq_script_encoding="utf-8",
            timeout=60,
            timeout_rc=None,
            bteq_quit_rc=0,
            bteq_session_encoding="ASCII",
            tmp_dir=tmp_dir,
            remote_host="h",
            remote_user="u",
            remote_password="p",
            ssh_key_path=None,
            remote_port=22,
        )
        args.update(overrides)
        return op._transfer_to_and_execute_bteq_on_remote(**args)


class TestRemoteTransfer:
    def test_success_returns_exit_and_removes_remote_file(self):
        op = _bteq()
        patchers, mocks = _remote_patches(op, exit_status=0, stdout="OK")
        try:
            assert _transfer(op) == 0
            mocks["decrypt_remote_file_to_string"].assert_called_once()
            cleanup = op.ssh_client.exec_command.call_args.args[0]
            assert cleanup.startswith("rm -f ")
        finally:
            _stop(patchers)

    def test_failure_text_with_nonzero_exit_raises(self):
        op = _bteq()
        patchers, _ = _remote_patches(
            op, exit_status=8, stderr="Failure 3807 table missing"
        )
        try:
            with pytest.raises(
                DagsterError, match="BTEQ task failed with error: Failure 3807"
            ):
                _transfer(op)
        finally:
            _stop(patchers)

    def test_allowed_return_code_is_warned_not_raised(self):
        op = _bteq()
        patchers, _ = _remote_patches(op, exit_status=4, stderr="Error: soft")
        try:
            assert _transfer(op, bteq_quit_rc=[0, 4]) == 4
            op.log.warning.assert_called_once()
        finally:
            _stop(patchers)

    def test_error_text_with_zero_exit_is_warned(self):
        op = _bteq()
        patchers, _ = _remote_patches(op, exit_status=0, stderr="Error: noisy but ok")
        try:
            assert _transfer(op) == 0
            op.log.warning.assert_called_once()
        finally:
            _stop(patchers)

    def test_ssh_setup_failure_raises(self):
        op = _bteq()
        patchers, mocks = _remote_patches(op)
        mocks["_setup_ssh_connection"].side_effect = None
        mocks["_setup_ssh_connection"].return_value = False
        try:
            with pytest.raises(
                DagsterError, match="Failed to establish SSH connection"
            ):
                _transfer(op)
        finally:
            _stop(patchers)

    def test_missing_ssh_client_raises(self):
        op = _bteq()
        patchers, mocks = _remote_patches(op)
        mocks["_setup_ssh_connection"].side_effect = None
        mocks["_setup_ssh_connection"].return_value = True  # ssh_client stays None
        try:
            with pytest.raises(DagsterError, match="ssh_client` is None"):
                _transfer(op)
        finally:
            _stop(patchers)

    @pytest.mark.parametrize(
        "exc, expected",
        [
            (OSError("net"), "SSH connection timed out"),
            (socket.gaierror("dns"), "SSH connection timed out"),
            (
                SSHException("auth"),
                "unexpected error occurred during SSH connection: auth",
            ),
            (
                RuntimeError("boom"),
                "unexpected error occurred while executing BTEQ script",
            ),
        ],
    )
    def test_exceptions_are_mapped(self, exc, expected):
        op = _bteq()
        patchers, mocks = _remote_patches(op)
        mocks["verify_bteq_installed_remote"].side_effect = exc
        try:
            with pytest.raises(DagsterError, match=expected):
                _transfer(op)
        finally:
            _stop(patchers)


# ---------------------------------------------------------------------------
# execute_bteq_script_at_local (subprocess)
# ---------------------------------------------------------------------------


@pytest.fixture
def posix_setsid(monkeypatch):
    # The local BTEQ path passes preexec_fn=os.setsid (POSIX-only); see xfail below.
    monkeypatch.setattr(os, "setsid", lambda: None, raising=False)


def _proc(stdout=b"", returncode=0, wait_side_effect=None):
    proc = mock.MagicMock()
    proc.communicate.return_value = (stdout, None)
    proc.returncode = returncode
    proc.wait.side_effect = wait_side_effect
    return proc


def _run_local(op, proc, **overrides):
    args = dict(
        bteq_script="SELECT 1;",
        bteq_script_encoding="utf-8",
        timeout=60,
        timeout_rc=None,
        bteq_quit_rc=0,
        bteq_session_encoding="ASCII",
        temp_file_read_encoding="UTF-8",
    )
    args.update(overrides)
    with (
        mock.patch(f"{MODULE}.verify_bteq_installed"),
        mock.patch(
            f"{MODULE}.prepare_bteq_command_for_local_execution", return_value="bteq"
        ),
        mock.patch(f"{MODULE}.subprocess.Popen", return_value=proc) as popen,
    ):
        rc = op.execute_bteq_script_at_local(**args)
    return rc, popen


@pytest.mark.usefixtures("posix_setsid")
class TestLocalExecution:
    def test_success_returns_process_returncode(self):
        op = _bteq()
        rc, popen = _run_local(op, _proc(stdout=b"Query OK\n", returncode=0))
        assert rc == 0
        assert popen.call_args.kwargs["shell"] is True
        assert popen.call_args.args[0] == "bteq"

    def test_script_is_sent_on_stdin(self):
        op = _bteq()
        proc = _proc(stdout=b"")
        _run_local(op, proc, bteq_script="SELECT 'é';")
        proc.communicate.assert_called_once_with(input="SELECT 'é';".encode("UTF-8"))

    def test_failure_line_with_nonzero_rc_raises(self):
        op = _bteq()
        proc = _proc(stdout=b"Error: 3807 table not found\n", returncode=8)
        with pytest.raises(
            DagsterError, match="BTEQ task failed with error: Error: 3807"
        ):
            _run_local(op, proc)

    def test_allowed_rc_with_failure_line_is_warned(self):
        op = _bteq()
        proc = _proc(stdout=b"Failure: soft\n", returncode=4)
        rc, _ = _run_local(op, proc, bteq_quit_rc=[0, 4])
        assert rc == 4
        op.log.warning.assert_called_once()

    def test_undecodable_line_is_warned_and_skipped(self):
        op = _bteq()
        proc = _proc(stdout=b"\xff\xfe junk\n", returncode=0)
        rc, _ = _run_local(op, proc)
        assert rc == 0
        op.log.warning.assert_any_call("Failed to decode line: %s", b"\xff\xfe junk")

    def test_timeout_kills_and_raises(self):
        op = _bteq()
        proc = _proc(stdout=b"", wait_side_effect=subprocess.TimeoutExpired("bteq", 60))
        with mock.patch.object(op, "on_kill") as kill:
            with pytest.raises(DagsterError, match="timed out after 60 seconds"):
                _run_local(op, proc)
        kill.assert_called_once()

    def test_missing_stdout_raises(self):
        op = _bteq()
        proc = _proc(stdout=None)
        with pytest.raises(DagsterError, match="Process stdout is None"):
            _run_local(op, proc)


@pytest.mark.xfail(
    sys.platform == "win32",
    strict=True,
    raises=AttributeError,
    reason=(
        "Known bug on Windows: execute_bteq_script_at_local passes preexec_fn=os.setsid, "
        "which does not exist on Windows, so local BTEQ runs cannot start there."
    ),
)
def test_local_bteq_starts_without_setsid_monkeypatch():
    op = _bteq()
    rc, _ = _run_local(op, _proc(stdout=b"", returncode=0))
    assert rc == 0


# ---------------------------------------------------------------------------
# File handlers
# ---------------------------------------------------------------------------


class TestFileHandlers:
    def test_local_file_handler_returns_none_for_invalid_file(self):
        assert _bteq()._handle_local_bteq_file("/nope.sql") is None

    def test_local_file_handler_executes_script(self):
        op = _bteq()
        with (
            mock.patch(f"{MODULE}.is_valid_file", return_value=True),
            mock.patch(f"{MODULE}.read_file", return_value="SELECT 1;"),
            mock.patch(
                f"{MODULE}.prepare_bteq_script_for_local_execution", return_value="S"
            ),
            mock.patch.object(op, "execute_bteq_script", return_value=0) as run,
        ):
            assert op._handle_local_bteq_file("/f.sql") == 0
        assert run.call_args.args[0] == "S"

    def test_remote_file_handler_requires_path(self):
        with pytest.raises(ValueError, match="valid file path"):
            _bteq()._handle_remote_bteq_file(mock.MagicMock(), None)

    def test_remote_file_handler_reads_and_executes(self):
        op = _bteq()
        ssh = mock.MagicMock()
        sftp = ssh.open_sftp.return_value
        sftp.open.return_value.__enter__.return_value.read.return_value = b"SELECT 1;"
        with (
            mock.patch(
                f"{MODULE}.prepare_bteq_script_for_remote_execution", return_value="S"
            ),
            mock.patch.object(
                op, "execute_bteq_script_at_remote", return_value=0
            ) as run,
        ):
            assert op._handle_remote_bteq_file(ssh, "/r/f.sql") == 0
        sftp.close.assert_called_once()
        assert run.call_args.args[0] == "S"
        assert run.call_args.kwargs["remote_port"] == 22


# ---------------------------------------------------------------------------
# on_kill and temp directory fallback
# ---------------------------------------------------------------------------


class TestOnKillAndTempDir:
    def test_on_kill_terminates(self):
        op = _bteq()
        proc = mock.MagicMock()
        op.connection = {"sp": proc}
        op.on_kill()
        proc.terminate.assert_called_once()

    def test_on_kill_forces_kill_on_timeout(self):
        op = _bteq()
        proc = mock.MagicMock()
        proc.wait.side_effect = subprocess.TimeoutExpired("bteq", 5)
        op.connection = {"sp": proc}
        op.on_kill()
        proc.kill.assert_called_once()

    def test_on_kill_logs_unexpected_error(self):
        op = _bteq()
        proc = mock.MagicMock()
        proc.terminate.side_effect = RuntimeError("x")
        op.connection = {"sp": proc}
        op.on_kill()
        op.log.error.assert_called_once()

    def test_preferred_temp_directory_normal_path(self):
        op = _bteq()
        with op.preferred_temp_directory(prefix="t_") as tmp:
            assert os.path.isdir(tmp)
            assert os.path.basename(tmp).startswith("t_")
        assert not os.path.exists(tmp)

    def test_get_dagster_home_dir_honours_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DAGSTER_HOME", str(tmp_path))
        assert _bteq().get_dagster_home_dir() == str(tmp_path)

    @pytest.mark.xfail(
        strict=True,
        raises=FileNotFoundError,
        reason=(
            "Known bug: the temp-dir fallback uses the literal '~/.dagster_home' without "
            "expanding '~' and without creating it, so TemporaryDirectory cannot use it."
        ),
    )
    def test_temp_fallback_works_with_default_home(self, monkeypatch):
        monkeypatch.delenv("DAGSTER_HOME", raising=False)
        op = _bteq()
        with mock.patch(f"{MODULE}.tempfile.gettempdir", return_value="/no/such/dir"):
            with op.preferred_temp_directory() as tmp:
                assert os.path.isdir(tmp)


# ---------------------------------------------------------------------------
# SSH setup
# ---------------------------------------------------------------------------


class TestBteqSshSetup:
    def test_key_auth(self):
        op = _bteq()
        with (
            mock.patch(f"{MODULE}.paramiko.SSHClient") as client_cls,
            mock.patch(
                f"{MODULE}.paramiko.RSAKey.from_private_key_file", return_value="KEY"
            ),
        ):
            assert op._setup_ssh_connection("h", "u", None, "/k", 2200) is True
        client_cls.return_value.connect.assert_called_once_with(
            "h", port=2200, username="u", pkey="KEY"
        )

    def test_password_auth(self):
        op = _bteq()
        with mock.patch(f"{MODULE}.paramiko.SSHClient") as client_cls:
            op._setup_ssh_connection("h", "u", "pw", None, 22)
        client_cls.return_value.connect.assert_called_once_with(
            "h", port=22, username="u", password="pw"
        )

    def test_stored_credentials_used_when_no_password(self):
        op = _bteq()
        with (
            mock.patch(f"{MODULE}.paramiko.SSHClient") as client_cls,
            mock.patch(
                f"{MODULE}.get_stored_credentials", return_value={"password": "ENC"}
            ),
            mock.patch.object(op.cred_manager, "decrypt", return_value="stored"),
        ):
            op._setup_ssh_connection("h", "u", None, None, 22)
        assert client_cls.return_value.connect.call_args.kwargs["password"] == "stored"

    def test_no_credentials_connects_with_none(self):
        op = _bteq()
        with (
            mock.patch(f"{MODULE}.paramiko.SSHClient") as client_cls,
            mock.patch(f"{MODULE}.get_stored_credentials", return_value=None),
        ):
            op._setup_ssh_connection("h", "u", None, None, 22)
        assert client_cls.return_value.connect.call_args.kwargs["password"] is None

    def test_missing_user_rejected(self):
        op = _bteq()
        with mock.patch(f"{MODULE}.paramiko.SSHClient"):
            with pytest.raises(DagsterError, match="Username is required"):
                op._setup_ssh_connection("h", None, None, None, 22)

    def test_connect_failure_wrapped(self):
        op = _bteq()
        with mock.patch(f"{MODULE}.paramiko.SSHClient") as client_cls:
            client_cls.return_value.connect.side_effect = OSError("refused")
            with pytest.raises(DagsterError, match="SSH connection failed: refused"):
                op._setup_ssh_connection("h", "u", "pw", None, 22)
