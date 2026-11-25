"""Unit tests for dagster_teradata.ttu.tpt_executer.

These call the executor helpers directly with a fake ``self`` and mock every
external dependency (subprocess, SSH, file transfer), so they run without a
Teradata host or TTU binaries.
"""

import io
import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
from dagster import DagsterError
from paramiko.ssh_exception import SSHException

from dagster_teradata.ttu import tpt_executer as ex

MODULE = "dagster_teradata.ttu.tpt_executer"


def _self(ssh_client=None):
    return SimpleNamespace(log=mock.MagicMock(), ssh_client=ssh_client)


def _fake_process(lines=(), returncode=0):
    proc = mock.MagicMock()
    proc.stdout = io.BytesIO("".join(f"{line}\n" for line in lines).encode("utf-8"))
    proc.returncode = returncode
    proc.wait.return_value = returncode
    return proc


@pytest.fixture(autouse=True)
def _no_real_local_side_effects():
    """These helpers shell out (icacls, cipher, etc.); keep them out of unit tests."""
    with (
        mock.patch(f"{MODULE}.set_local_file_permissions"),
        mock.patch(f"{MODULE}.terminate_subprocess"),
        mock.patch(f"{MODULE}.secure_delete"),
    ):
        yield


# ---------------------------------------------------------------------------
# execute_ddl
# ---------------------------------------------------------------------------


class TestExecuteDdl:
    def test_empty_script_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            ex.execute_ddl(_self(), "", "/remote")

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            ex.execute_ddl(_self(), [], "/remote")

    def test_whitespace_only_script_raises(self):
        with pytest.raises(ValueError, match="content cannot be empty"):
            ex.execute_ddl(_self(), "   \n  ", "/remote")

    def test_list_is_joined_and_run_locally(self):
        s = _self(ssh_client=None)
        with mock.patch(f"{MODULE}._execute_tbuild_locally", return_value=0) as local:
            assert (
                ex.execute_ddl(s, ["CREATE TABLE a (i INT);", "DROP TABLE a;"], "/r")
                == 0
            )
        local.assert_called_once_with(s, "CREATE TABLE a (i INT);\nDROP TABLE a;")

    def test_ssh_client_routes_to_remote_execution(self):
        s = _self(ssh_client=mock.MagicMock())
        with mock.patch(f"{MODULE}._execute_tbuild_via_ssh", return_value=7) as remote:
            assert ex.execute_ddl(s, "SELECT 1;", "/remote") == 7
        remote.assert_called_once_with(s, "SELECT 1;", "/remote")


# ---------------------------------------------------------------------------
# _execute_tbuild_locally
# ---------------------------------------------------------------------------


class TestExecuteTbuildLocally:
    @pytest.fixture(autouse=True)
    def _posix_setsid(self, monkeypatch):
        # tbuild is launched with preexec_fn=os.setsid, which only exists on POSIX.
        # The Windows-specific failure is covered by the xfail test at the bottom.
        monkeypatch.setattr(os, "setsid", lambda: None, raising=False)

    def test_missing_binary_raises(self):
        with mock.patch(f"{MODULE}.shutil.which", return_value=None):
            with pytest.raises(DagsterError, match="tbuild binary not found"):
                ex._execute_tbuild_locally(_self(), "SELECT 1;")

    def test_success_returns_zero_and_logs_output(self):
        s = _self()
        proc = _fake_process(["starting job", "done"], returncode=0)
        with (
            mock.patch(f"{MODULE}.shutil.which", return_value="/bin/tbuild"),
            mock.patch(f"{MODULE}.subprocess.Popen", return_value=proc) as popen,
            mock.patch(f"{MODULE}.terminate_subprocess") as term,
        ):
            assert ex._execute_tbuild_locally(s, "SELECT 1;") == 0
        cmd = popen.call_args.args[0]
        assert cmd[0:2] == ["tbuild", "-f"]
        assert cmd[2].endswith(".sql")
        assert cmd[3].startswith("tbuild_job_")
        s.log.info.assert_any_call("done")
        term.assert_called_once()

    def test_nonzero_exit_collects_error_lines(self):
        proc = _fake_process(["ok line", "Error: table missing", "other"], returncode=3)
        with (
            mock.patch(f"{MODULE}.shutil.which", return_value="/bin/tbuild"),
            mock.patch(f"{MODULE}.subprocess.Popen", return_value=proc),
            mock.patch(f"{MODULE}.terminate_subprocess"),
        ):
            with pytest.raises(
                DagsterError, match="tbuild failed with code 3: Error: table missing"
            ):
                ex._execute_tbuild_locally(_self(), "SELECT 1;")

    def test_nonzero_exit_without_error_lines_reports_unknown(self):
        proc = _fake_process(["just output"], returncode=2)
        with (
            mock.patch(f"{MODULE}.shutil.which", return_value="/bin/tbuild"),
            mock.patch(f"{MODULE}.subprocess.Popen", return_value=proc),
            mock.patch(f"{MODULE}.terminate_subprocess"),
        ):
            with pytest.raises(DagsterError, match="Unknown error"):
                ex._execute_tbuild_locally(_self(), "SELECT 1;")

    def test_popen_failure_is_wrapped(self):
        with (
            mock.patch(f"{MODULE}.shutil.which", return_value="/bin/tbuild"),
            mock.patch(f"{MODULE}.subprocess.Popen", side_effect=OSError("boom")),
            mock.patch(f"{MODULE}.terminate_subprocess"),
        ):
            with pytest.raises(DagsterError, match="tbuild execution failed: boom"):
                ex._execute_tbuild_locally(_self(), "SELECT 1;")


# ---------------------------------------------------------------------------
# _execute_tbuild_via_ssh
# ---------------------------------------------------------------------------


REMOTE_HELPERS = [
    "verify_tpt_utility_on_remote_host",
    "generate_random_password",
    "generate_encrypted_file_with_openssl",
    "transfer_file_sftp",
    "decrypt_remote_file",
    "set_remote_file_permissions",
    "execute_remote_command",
    "remote_secure_delete",
    "secure_delete",
]


def _ssh_patches(exit_status=0, error=""):
    """Patch every remote helper used by the tbuild SSH path (not yet started)."""
    patches = {n: mock.patch(f"{MODULE}.{n}") for n in REMOTE_HELPERS}
    return patches


def _start_all(patches):
    mocks = {n: p.start() for n, p in patches.items()}
    mocks["generate_random_password"].return_value = "pw"
    mocks["execute_remote_command"].return_value = (0, "out", "")
    return mocks


class TestExecuteTbuildViaSsh:
    def test_success_runs_full_remote_sequence(self):
        s = _self(ssh_client=mock.MagicMock())
        patches = _ssh_patches()
        mocks = _start_all(patches)
        try:
            rc = ex._execute_tbuild_via_ssh(s, "SELECT 1;", "/remote")
        finally:
            for p in patches.values():
                p.stop()
        assert rc == 0
        mocks["verify_tpt_utility_on_remote_host"].assert_called_once()
        mocks["transfer_file_sftp"].assert_called_once()
        mocks["decrypt_remote_file"].assert_called_once()
        mocks["set_remote_file_permissions"].assert_called_once()
        mocks["remote_secure_delete"].assert_called_once()
        remote_cmd = mocks["execute_remote_command"].call_args.args[1]
        assert remote_cmd.startswith("tbuild -f ")
        assert "tbuild_script_" in remote_cmd
        # "tbuild -f <script> <job>" -> 4 whitespace-separated tokens
        assert len(remote_cmd.split()) == 4

    def test_nonzero_remote_exit_raises_after_cleanup(self):
        s = _self(ssh_client=mock.MagicMock())
        patches = _ssh_patches()
        mocks = _start_all(patches)
        mocks["execute_remote_command"].return_value = (4, "", "bad script")
        try:
            with pytest.raises(
                DagsterError, match="tbuild failed with code 4: bad script"
            ):
                ex._execute_tbuild_via_ssh(s, "SELECT 1;", "/remote")
        finally:
            for p in patches.values():
                p.stop()
        mocks["remote_secure_delete"].assert_called_once()

    def test_socket_error_maps_to_timeout(self):
        s = _self(ssh_client=mock.MagicMock())
        patches = _ssh_patches()
        mocks = _start_all(patches)
        mocks["transfer_file_sftp"].side_effect = OSError("network down")
        try:
            with pytest.raises(DagsterError, match="SSH connection timeout"):
                ex._execute_tbuild_via_ssh(s, "SELECT 1;", "/remote")
        finally:
            for p in patches.values():
                p.stop()

    def test_ssh_exception_is_wrapped(self):
        s = _self(ssh_client=mock.MagicMock())
        patches = _ssh_patches()
        mocks = _start_all(patches)
        mocks["decrypt_remote_file"].side_effect = SSHException("auth failed")
        try:
            with pytest.raises(DagsterError, match="SSH error: auth failed"):
                ex._execute_tbuild_via_ssh(s, "SELECT 1;", "/remote")
        finally:
            for p in patches.values():
                p.stop()

    def test_generic_error_is_wrapped(self):
        s = _self(ssh_client=mock.MagicMock())
        patches = _ssh_patches()
        mocks = _start_all(patches)
        mocks["verify_tpt_utility_on_remote_host"].side_effect = RuntimeError(
            "no tbuild"
        )
        try:
            with pytest.raises(
                DagsterError, match="Remote tbuild execution failed: no tbuild"
            ):
                ex._execute_tbuild_via_ssh(s, "SELECT 1;", "/remote")
        finally:
            for p in patches.values():
                p.stop()


# ---------------------------------------------------------------------------
# execute_tdload dispatch and local execution
# ---------------------------------------------------------------------------


class TestExecuteTdload:
    def test_ssh_dispatch(self):
        log = mock.MagicMock()
        with mock.patch(f"{MODULE}._execute_tdload_via_ssh", return_value=0) as remote:
            rc = ex.execute_tdload(
                _self(),
                log,
                "/remote",
                ssh_client=mock.MagicMock(),
                job_var_content="x",
                tdload_options="-h",
                tdload_job_name="job1",
            )
        assert rc == 0
        assert remote.call_args.kwargs["tdload_job_name"] == "job1"

    def test_local_dispatch_generates_job_name_when_missing(self):
        log = mock.MagicMock()
        with mock.patch(f"{MODULE}._execute_tdload_locally", return_value=0) as local:
            ex.execute_tdload(_self(), log, "/remote", job_var_content="x")
        name = local.call_args.kwargs["tdload_job_name"]
        assert name.startswith("tdload_job_")

    def test_local_missing_binary_raises(self):
        with mock.patch(f"{MODULE}.shutil.which", return_value=None):
            with pytest.raises(DagsterError, match="tdload binary not found"):
                ex._execute_tdload_locally(mock.MagicMock(), "x", None, "job")

    def test_local_success(self):
        proc = _fake_process(["loaded 10 rows"], returncode=0)
        with (
            mock.patch(f"{MODULE}.shutil.which", return_value="/bin/tdload"),
            mock.patch(f"{MODULE}.subprocess.Popen", return_value=proc),
            mock.patch(f"{MODULE}.terminate_subprocess"),
        ):
            assert ex._execute_tdload_locally(mock.MagicMock(), "x", "-h", "job") == 0

    def test_local_failure_collects_errors(self):
        proc = _fake_process(["Error: bad layout"], returncode=8)
        with (
            mock.patch(f"{MODULE}.shutil.which", return_value="/bin/tdload"),
            mock.patch(f"{MODULE}.subprocess.Popen", return_value=proc),
            mock.patch(f"{MODULE}.terminate_subprocess"),
        ):
            with pytest.raises(
                DagsterError, match="tdload failed with code 8: Error: bad layout"
            ):
                ex._execute_tdload_locally(mock.MagicMock(), "x", None, None)

    def test_local_popen_failure_is_wrapped(self):
        with (
            mock.patch(f"{MODULE}.shutil.which", return_value="/bin/tdload"),
            mock.patch(f"{MODULE}.subprocess.Popen", side_effect=OSError("no exec")),
            mock.patch(f"{MODULE}.terminate_subprocess"),
        ):
            with pytest.raises(DagsterError, match="tdload execution failed: no exec"):
                ex._execute_tdload_locally(mock.MagicMock(), "x", None, None)

    def test_ssh_path_without_client_raises(self):
        with pytest.raises(DagsterError, match="SSH connection not established"):
            ex._transfer_to_and_execute_tdload_on_remote(
                mock.MagicMock(), None, "/tmp/x.txt", "/remote", None, None
            )

    def test_ssh_path_success(self):
        log = mock.MagicMock()
        names = [
            "verify_tpt_utility_on_remote_host",
            "generate_random_password",
            "generate_encrypted_file_with_openssl",
            "transfer_file_sftp",
            "decrypt_remote_file",
            "set_remote_file_permissions",
            "execute_remote_command",
            "remote_secure_delete",
            "secure_delete",
        ]
        patches = {n: mock.patch(f"{MODULE}.{n}") for n in names}
        mocks = {n: p.start() for n, p in patches.items()}
        mocks["generate_random_password"].return_value = "pw"
        mocks["execute_remote_command"].return_value = (0, "ok", "")
        try:
            rc = ex._transfer_to_and_execute_tdload_on_remote(
                log, mock.MagicMock(), "/local/job.txt", "/remote", "-h", "job1"
            )
        finally:
            for p in patches.values():
                p.stop()
        assert rc == 0
        remote_cmd = mocks["execute_remote_command"].call_args.args[1]
        assert remote_cmd.startswith("tdload -j ")
        assert remote_cmd.endswith(" -h job1")

    def test_ssh_path_nonzero_exit_raises(self):
        names = [
            "verify_tpt_utility_on_remote_host",
            "generate_random_password",
            "generate_encrypted_file_with_openssl",
            "transfer_file_sftp",
            "decrypt_remote_file",
            "set_remote_file_permissions",
            "execute_remote_command",
            "remote_secure_delete",
            "secure_delete",
        ]
        patches = {n: mock.patch(f"{MODULE}.{n}") for n in names}
        mocks = {n: p.start() for n, p in patches.items()}
        mocks["execute_remote_command"].return_value = (5, "", "fatal")
        try:
            with pytest.raises(DagsterError, match="tdload failed with code 5: fatal"):
                ex._transfer_to_and_execute_tdload_on_remote(
                    mock.MagicMock(),
                    mock.MagicMock(),
                    "/local/job.txt",
                    "/remote",
                    None,
                    None,
                )
        finally:
            for p in patches.values():
                p.stop()


# ---------------------------------------------------------------------------
# _build_tdload_command
# ---------------------------------------------------------------------------


class TestBuildTdloadCommand:
    def test_minimal_command(self):
        assert ex._build_tdload_command(mock.MagicMock(), "/j.txt", None, None) == [
            "tdload",
            "-j",
            "/j.txt",
        ]

    def test_options_are_shell_split_and_job_name_appended(self):
        cmd = ex._build_tdload_command(
            mock.MagicMock(), "/j.txt", '-h "my host"', "job1"
        )
        assert cmd == ["tdload", "-j", "/j.txt", "-h", "my host", "job1"]

    def test_unbalanced_quote_falls_back_to_simple_split(self):
        log = mock.MagicMock()
        cmd = ex._build_tdload_command(log, "/j.txt", '-h "unclosed', None)
        assert cmd == ["tdload", "-j", "/j.txt", "-h", '"unclosed']
        log.warning.assert_called_once()


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def test_on_kill_logs_cleanup():
    s = _self()
    ex.on_kill(s)
    s.log.info.assert_called_once_with("TPT Hook cleanup initiated")


def test_preferred_temp_directory_creates_and_removes_dir():
    with ex.preferred_temp_directory(prefix="unit_") as tmp:
        assert os.path.isdir(tmp)
        assert os.path.basename(tmp).startswith("unit_")
    assert not os.path.exists(tmp)


@pytest.mark.xfail(
    sys.platform == "win32",
    strict=True,
    reason=(
        "Known bug on Windows: _execute_tbuild_locally passes preexec_fn=os.setsid, "
        "which does not exist on Windows, so every local tbuild run fails."
    ),
)
def test_local_tbuild_succeeds_without_setsid_monkeypatch():
    proc = _fake_process([], returncode=0)
    with (
        mock.patch(f"{MODULE}.shutil.which", return_value="/bin/tbuild"),
        mock.patch(f"{MODULE}.subprocess.Popen", return_value=proc),
    ):
        assert ex._execute_tbuild_locally(_self(), "SELECT 1;") == 0


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason=(
        "Known bug: get_dagster_home_dir is declared with a `self` parameter but is "
        "called without arguments from preferred_temp_directory's fallback branch."
    ),
)
def test_preferred_temp_directory_fallback_when_temp_unusable():
    with mock.patch(
        f"{MODULE}.tempfile.gettempdir", return_value="/definitely/not/a/dir"
    ):
        with ex.preferred_temp_directory() as tmp:
            assert os.path.isdir(tmp)
