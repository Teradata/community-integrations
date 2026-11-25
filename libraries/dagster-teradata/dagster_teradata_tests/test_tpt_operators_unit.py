"""Unit tests for the operator classes in dagster_teradata.ttu.tpt.

External helpers (TPT script building, SSH, file checks, tdload execution) are
patched at the tpt module, so nothing touches a database, host, or the home
directory.
"""

import os
import subprocess
from unittest import mock

import pytest
from dagster import DagsterError

from dagster_teradata.ttu import tpt
from dagster_teradata.ttu.tpt import DdlOperator, TdLoadOperator

MODULE = "dagster_teradata.ttu.tpt"


@pytest.fixture(autouse=True)
def _sandbox_home(tmp_path, monkeypatch):
    """SecureCredentialManager() reads/creates ~/.ssh files; keep them in tmp_path."""
    real_expanduser = os.path.expanduser
    targets = {
        "~/.ssh/bteq_cred_key": str(tmp_path / "ssh" / "bteq_cred_key"),
        "~/.ssh/bteq_credentials.json": str(tmp_path / "ssh" / "bteq_credentials.json"),
    }
    monkeypatch.setattr(
        os.path, "expanduser", lambda p: targets.get(p, real_expanduser(p))
    )


def _ddl(connection=None):
    return DdlOperator(
        connection or mock.MagicMock(), mock.MagicMock(), mock.MagicMock()
    )


def _tdload(target_resource=None):
    return TdLoadOperator(
        mock.MagicMock(), mock.MagicMock(), target_resource, mock.MagicMock()
    )


# ---------------------------------------------------------------------------
# DdlOperator: input validation and error-list normalization
# ---------------------------------------------------------------------------


class TestDdlOperatorValidation:
    @pytest.mark.parametrize(
        "ddl", [None, [], "CREATE TABLE t (i INT)", ["  "], [""], [None]]
    )
    def test_invalid_ddl_rejected(self, ddl):
        with pytest.raises(ValueError, match="DDL must be"):
            _ddl().ddl_operator(ddl=ddl)

    @pytest.mark.parametrize(
        "given, expected",
        [(None, []), (7, [7]), ([1, 2], [1, 2])],
    )
    def test_normalize_error_list(self, given, expected):
        assert _ddl()._normalize_error_list(given) == expected

    @pytest.mark.parametrize("bad", ["3807", [1, "x"], 1.5])
    def test_normalize_error_list_rejects_bad_types(self, bad):
        with pytest.raises(
            ValueError, match="error_list must be an int or list of ints"
        ):
            _ddl()._normalize_error_list(bad)


# ---------------------------------------------------------------------------
# DdlOperator: execution
# ---------------------------------------------------------------------------


class TestDdlOperatorExecution:
    def test_local_run_passes_script_and_default_dir(self):
        op = _ddl()
        with (
            mock.patch(
                f"{MODULE}.prepare_tpt_ddl_script", return_value="SCRIPT"
            ) as prep,
            mock.patch(f"{MODULE}.execute_ddl", return_value=0) as run,
        ):
            assert op.ddl_operator(ddl=["CREATE TABLE t (i INT)"], error_list=3807) == 0
        assert prep.call_args.kwargs["error_list"] == [3807]
        assert prep.call_args.kwargs["sql"] == ["CREATE TABLE t (i INT)"]
        run.assert_called_once_with(op, "SCRIPT", "/tmp")

    def test_execution_error_is_logged_and_reraised(self):
        op = _ddl()
        with (
            mock.patch(f"{MODULE}.prepare_tpt_ddl_script", return_value="S"),
            mock.patch(
                f"{MODULE}.execute_ddl", side_effect=DagsterError("tbuild boom")
            ),
        ):
            with pytest.raises(DagsterError, match="tbuild boom"):
                op.ddl_operator(ddl=["SELECT 1"])
        op.log.error.assert_called_once()

    def test_remote_host_sets_up_ssh_and_uses_temp_dir(self):
        op = _ddl()

        def fake_setup(**kwargs):
            op.ssh_client = mock.MagicMock()
            return True

        with (
            mock.patch.object(
                op, "_setup_ssh_connection", side_effect=fake_setup
            ) as setup,
            mock.patch(f"{MODULE}.prepare_tpt_ddl_script", return_value="S"),
            mock.patch(
                f"{MODULE}.get_remote_temp_directory", return_value="/rtmp"
            ) as tmpdir,
            mock.patch(f"{MODULE}.execute_ddl", return_value=0) as run,
        ):
            op.ddl_operator(
                ddl=["SELECT 1"],
                remote_host="h",
                remote_user="u",
                remote_password="p",
                remote_working_dir="",
            )
        assert setup.call_args.kwargs["host"] == "h"
        tmpdir.assert_called_once()
        assert run.call_args.args[2] == "/rtmp"

    def test_failed_ssh_setup_raises(self):
        op = _ddl()
        with (
            mock.patch.object(op, "_setup_ssh_connection", return_value=False),
            mock.patch(f"{MODULE}.prepare_tpt_ddl_script", return_value="S"),
        ):
            with pytest.raises(
                DagsterError, match="Failed to establish SSH connection"
            ):
                op.ddl_operator(ddl=["SELECT 1"], remote_host="h")


# ---------------------------------------------------------------------------
# SSH setup (shared behavior of both operators)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [_ddl, _tdload], ids=["ddl", "tdload"])
class TestSshSetup:
    def test_key_auth_uses_private_key(self, factory):
        op = factory()
        with (
            mock.patch(f"{MODULE}.paramiko.SSHClient") as client_cls,
            mock.patch(
                f"{MODULE}.paramiko.RSAKey.from_private_key_file", return_value="KEY"
            ) as load,
        ):
            assert op._setup_ssh_connection("h", "u", None, "/k/id_rsa", 2222) is True
        load.assert_called_once_with("/k/id_rsa")
        client_cls.return_value.connect.assert_called_once_with(
            "h", port=2222, username="u", pkey="KEY"
        )

    def test_password_auth_connects_with_password(self, factory):
        op = factory()
        with mock.patch(f"{MODULE}.paramiko.SSHClient") as client_cls:
            assert op._setup_ssh_connection("h", "u", "pw", None, 22) is True
        client_cls.return_value.connect.assert_called_once_with(
            "h", port=22, username="u", password="pw"
        )

    def test_uses_stored_credentials_when_no_password(self, factory):
        op = factory()
        stored = {"password": "ENC"}
        with (
            mock.patch(f"{MODULE}.paramiko.SSHClient") as client_cls,
            mock.patch(f"{MODULE}.get_stored_credentials", return_value=stored),
            mock.patch.object(
                op.cred_manager, "decrypt", return_value="from-store"
            ) as dec,
        ):
            op._setup_ssh_connection("h", "u", None, None, 22)
        dec.assert_called_once_with("ENC")
        client_cls.return_value.connect.assert_called_once_with(
            "h", port=22, username="u", password="from-store"
        )

    def test_no_password_and_no_stored_credentials_connects_with_none(self, factory):
        op = factory()
        with (
            mock.patch(f"{MODULE}.paramiko.SSHClient") as client_cls,
            mock.patch(f"{MODULE}.get_stored_credentials", return_value=None),
        ):
            op._setup_ssh_connection("h", "u", None, None, 22)
        assert client_cls.return_value.connect.call_args.kwargs["password"] is None

    def test_missing_username_without_password_is_rejected(self, factory):
        op = factory()
        with mock.patch(f"{MODULE}.paramiko.SSHClient"):
            with pytest.raises(DagsterError, match="Username is required"):
                op._setup_ssh_connection("h", None, None, None, 22)

    def test_connect_failure_is_wrapped(self, factory):
        op = factory()
        with mock.patch(f"{MODULE}.paramiko.SSHClient") as client_cls:
            client_cls.return_value.connect.side_effect = OSError("refused")
            with pytest.raises(DagsterError, match="SSH connection failed: refused"):
                op._setup_ssh_connection("h", "u", "pw", None, 22)


# ---------------------------------------------------------------------------
# on_kill (shared behavior)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [_ddl, _tdload], ids=["ddl", "tdload"])
class TestOnKill:
    def test_no_process_is_a_noop(self, factory):
        op = factory()
        op.connection = {}
        op.on_kill()
        op.log.warning.assert_not_called()

    def test_terminates_running_process(self, factory):
        op = factory()
        proc = mock.MagicMock()
        op.connection = {"sp": proc}
        op.on_kill()
        proc.terminate.assert_called_once()
        proc.wait.assert_called_once_with(timeout=5)
        proc.kill.assert_not_called()

    def test_kills_when_terminate_times_out(self, factory):
        op = factory()
        proc = mock.MagicMock()
        proc.wait.side_effect = subprocess.TimeoutExpired(cmd="tbuild", timeout=5)
        op.connection = {"sp": proc}
        op.on_kill()
        proc.kill.assert_called_once()
        op.log.warning.assert_called_once()

    def test_unexpected_error_is_logged(self, factory):
        op = factory()
        proc = mock.MagicMock()
        proc.terminate.side_effect = RuntimeError("weird")
        op.connection = {"sp": proc}
        op.on_kill()
        op.log.error.assert_called_once()


# ---------------------------------------------------------------------------
# TdLoadOperator: mode validation
# ---------------------------------------------------------------------------


class TestTdLoadModeValidation:
    def test_source_table_and_select_are_mutually_exclusive(self):
        op = _tdload()
        op.source_table, op.select_stmt = "a", "SELECT 1"
        with pytest.raises(ValueError, match="Cannot specify both"):
            op._validate_and_determine_mode()

    def test_insert_stmt_needs_target_table(self):
        op = _tdload()
        op.insert_stmt = "INSERT INTO t VALUES (1)"
        with pytest.raises(ValueError, match="insert_stmt requires target_table"):
            op._validate_and_determine_mode()

    def test_file_to_table_defaults_target_connection_to_source(self):
        src = mock.MagicMock()
        op = TdLoadOperator(mock.MagicMock(), src, None, mock.MagicMock())
        op.source_file_name, op.target_table = "in.csv", "t"
        assert op._validate_and_determine_mode() == "file_to_table"
        assert op.target_teradata_connection_resource is src

    def test_table_to_file(self):
        op = _tdload()
        op.source_table, op.target_file_name = "a", "out.csv"
        assert op._validate_and_determine_mode() == "table_to_file"

    def test_table_to_table_requires_target_connection(self):
        op = _tdload(target_resource=None)
        op.select_stmt, op.target_table = "SELECT 1", "t"
        with pytest.raises(ValueError, match="Target connection required"):
            op._validate_and_determine_mode()

    def test_table_to_table_with_target_connection(self):
        op = _tdload(target_resource=mock.MagicMock())
        op.source_table, op.target_table = "a", "t"
        assert op._validate_and_determine_mode() == "table_to_table"

    def test_job_var_file_mode(self):
        op = _tdload()
        op.tdload_job_var_file = "/j.txt"
        assert op._validate_and_determine_mode() == "job_var_file"

    def test_nothing_specified_is_invalid(self):
        with pytest.raises(ValueError, match="Invalid parameter combination"):
            _tdload()._validate_and_determine_mode()

    def test_prepare_job_var_content_delegates(self):
        op = _tdload()
        op.source_table, op.target_file_name = "a", "f"
        with mock.patch(
            f"{MODULE}.prepare_tdload_job_var_file", return_value="VARS"
        ) as prep:
            assert op._prepare_job_var_content("table_to_file") == "VARS"
        assert prep.call_args.kwargs["mode"] == "table_to_file"
        assert prep.call_args.kwargs["source_table"] == "a"


# ---------------------------------------------------------------------------
# TdLoadOperator: end-to-end dispatch (generated job vars)
# ---------------------------------------------------------------------------


class TestTdLoadGeneratedJobVars:
    def test_local_generated_job_vars_run_tdload(self):
        op = _tdload()
        with (
            mock.patch(f"{MODULE}.prepare_tdload_job_var_file", return_value="VARS"),
            mock.patch(f"{MODULE}.execute_tdload", return_value=0) as run,
        ):
            rc = op.tdload_operator(
                source_table="a", target_file_name="f.csv", tdload_options="-h"
            )
        assert rc == 0
        kwargs = run.call_args.kwargs
        assert kwargs["job_var_content"] == "VARS"
        assert kwargs["remote_working_dir"] == "/tmp"
        assert kwargs["tdload_options"] == "-h"

    def test_remote_generated_job_vars_use_ssh_client(self):
        op = _tdload()
        ssh = mock.MagicMock()

        def fake_setup(**kwargs):
            op.ssh_client = ssh
            return True

        with (
            mock.patch.object(op, "_setup_ssh_connection", side_effect=fake_setup),
            mock.patch(f"{MODULE}.prepare_tdload_job_var_file", return_value="VARS"),
            mock.patch(f"{MODULE}.get_remote_temp_directory", return_value="/rtmp"),
            mock.patch(f"{MODULE}.execute_tdload", return_value=0) as run,
        ):
            op.tdload_operator(
                source_table="a",
                target_file_name="f.csv",
                remote_host="h",
                remote_user="u",
                remote_password="p",
            )
        assert run.call_args.kwargs["ssh_client"] is ssh
        assert run.call_args.kwargs["remote_working_dir"] == "/rtmp"

    def test_failure_is_logged_and_reraised(self):
        op = _tdload()
        with (
            mock.patch(f"{MODULE}.prepare_tdload_job_var_file", return_value="V"),
            mock.patch(
                f"{MODULE}.execute_tdload", side_effect=DagsterError("tdload boom")
            ),
        ):
            with pytest.raises(DagsterError, match="tdload boom"):
                op.tdload_operator(source_table="a", target_file_name="f.csv")
        op.log.error.assert_called_once()

    def test_invalid_mode_raises_before_execution(self):
        with pytest.raises(ValueError, match="Invalid parameter combination"):
            _tdload().tdload_operator()


# ---------------------------------------------------------------------------
# TdLoadOperator: existing job-var-file paths
# ---------------------------------------------------------------------------


class TestTdLoadJobVarFiles:
    def test_local_invalid_file_path_rejected(self):
        op = _tdload()
        with mock.patch(f"{MODULE}.is_valid_file", return_value=False):
            with pytest.raises(ValueError, match="Invalid local job var file path"):
                op.tdload_operator(tdload_job_var_file="/missing.txt")

    def test_remote_invalid_job_var_file_rejected(self):
        op = _tdload()

        def fake_setup(**kwargs):
            op.ssh_client = mock.MagicMock()
            return True

        with (
            mock.patch.object(op, "_setup_ssh_connection", side_effect=fake_setup),
            mock.patch(f"{MODULE}.is_valid_remote_job_var_file", return_value=False),
        ):
            with pytest.raises(ValueError, match="Invalid remote job var file path"):
                op.tdload_operator(
                    tdload_job_var_file="/r/j.txt",
                    remote_host="h",
                    remote_user="u",
                    remote_password="p",
                    remote_working_dir="/r",
                )

    def test_handle_remote_requires_path(self):
        with pytest.raises(ValueError, match="Remote job var file path required"):
            _tdload()._handle_remote_job_var_file(mock.MagicMock(), None)

    def test_handle_local_requires_path(self):
        with pytest.raises(ValueError, match="Local job var file path required"):
            _tdload()._handle_local_job_var_file(None)

    def test_handle_local_rejects_invalid_file(self):
        with mock.patch(f"{MODULE}.is_valid_file", return_value=False):
            with pytest.raises(ValueError, match="Invalid local job var file"):
                _tdload()._handle_local_job_var_file("/bad.txt")

    @pytest.mark.xfail(
        strict=True,
        raises=ValueError,
        reason=(
            "Known bug: _handle_local_job_var_file branches on `if self.remote_host:` "
            "(inverted), so a local job variable file always fails with "
            "'Local execution not configured'."
        ),
    )
    def test_local_job_var_file_runs(self):
        op = _tdload()
        with (
            mock.patch(f"{MODULE}.is_valid_file", return_value=True),
            mock.patch(f"{MODULE}.read_file", return_value="VARS"),
            mock.patch(f"{MODULE}._execute_tdload_locally", return_value=0) as run,
        ):
            assert op.tdload_operator(tdload_job_var_file="/j.txt") == 0
        run.assert_called_once()

    @pytest.mark.xfail(
        strict=True,
        raises=TypeError,
        reason=(
            "Known bug: _handle_remote_job_var_file calls _execute_tdload_via_ssh with "
            "4 positional arguments, but it takes 6 (log, ssh_client, remote_working_dir, "
            "job_var_content, tdload_options, tdload_job_name)."
        ),
    )
    def test_remote_job_var_file_runs(self):
        op = _tdload()
        ssh = mock.MagicMock()
        sftp = ssh.open_sftp.return_value
        sftp.open.return_value.__enter__.return_value.read.return_value = b"VARS"

        def fake_setup(**kwargs):
            op.ssh_client = ssh
            return True

        # _execute_tdload_via_ssh is deliberately NOT patched so the bad call is exercised.
        with (
            mock.patch.object(op, "_setup_ssh_connection", side_effect=fake_setup),
            mock.patch(f"{MODULE}.is_valid_remote_job_var_file", return_value=True),
        ):
            assert (
                op.tdload_operator(
                    tdload_job_var_file="/r/j.txt",
                    remote_host="h",
                    remote_user="u",
                    remote_password="p",
                    remote_working_dir="/r",
                )
                == 0
            )


# ---------------------------------------------------------------------------
# Module sanity
# ---------------------------------------------------------------------------


def test_public_operator_names_exported():
    assert tpt.DdlOperator is DdlOperator
    assert tpt.TdLoadOperator is TdLoadOperator
