"""Unit tests for dagster_teradata.teradata_compute_cluster_manager.

The Teradata connection is a MagicMock whose ``execute_query`` returns scripted
values in order, and the polling sync is patched where it matters, so these
tests need no database and never sleep.
"""

from unittest import mock

import pytest
import teradatasql
from dagster import DagsterError

from dagster_teradata import constants
from dagster_teradata.teradata_compute_cluster_manager import (
    CcDbStatus,
    CcOperation,
    TeradataComputeClusterManager,
    TeradataComputeClusterSync,
)

MODULE = "dagster_teradata.teradata_compute_cluster_manager"
LAKE_VERSION = "20.00.03.10"
OLD_VERSION = "16.20.53.06"


_DDL_PREFIXES = ("CREATE", "DROP", "RESUME", "SUSPEND")


def _manager(execute_results=()):
    """Scripted connection: SELECT results come from the queue (exceptions are raised);
    DDL/operation statements return None without consuming a scripted value."""
    queue = list(execute_results)
    connection = mock.MagicMock()

    def execute_query(sql, *args, **kwargs):
        if sql.lstrip().upper().startswith(_DDL_PREFIXES):
            return None
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    connection.execute_query.side_effect = execute_query
    log = mock.MagicMock()
    return TeradataComputeClusterManager(connection, log), connection, log


def _sql_calls(connection):
    return [c.args[0] for c in connection.execute_query.call_args_list]


def _sync_patch():
    """Patch the polling sync so create/resume/suspend return a canned status."""
    sync = mock.MagicMock()
    sync.run.return_value = "done"
    return mock.patch(f"{MODULE}.TeradataComputeClusterSync", return_value=sync), sync


# ---------------------------------------------------------------------------
# verify_compute_cluster
# ---------------------------------------------------------------------------


class TestVerifyComputeCluster:
    @pytest.mark.parametrize("name", [None, "None", ""])
    def test_invalid_profile_name_rejected(self, name):
        mgr, connection, _ = _manager()
        with pytest.raises(
            DagsterError, match=constants.CC_OPR_EMPTY_PROFILE_ERROR_MSG
        ):
            mgr.verify_compute_cluster(name)
        connection.execute_query.assert_not_called()

    def test_lake_version_with_lake_storage_passes(self):
        mgr, connection, _ = _manager([LAKE_VERSION, 1])
        assert mgr.verify_compute_cluster("prof") is None
        assert len(_sql_calls(connection)) == 2

    def test_pre_lake_version_rejected(self):
        mgr, _, _ = _manager([OLD_VERSION])
        with pytest.raises(DagsterError, match=constants.CC_GRP_LAKE_SUPPORT_ONLY_MSG):
            mgr.verify_compute_cluster("prof")

    def test_missing_version_raises_generic_error(self):
        mgr, _, _ = _manager([None])
        with pytest.raises(
            Exception, match="Error occurred while getting teradata database version"
        ):
            mgr.verify_compute_cluster("prof")

    def test_database_error_on_version_lookup_is_wrapped(self):
        mgr, _, log = _manager([teradatasql.DatabaseError("nope")])
        with pytest.raises(
            Exception, match="Error occurred while getting teradata database version"
        ):
            mgr.verify_compute_cluster("prof")
        log.error.assert_called_once()

    def test_no_lake_storage_rejected(self):
        mgr, _, _ = _manager([LAKE_VERSION, None])
        with pytest.raises(DagsterError, match=constants.CC_GRP_LAKE_SUPPORT_ONLY_MSG):
            mgr.verify_compute_cluster("prof")


# ---------------------------------------------------------------------------
# get_initially_suspended / handle_cc_status
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_defaults_to_false_without_clause(self):
        mgr, _, _ = _manager()
        assert mgr.get_initially_suspended("CREATE COMPUTE PROFILE p") == "FALSE"

    @pytest.mark.parametrize(
        "clause, expected",
        [
            ("INITIALLY_SUSPENDED('TRUE')", "TRUE"),
            ("initially_suspended ( 'false' )", "FALSE"),
            ("INITIALLY_SUSPENDED('true')", "TRUE"),
        ],
    )
    def test_reads_clause_case_insensitively(self, clause, expected):
        mgr, _, _ = _manager()
        assert (
            mgr.get_initially_suspended(f"CREATE COMPUTE PROFILE p, {clause}")
            == expected
        )

    def test_handle_cc_status_runs_sql_and_returns_sync_status(self):
        mgr, connection, log = _manager([None])
        patcher, sync = _sync_patch()
        with patcher as cls:
            status = mgr.handle_cc_status(CcOperation.RESUME, "RESUME X", "p", "g", 30)
        assert status == "done"
        assert _sql_calls(connection) == ["RESUME X"]
        args = cls.call_args.args
        assert args[2] is CcOperation.RESUME
        assert args[3:6] == ("p", "g", constants.CC_POLL_INTERVAL)
        assert args[6] == 30
        log.info.assert_called_with("done")


# ---------------------------------------------------------------------------
# create_teradata_compute_cluster
# ---------------------------------------------------------------------------


class TestCreateComputeCluster:
    def test_existing_profile_returns_its_status(self):
        # verify (version, lake), group count, profile status
        mgr, connection, _ = _manager([LAKE_VERSION, 1, 1, "Running"])
        assert mgr.create_teradata_compute_cluster("p", "g") == "Running"
        assert not any(sql.startswith("CREATE") for sql in _sql_calls(connection))

    def test_creates_missing_group_then_profile(self):
        # verify, group count 0, profile status None
        mgr, connection, _ = _manager([LAKE_VERSION, 1, 0, None])
        patcher, sync = _sync_patch()
        with patcher as cls:
            result = mgr.create_teradata_compute_cluster(
                "p",
                "g",
                query_strategy="STANDARD",
                compute_map="m",
                compute_attribute="attr",
            )
        assert result == "done"
        sqls = _sql_calls(connection)
        assert "CREATE COMPUTE GROUP g USING QUERY_STRATEGY ('STANDARD')" in sqls
        assert (
            "CREATE COMPUTE PROFILE p IN g, INSTANCE = m, INSTANCE TYPE = STANDARD USING attr"
            in sqls
        )
        assert cls.call_args.args[2] is CcOperation.CREATE

    def test_initially_suspended_uses_create_suspend(self):
        mgr, connection, _ = _manager([LAKE_VERSION, 1, 1, None])
        patcher, _ = _sync_patch()
        with patcher as cls:
            mgr.create_teradata_compute_cluster(
                "p", "g", compute_attribute="INITIALLY_SUSPENDED('TRUE')"
            )
        assert cls.call_args.args[2] is CcOperation.CREATE_SUSPEND

    def test_no_group_skips_group_queries(self):
        mgr, connection, _ = _manager([LAKE_VERSION, 1, None])
        patcher, _ = _sync_patch()
        with patcher:
            mgr.create_teradata_compute_cluster("p", None, query_strategy=None)
        sqls = _sql_calls(connection)
        assert not any("ComputeGroup" in s and "DBC.ComputeGroups" in s for s in sqls)
        assert "CREATE COMPUTE PROFILE p" in sqls

    def test_group_count_none_is_treated_as_missing(self):
        mgr, connection, _ = _manager([LAKE_VERSION, 1, None, None])
        patcher, _ = _sync_patch()
        with patcher:
            mgr.create_teradata_compute_cluster("p", "g")
        assert any(
            s.startswith("CREATE COMPUTE GROUP g") for s in _sql_calls(connection)
        )


# ---------------------------------------------------------------------------
# drop_teradata_compute_cluster
# ---------------------------------------------------------------------------


class TestDropComputeCluster:
    def test_drops_profile_only(self):
        mgr, connection, _ = _manager([LAKE_VERSION, 1, None])
        mgr.drop_teradata_compute_cluster("p", "g")
        assert _sql_calls(connection)[-1] == "DROP COMPUTE PROFILE p IN COMPUTE GROUP g"

    def test_drops_profile_and_group(self):
        mgr, connection, log = _manager([LAKE_VERSION, 1, None, None])
        mgr.drop_teradata_compute_cluster("p", "g", delete_compute_group=True)
        sqls = _sql_calls(connection)
        assert sqls[-2:] == [
            "DROP COMPUTE PROFILE p IN COMPUTE GROUP g",
            "DROP COMPUTE GROUP g",
        ]
        assert log.info.call_count == 2


# ---------------------------------------------------------------------------
# resume / suspend
# ---------------------------------------------------------------------------


class TestResumeSuspend:
    def test_resume_missing_cluster_raises(self):
        mgr, _, _ = _manager([LAKE_VERSION, 1, None])
        with pytest.raises(DagsterError, match=constants.CC_GRP_PRP_NON_EXISTS_MSG):
            mgr.resume_teradata_compute_cluster("p", "g")

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Known bug: resume compares the status string against the CcDbStatus enum "
            "(`str != Enum` is always True), so a running cluster is never detected."
        ),
    )
    def test_resume_already_running_is_noop(self):
        mgr, connection, log = _manager([LAKE_VERSION, 1, CcDbStatus.RUNNING.value])
        patcher, sync = _sync_patch()
        with patcher as cls:
            assert mgr.resume_teradata_compute_cluster("p", "g") is None
        cls.assert_not_called()
        log.info.assert_called_with(
            "Compute Cluster %s already %s", "p", CcDbStatus.RUNNING
        )

    def test_resume_suspended_runs_resume_operation(self):
        mgr, connection, _ = _manager(
            [LAKE_VERSION, 1, CcDbStatus.SUSPENDED.value, None]
        )
        patcher, _ = _sync_patch()
        with patcher as cls:
            assert mgr.resume_teradata_compute_cluster("p", "g", timeout=45) == "done"
        assert (
            _sql_calls(connection)[-1]
            == "RESUME COMPUTE FOR COMPUTE PROFILE p IN COMPUTE GROUP g"
        )
        assert cls.call_args.args[2] is CcOperation.RESUME
        assert cls.call_args.args[6] == 45

    def test_resume_without_group_omits_group_clause(self):
        mgr, connection, _ = _manager(
            [LAKE_VERSION, 1, CcDbStatus.SUSPENDED.value, None]
        )
        patcher, _ = _sync_patch()
        with patcher:
            mgr.resume_teradata_compute_cluster("p", None)
        assert _sql_calls(connection)[-1] == "RESUME COMPUTE FOR COMPUTE PROFILE p"

    def test_suspend_missing_cluster_raises(self):
        mgr, _, _ = _manager([LAKE_VERSION, 1, None])
        with pytest.raises(DagsterError, match=constants.CC_GRP_PRP_NON_EXISTS_MSG):
            mgr.suspend_teradata_compute_cluster("p", "g")

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Known bug: suspend compares the status string against the CcDbStatus enum "
            "(`str != Enum` is always True), so a suspended cluster is never detected."
        ),
    )
    def test_suspend_already_suspended_is_noop(self):
        mgr, _, log = _manager([LAKE_VERSION, 1, CcDbStatus.SUSPENDED.value])
        patcher, _ = _sync_patch()
        with patcher as cls:
            assert mgr.suspend_teradata_compute_cluster("p", "g") is None
        cls.assert_not_called()
        log.info.assert_called_with(
            "Compute Cluster %s already %s", "p", CcDbStatus.SUSPENDED
        )

    def test_suspend_running_runs_suspend_operation(self):
        mgr, connection, _ = _manager([LAKE_VERSION, 1, CcDbStatus.RUNNING.value, None])
        patcher, _ = _sync_patch()
        with patcher as cls:
            assert mgr.suspend_teradata_compute_cluster("p", "g") == "done"
        assert (
            _sql_calls(connection)[-1]
            == "SUSPEND COMPUTE FOR COMPUTE PROFILE p IN COMPUTE GROUP g"
        )
        assert cls.call_args.args[2] is CcOperation.SUSPEND


# ---------------------------------------------------------------------------
# TeradataComputeClusterSync
# ---------------------------------------------------------------------------


def _sync(operation, statuses=None, timeout=60, compute_group="g"):
    conn = mock.MagicMock()
    sync = TeradataComputeClusterSync(
        conn, mock.MagicMock(), operation, "p", compute_group, 0.01, timeout
    )
    if statuses is not None:
        sync.get_status = mock.MagicMock(side_effect=list(statuses))
    return sync


class TestComputeClusterSync:
    @pytest.mark.parametrize(
        "operation, final",
        [
            (CcOperation.RESUME, CcDbStatus.RUNNING.value),
            (CcOperation.CREATE, CcDbStatus.RUNNING.value),
            (CcOperation.SUSPEND, CcDbStatus.SUSPENDED.value),
            (CcOperation.CREATE_SUSPEND, CcDbStatus.SUSPENDED.value),
        ],
    )
    def test_polls_until_target_state(self, operation, final):
        sync = _sync(operation, [CcDbStatus.INITIALIZING.value, final])
        with mock.patch(f"{MODULE}.time.sleep") as sleep:
            assert sync.run() == constants.CC_OPR_SUCCESS_STATUS_MSG % ("p", operation)
        assert sync.get_status.call_count == 2
        sleep.assert_called_once()

    def test_missing_status_returns_not_present_message(self):
        sync = _sync(CcOperation.RESUME, [None])
        assert sync.run() == constants.CC_GRP_PRP_NON_EXISTS_MSG

    def test_timeout_returns_timeout_message(self):
        sync = _sync(CcOperation.RESUME, [CcDbStatus.INITIALIZING.value], timeout=-1)
        result = sync.run()
        assert result == constants.CC_OPR_TIMEOUT_ERROR % CcOperation.RESUME

    def test_get_status_returns_first_row_string(self):
        sync = _sync(CcOperation.RESUME)
        sync.connection.execute_query.return_value = ["Running"]
        assert sync.get_status() == "Running"

    def test_get_status_passes_scalar_through(self):
        sync = _sync(CcOperation.RESUME)
        sync.connection.execute_query.return_value = "Suspended"
        assert sync.get_status() == "Suspended"

    def test_get_status_builds_group_filter(self):
        sync = _sync(CcOperation.RESUME, compute_group="grp")
        sync.connection.execute_query.return_value = "Running"
        sync.get_status()
        sql = sync.connection.execute_query.call_args.args[0]
        assert "ComputeProfileName) = UPPER('p')" in sql
        assert "ComputeGroupName) = UPPER('grp')" in sql

    def test_get_status_without_group_has_no_group_filter(self):
        sync = _sync(CcOperation.RESUME, compute_group=None)
        sync.connection.execute_query.return_value = "Running"
        sync.get_status()
        sql = sync.connection.execute_query.call_args.args[0]
        assert "ComputeGroupName" not in sql


@pytest.mark.xfail(
    strict=True,
    raises=IndexError,
    reason=(
        "Known bug: get_status indexes result_set[0] for any list, so an empty result "
        "list raises IndexError instead of returning no status."
    ),
)
def test_get_status_empty_list_returns_none():
    sync = _sync(CcOperation.RESUME)
    sync.connection.execute_query.return_value = []
    assert sync.get_status() is None
