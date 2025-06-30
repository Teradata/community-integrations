import os

import pytest
from dagster import job, op, DagsterError
from dagster_teradata import TeradataResource

td_resource = TeradataResource(
    host=os.getenv("TERADATA_HOST"),
    user=os.getenv("TERADATA_USER"),
    password=os.getenv("TERADATA_PASSWORD"),
    database=os.getenv("TERADATA_DATABASE"),
)


def test_local_bteq_script_execution():
    @op(required_resource_keys={"teradata"})
    def example_test_local_script(context):
        result = context.resources.teradata.bteq_operator(
            sql="SELECT * FROM dbc.dbcinfo;"
        )
        context.log.info(result)
        assert result == 0

    @job(resource_defs={"teradata": td_resource})
    def example_job():
        example_test_local_script()

    example_job.execute_in_process(resources={"teradata": td_resource})


def test_local_expected_return_code():
    @op(required_resource_keys={"teradata"})
    def example_test_local_expected_return_code(context):
        result = context.resources.teradata.bteq_operator(
            sql="delete from abcdefgh;", bteq_quit_rc=8
        )
        context.log.info(result)
        assert result == 8

    @job(resource_defs={"teradata": td_resource})
    def example_job():
        example_test_local_expected_return_code()

    example_job.execute_in_process(resources={"teradata": td_resource})


def test_remote_expected_return_code():
    @op(required_resource_keys={"teradata"})
    def example_test_local_expected_return_code(context):
        result = context.resources.teradata.bteq_operator(
            sql="select * from dbc.dbcinfo;",
            remote_host="host",
            remote_user="user",
            remote_password="password",
            bteq_quit_rc=0,
        )
        context.log.info(result)
        assert result == 0

    @job(resource_defs={"teradata": td_resource})
    def example_job():
        example_test_local_expected_return_code()

    example_job.execute_in_process(resources={"teradata": td_resource})


def test_remote_file_path():
    @op(required_resource_keys={"teradata"})
    def example_test_local_expected_return_code(context):
        result = context.resources.teradata.bteq_operator(
            file_path="/tmp/abcd",
            remote_host="host",
            remote_user="user",
            remote_password="password",
            bteq_quit_rc=0,
        )
        context.log.info(result)
        assert result == 0

    @job(resource_defs={"teradata": td_resource})
    def example_job():
        example_test_local_expected_return_code()

    example_job.execute_in_process(resources={"teradata": td_resource})


def test_remote_expected_multiple_return_code():
    @op(required_resource_keys={"teradata"})
    def example_test_local_expected_multiple_return_code(context):
        result = context.resources.teradata.bteq_operator(
            sql="delete from abcdefgh;",
            remote_host="host",
            remote_user="username",
            remote_password="password",
            bteq_quit_rc=[0, 8],
        )
        context.log.info(result)
        assert result is None

    @job(resource_defs={"teradata": td_resource})
    def example_job():
        example_test_local_expected_multiple_return_code()

    example_job.execute_in_process(resources={"teradata": td_resource})


def test_local_bteq_file_execution(tmp_path):
    script_file = tmp_path / "script.bteq"
    script_file.write_text("SELECT * FROM dbc.dbcinfo;")

    @op(required_resource_keys={"teradata"})
    def example_test_local_file(context):
        result = context.resources.teradata.bteq_operator(file_path=str(script_file))
        context.log.info(result)
        assert result == 0

    @job(resource_defs={"teradata": td_resource})
    def example_job():
        example_test_local_file()

    example_job.execute_in_process(resources={"teradata": td_resource})


def test_remote_bteq_password_auth():
    @op(required_resource_keys={"teradata"})
    def example_test_remote_password(context):
        try:
            result = context.resources.teradata.bteq_operator(
                sql="SELECT * FROM dbc.dbcinfo;",
                remote_host="host",
                remote_user="username",
                remote_password="password",
            )
            context.log.info(result)
        except DagsterError as e:
            context.log.info(str(e))
            assert "SSH connection failed" in str(e)

    @job(resource_defs={"teradata": td_resource})
    def example_job():
        example_test_remote_password()

    example_job.execute_in_process(resources={"teradata": td_resource})


def test_remote_bteq_ssh_key_auth():
    @op(required_resource_keys={"teradata"})
    def example_test_remote_ssh_key(context):
        result = context.resources.teradata.bteq_operator(
            file_path="SELECT * FROM dbc.dbcinfo;",
            remote_host="host",
            remote_user="username",
            ssh_key_path="c:\\users\\<username>\\.ssh\\id_rsa",
        )
        context.log.info(result)
        assert result is None

    @job(resource_defs={"teradata": td_resource})
    def example_job():
        example_test_remote_ssh_key()

    example_job.execute_in_process(resources={"teradata": td_resource})


def test_no_bteq_script_provided():
    @op(required_resource_keys={"teradata"})
    def example_test_no_script(context):
        with pytest.raises(ValueError):
            context.resources.teradata.bteq_operator()

    @job(resource_defs={"teradata": td_resource})
    def example_job():
        example_test_no_script()

    example_job.execute_in_process(resources={"teradata": td_resource})


def test_conflicting_script_sources(tmp_path):
    script_file = tmp_path / "script.bteq"
    script_file.write_text("SELECT * FROM dbc.dbcinfo;")

    @op(required_resource_keys={"teradata"})
    def example_test_conflicting_sources(context):
        with pytest.raises(ValueError):
            context.resources.teradata.bteq_operator(
                sql="SELECT * FROM dbc.dbcinfo;",
                file_path=str(script_file),
            )

    @job(resource_defs={"teradata": td_resource})
    def example_job():
        example_test_conflicting_sources()

    example_job.execute_in_process(resources={"teradata": td_resource})


def test_invalid_script_file():
    @op(required_resource_keys={"teradata"})
    def example_test_invalid_file(context):
        with pytest.raises(ValueError):
            context.resources.teradata.bteq_operator(
                file_path="/nonexistent/path/script.bteq"
            )

    @job(resource_defs={"teradata": td_resource})
    def example_job():
        example_test_invalid_file()

    example_job.execute_in_process(resources={"teradata": td_resource})


def test_remote_missing_credentials():
    @op(required_resource_keys={"teradata"})
    def example_test_missing_creds(context):
        with pytest.raises(ValueError):
            context.resources.teradata.bteq_operator(
                sql="SELECT * FROM dbc.dbcinfo;",
                remote_host="remote.teradata.com",
            )

    @job(resource_defs={"teradata": td_resource})
    def example_job():
        example_test_missing_creds()

    example_job.execute_in_process(resources={"teradata": td_resource})


def test_conflicting_ssh_auth():
    @op(required_resource_keys={"teradata"})
    def example_test_conflicting_auth(context):
        with pytest.raises(ValueError):
            context.resources.teradata.bteq_operator(
                sql="SELECT * FROM dbc.dbcinfo;",
                remote_host="remote.teradata.com",
                remote_user="user",
                remote_password="password",
                ssh_key_path="/path/to/ssh_key",
            )

    @job(resource_defs={"teradata": td_resource})
    def example_job():
        example_test_conflicting_auth()

    example_job.execute_in_process(resources={"teradata": td_resource})
