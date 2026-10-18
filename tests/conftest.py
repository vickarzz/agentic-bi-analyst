import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import generate_data  # noqa: E402

from bi_agent import semantic  # noqa: E402


@pytest.fixture(scope="session")
def sem():
    return semantic.load()


@pytest.fixture(scope="session")
def db_path(tmp_path_factory):
    """A small synthetic database built once per test session."""
    path = tmp_path_factory.mktemp("data") / "test.duckdb"
    generate_data.write_duckdb(generate_data.build(n_customers=3000, seed=7), str(path))
    return str(path)


@pytest.fixture(scope="session")
def agent(db_path, sem):
    from bi_agent.graph import BIAgent
    return BIAgent(sem=sem, db_path=db_path, planner="rules")


@pytest.fixture(scope="session")
def ds_db_path(tmp_path_factory):
    """A medium database for the Data Science Lab (models need more customers than the BI tests)."""
    path = tmp_path_factory.mktemp("ds") / "ds.duckdb"
    generate_data.write_duckdb(generate_data.build(n_customers=8000, seed=11), str(path))
    return str(path)


@pytest.fixture(scope="session")
def ds_lab(ds_db_path):
    from bi_agent.ds.lab import run_lab
    return run_lab(ds_db_path)          # cached next to the test database for later tests
