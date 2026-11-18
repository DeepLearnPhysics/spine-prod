"""Tests for the public scheduler-client package API."""

from src.client import PBSClient, SlurmClient
from src.client.pbs import PBSClient as ConcretePBSClient
from src.client.slurm import SlurmClient as ConcreteSlurmClient


def test_client_package_exports_concrete_clients():
    assert PBSClient is ConcretePBSClient
    assert SlurmClient is ConcreteSlurmClient
