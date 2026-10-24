"""The run tests' fixtures: vecadd's cluster on snax_forge (helpers.files) and the recorded
run's task list."""

from __future__ import annotations

import pytest

from snax_forge.lower.tasks import TaskList
from snax_forge.run import Cluster, cluster_from

from .helpers import TASKS, files


@pytest.fixture
def cluster() -> Cluster:
    return cluster_from(*files())


@pytest.fixture
def tasks() -> TaskList:
    return TaskList.load(TASKS)
