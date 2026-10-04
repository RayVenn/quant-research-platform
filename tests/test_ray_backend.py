import pytest

ray = pytest.importorskip("ray")

from pairlab.pipeline import run_job  # noqa: E402


@pytest.mark.ray
def test_ray_backend(spec, workspace):
    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
    try:
        ex = spec.execution.model_copy(update={"backend": "ray"})
        s = run_job(spec.model_copy(update={"execution": ex}), workspace)
        assert s["tasks"]["succeeded"] == 4 and s["tasks"]["failed"] == 0
    finally:
        ray.shutdown()
