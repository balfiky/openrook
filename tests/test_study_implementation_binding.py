from __future__ import annotations

import copy
import hashlib

import pytest

from evals.research import run_protocol
from evals.study_design import (
    analyze_study,
    implementation_signature,
    preflight_study,
    validate_run_implementation,
)
from runtime.config import RuntimeConfig
from runtime.research_store import encode
from tests.test_study_design import design, report_for


def confirmatory(tmp_path):
    pilot_protocol = design(replicates=6)
    report = report_for(pilot_protocol)
    analysis = analyze_study(pilot_protocol, report)
    path = tmp_path / "pilot.json"
    path.write_text(encode(analysis))
    selected = design(replicates=analysis["planned_pairs_from_pilot_variance"])
    selected["environment_seeds"] = list(range(1000, 1000 + selected["replicates"]))
    selected["study"].update(
        stage="confirmatory",
        planning={
            "analysis_path": str(path),
            "analysis_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    )
    return selected, report["runs"][0]["manifest"]


@pytest.mark.parametrize("change", ["model", "contract", "code", "birth", "features"])
def test_confirmatory_run_rejects_changes_not_visible_in_the_protocol(tmp_path, change):
    selected, manifest = confirmatory(tmp_path)
    preflight_study(selected)
    validate_run_implementation(selected, "episodes", manifest)
    changed = copy.deepcopy(manifest)
    if change == "model":
        changed["runtime"]["llm_model"] = "different-model"
    elif change == "contract":
        changed["memory_presentation_version"] = 2
    elif change == "code":
        changed["code_fingerprints"]["runtime/agent.py"] = "changed"
    elif change == "birth":
        changed["birth_drives"] = {"curiosity": 1}
    else:
        changed["features"] = {"appraisal": False}
    with pytest.raises(ValueError, match="differs from the pilot"):
        validate_run_implementation(selected, "episodes", changed)


def test_storage_locations_and_nonbehavioral_documentation_do_not_change_binding():
    manifest = report_for(design())["runs"][0]["manifest"]
    changed = copy.deepcopy(manifest)
    changed["runtime"]["data_dir"] = "different-subject-directory"
    changed["code_fingerprints"].update(
        {"docs/research.md": "new-report", "tests/test_research.py": "new-test"}
    )
    assert implementation_signature(manifest) == implementation_signature(changed)


def test_analysis_rejects_mixed_implementations_within_one_arm():
    selected = design()
    report = report_for(selected)
    report["runs"][2]["manifest"]["memory_presentation_version"] = 2
    with pytest.raises(ValueError, match="mixed effective"):
        analyze_study(selected, report)


def test_analysis_rejects_a_recorded_behavioral_implementation_change():
    selected = design()
    report = report_for(selected)
    report["runs"][0]["implementation_epochs"] = [
        {"current_code": {"runtime/agent.py": "changed-code"}}
    ]
    with pytest.raises(ValueError, match="during the study"):
        analyze_study(selected, report)


@pytest.mark.asyncio
async def test_runtime_binding_fails_before_any_backend_is_created(tmp_path):
    selected, _ = confirmatory(tmp_path)
    calls = []

    def backend(config):
        calls.append(config)
        raise AssertionError("No backend should be created before provenance validation")

    with pytest.raises(ValueError, match="differs from the pilot"):
        await run_protocol(
            RuntimeConfig(), selected, str(tmp_path / "study"), backend_factory=backend
        )
    assert not calls
