"""Research resource signals must bypass optional LangGraph fallbacks."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.emotional_engine import EmotionalEngine
from core.inference_errors import ResearchBudgetExceeded, ResearchStudyLimit
from core.types import ToolCapability, ToolCategory
from openrook_tools import langgraph_orchestrator
from openrook_tools.executor import ToolExecutor
from openrook_tools.registry import ToolRegistry


pytestmark = pytest.mark.skipif(
    langgraph_orchestrator._LANGGRAPH_IMPORT_ERROR is not None,
    reason="optional langgraph dependencies are not installed",
)


@pytest.fixture(params=["budget", "study"])
def resource_signal(request):
    if request.param == "budget":
        return ResearchBudgetExceeded(resume_at=1234.0)
    return ResearchStudyLimit("study call limit reached")


def _runner(model, *, fallback_to_heuristic=False):
    registry = ToolRegistry()
    registry.register(ToolCapability(
        name="test.read",
        description="Read test data",
        category=ToolCategory.READ_ONLY,
        arg_schema={},
    ))
    return langgraph_orchestrator.LangGraphToolRunner(
        executor=ToolExecutor(registry),
        base_url="",
        model="test",
        chat_model=model,
        fallback_to_heuristic=fallback_to_heuristic,
    )


@pytest.mark.parametrize("fallback_to_heuristic", [False, True])
def test_graph_resource_signal_propagates(resource_signal, fallback_to_heuristic):
    model = SimpleNamespace(invoke=Mock(side_effect=resource_signal))
    runner = _runner(model, fallback_to_heuristic=fallback_to_heuristic)
    runner._run_heuristic = Mock()
    engine = EmotionalEngine()

    with pytest.raises(type(resource_signal)) as raised:
        runner.run_tool_loop(
            user_message="read test data",
            state=engine.state,
            person=None,
            defense_active=False,
            engine=engine,
            autonomy_level="high_risk",
        )

    assert raised.value is resource_signal
    model.invoke.assert_called_once()
    runner._run_heuristic.assert_not_called()


@pytest.mark.parametrize("bound", [False, True])
def test_structured_router_resource_signal_propagates(resource_signal, bound):
    bound_model = SimpleNamespace(invoke=Mock(side_effect=resource_signal))
    model = SimpleNamespace(
        bind=Mock(return_value=bound_model),
        invoke=Mock(side_effect=resource_signal),
    )
    if not bound:
        model.bind.side_effect = ValueError("response format is unsupported")
    runner = _runner(model, fallback_to_heuristic=True)

    with pytest.raises(type(resource_signal)) as raised:
        runner._ask_structured_router("read test data", [])

    assert raised.value is resource_signal
    if bound:
        bound_model.invoke.assert_called_once()
        model.invoke.assert_not_called()
    else:
        bound_model.invoke.assert_not_called()
        model.invoke.assert_called_once()
