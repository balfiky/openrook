"""Display summaries of an already committed decision. No strategy policy."""
from __future__ import annotations

from core.types import AgencyDecision, ResponseStrategy, StrategyDecisionTrace
from core.willingness import TurnDecision


def summarize_decision(decision: TurnDecision) -> ResponseStrategy:
    scores = {
        ResponseStrategy.PRACTICAL_HELP: sum(c.willingness for c in decision.components if c.granted),
        ResponseStrategy.SET_BOUNDARY: decision.stance.boundary,
        ResponseStrategy.REPAIR: decision.stance.repair,
        ResponseStrategy.GIVE_SPACE: (decision.stance.delay + 1 - decision.stance.engagement) / 2,
        ResponseStrategy.REASSURE: decision.stance.warmth,
        ResponseStrategy.CHALLENGE_GENTLY: decision.stance.candor,
    }
    return max(scores, key=scores.get)


def agency_summary(decision: TurnDecision) -> AgencyDecision:
    action = ("comply" if decision.grants else "refuse" if decision.components
              else "disengage" if decision.chosen_silence else "comply")
    return AgencyDecision(action=action, rationale=f"{decision.policy_version}: optimized utility {decision.utility:.6f}",
                          turn_decision=decision)


def decision_trace(decision: TurnDecision) -> StrategyDecisionTrace:
    return StrategyDecisionTrace(selected=summarize_decision(decision).value,
                                 evidence={"turn_decision": decision.to_dict()})


def select_strategy(*, decision: TurnDecision, **_display_only) -> ResponseStrategy:
    return summarize_decision(decision)


def select_strategy_with_trace(*, decision: TurnDecision, **_display_only) -> StrategyDecisionTrace:
    return decision_trace(decision)


STRATEGY_INSTRUCTIONS: dict[str, str] = {}  # labels supply no behavioral sentences
