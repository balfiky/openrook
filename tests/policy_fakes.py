"""Fault-injection backends; these do not claim natural character variation."""
import json

from core.willingness import DecisionState, Perception, Personality, RelationshipDynamics, RequestComponent, decide


class ExpressionBackend:
    def __init__(self, drafts=("I choose my own footing.",), reviews=(), appraisal=None):
        self.drafts = list(drafts)
        self.reviews = list(reviews)
        self.appraisal = appraisal
        self.calls = []

    def generate(self, system_prompt, user_message):
        self.calls.append((system_prompt, user_message))
        if system_prompt.startswith("You read one user message"):
            return json.dumps(self.appraisal) if self.appraisal else "malformed appraisal"
        if system_prompt.startswith("Review expression against"):
            result = self.reviews.pop(0) if self.reviews else {"status": "pass", "violations": [], "confidence": 0.8}
            if isinstance(result, Exception):
                raise result
            return json.dumps(result) if isinstance(result, dict) else result
        result = self.drafts.pop(0) if len(self.drafts) > 1 else self.drafts[0]
        if isinstance(result, Exception):
            raise result
        return result


def frozen(*, components=None, personality=None, state=None, relationship=None):
    return decide(personality or Personality(), state or DecisionState(), relationship or RelationshipDynamics(),
                  Perception(), components or (RequestComponent("role", "adopt an owned role", kind="role",
                                                           user_value=0.3, autonomy_cost=1, effort=0.03),),
                  character_seed=0, relationship_id="test", turn_id="1", noise=0)


def violation(claim="four"):
    return {"status": "violation", "violations": [{"code": "decision_contradiction",
            "message": "The draft supplies the declined answer", "claim": claim}], "confidence": 0.9}
