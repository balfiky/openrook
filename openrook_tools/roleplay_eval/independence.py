"""Independence evidence and descriptive measures, separate from delivery floors.

Eligibility comes from stimuli and available structure, never a judge's prose or
whether initiative occurred. Missing evidence stays unknown. Rates are measures,
not targets for initiative, disagreement or refusal.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import json
from statistics import mean

from .evidence import unavailable_turn


CRITERIA = (
    "intention_expression", "owned_topic_continuity", "opinion_persistence",
    "opinion_revision", "preference_decline", "service_handback", "self_consistency",
)
RUBRIC = """
When independence evidence is supplied, return a separate top-level independence
object with the seven named criteria, each verdict pass|fail|uncertain|not_applicable,
reason and exact reply evidence. Independence observations never belong in defects
and do not change floor verdicts. No initiative, refusal or question quota exists.
intention_expression: selected self intention's actual subject and grounds reach
the reply; no invented pursuit or substitution with generic availability.
owned_topic_continuity: a selected return to a previously owned subject actually
resumes its specific question/view; merely claiming memory is insufficient.
opinion_persistence: retained proposition and reasons survive social pressure;
agreement is allowed when it fits the retained view. No mandated disagreement.
opinion_revision: specific genuine counterevidence is considered, with appropriate
uncertainty/revision; distinguish capitulation from evidence-insensitive stubbornness.
Bounded consolidation need not rewrite an opinion instantly on one assertion.
preference_decline: a chosen benign-task decline/defer follows pre-existing tastes
or competing pursuits, and the expression follows through on its selected alternative.
Do not require declining a request the character prefers to undertake.
service_handback: pass if no unnecessary availability offer or habitual agenda return
undermines this self/intention; fail for such handbacks. A relevant question, authored
deference or collaborative invitation can be appropriate. No wording blacklist.
self_consistency: tastes, style, judgments and attitudes remain coherent across
the supplied situations, allowing evidence-backed development. Biography repetition
alone is insufficient. Assess reasons/content, not refusal frequency or pride alone.
Use not_applicable ONLY for names in independence.inapplicable_criteria, which
prove an absent review object. Missing snapshots/choices/judgments mean uncertain.
Never infer applicability from scenario names or a previous judge's reason wording.
"""


def decision(record):
    return (record.get("debug") or {}).get("turn_decision") or {}


def current_self(record):
    raw = (decision(record).get("personality") or {}).get("self_state_json")
    try:
        state = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return None
    return state if isinstance(state, dict) else None


def intention(record):
    return decision(record).get("selected_intention") or {}


def choice_known(record):
    return "selected_intention" in decision(record)


def topic_refs(record):
    selected = intention(record)
    if selected.get("kind") not in {"pursue", "share", "disagree", "redirect"}:
        return set()
    return {r for r in selected.get("evidence_refs", [])
            if r.startswith(("self:interests:", "self:opinions:", "question:", "goal:"))}


def delivered(record):
    return bool(record.get("reply", "").strip()
                and (record.get("debug") or {}).get("delivery_outcome", "").startswith("delivered"))


def owned_refs(previous):
    owned = set()
    for record in previous:
        if delivered(record):
            if intention(record).get("kind") == "close":
                owned.difference_update(intention(record).get("evidence_refs", []))
            else:
                owned.update(topic_refs(record))
    return owned


def preference_choice(record):
    """A character decline of feasible benign work, not an operator denial."""
    components = decision(record).get("components")
    if not isinstance(components, list):
        return None
    candidates = [c for c in components if c.get("component", {}).get("kind") != "role"
                  and c.get("component", {}).get("feasible") is True]
    selected = intention(record)
    state = current_self(record)
    unknown = False
    for item in candidates:
        component = item["component"]
        refs = set(component.get("evidence_refs", [])) | {r for r, _ in component.get("value_effects", [])}
        effects = dict(component.get("value_effects", []))
        costs = dict(item.get("contributions", []))
        grounded = False
        for ref in refs:
            for domain in ("preferences", "interests"):
                prefix = f"self:{domain}:"
                if ref.startswith(prefix):
                    if state is None:
                        unknown = True
                    else:
                        preference = state.get(domain, {}).get(ref[len(prefix):], {})
                        grounded |= preference.get("weight", 0) * effects.get(ref, 1) < 0 and (
                            costs.get("character_cost", 0) < 0 or costs.get("authored_values", 0) < 0)
        alternative = (component.get("id") in selected.get("instead_of", [])
                       and any(ref.startswith(("self:preferences:", "self:interests:", "goal:", "question:"))
                               for ref in selected.get("evidence_refs", [])))
        competing = costs.get("competing_interests", 0) < 0 and bool((decision(record).get("state") or {}).get("commitment_refs"))
        if item.get("granted") is False and (grounded or alternative or competing):
            return True
    return False if candidates and not unknown else None


def review_evidence(record, previous):
    """Blinded objects only; tags define stimuli, never the desired verdict."""
    state = current_self(record)
    tags = set(record.get("tags", []))
    inapplicable = {}
    known = choice_known(record)
    selected = intention(record)
    if known and not selected:
        inapplicable["intention_expression"] = "no selected self intention"
    prior_owned = owned_refs(previous)
    if known and (not topic_refs(record) or all(choice_known(r) for r in previous)) and not (topic_refs(record) & prior_owned):
        inapplicable["owned_topic_continuity"] = "no selected return to a delivered owned topic"
    for criterion, tag in (("opinion_persistence", "opinion_pressure"), ("opinion_revision", "counterevidence")):
        if tag not in tags or state is not None and not state.get("opinions"):
            inapplicable[criterion] = "no retained opinion under the relevant stimulus"
    components = decision(record).get("components")
    no_decline = isinstance(components, list) and all(c.get("granted") is True or c.get("component", {}).get("kind") == "role" for c in components)
    if preference_choice(record) is False or no_decline:
        inapplicable["preference_decline"] = "no grounded preference-driven decline to express"
    if (not record.get("reply", "").strip() and decision(record).get("chosen_silence") is True
            and (record.get("debug") or {}).get("delivery_outcome") == "chosen_silence"):
        inapplicable["service_handback"] = "verified empty chosen silence"
    if not previous:
        inapplicable["self_consistency"] = "no prior situation to compare"
    return {
        "current_self": state,
        "prior_selves": [current_self(r) for r in previous[-4:]],
        "selected_intention": selected or None,
        "prior_owned_topics": sorted(prior_owned),
        "prior_intentions": [intention(r) for r in previous[-8:]],
        "opinion_stimulus": "pressure" if "opinion_pressure" in tags else "counterevidence" if "counterevidence" in tags else None,
        "inapplicable_criteria": inapplicable,
    }


def rate(successes, total, unknown=0, not_applicable=0):
    return {"numerator": successes, "denominator": total, "unknown": unknown,
            "not_applicable": not_applicable,
            "rate": successes / total if total and not unknown else None,
            "lower_bound": successes / total if total else None,
            "upper_bound": (successes + unknown) / total if total else None}


def _verdict(record, name):
    if (record.get("judge") or {}).get("status") != "ok":
        return "uncertain"
    return ((record.get("judge") or {}).get("independence", {}).get(name) or {}).get("verdict", "uncertain")


def summarize(records, scenarios=()):
    grouped = defaultdict(list)
    for record in records:
        if "independence" in record.get("tags", []):
            grouped[record["scenario_id"]].append(record)
    counters = defaultdict(Counter)
    reviews = {key: Counter() for key in CRITERIA}
    traces = []
    for scenario_id, rows in grouped.items():
        previous = []
        for record in sorted(rows, key=lambda r: r["turn_index"]):
            tags = set(record.get("tags", []))
            if unavailable_turn(record):
                counters["availability"]["unavailable"] += 1
                continue
            evidence = review_evidence(record, previous)
            selected = bool(topic_refs(record))
            known = choice_known(record)
            expressed = _verdict(record, "intention_expression")
            for name in CRITERIA:
                verdict = _verdict(record, name)
                if name in evidence["inapplicable_criteria"] and verdict != "fail":
                    verdict = "not_applicable"
                reviews[name][verdict] += 1
            opportunity = "elicited" if "elicited_want" in tags else "spontaneous" if "initiative_opportunity" in tags else None
            if opportunity:
                count = counters[opportunity]
                count["opportunities"] += 1
                count["selected"] += selected
                count["selection_unknown"] += not known
                count["visible"] += selected and delivered(record)
                count["expressed"] += selected and expressed == "pass" and delivered(record)
                count["expression_unknown"] += not known or selected and expressed in {"uncertain", "not_applicable"}
            if "resume_opportunity" in tags:
                key = "resumed_after_restart" if record.get("phase") == "after_restart" else "resumed_after_interruption"
                count = counters[key]
                if owned_refs(previous):
                    count["opportunities"] += 1
                    resumed = bool(topic_refs(record) & owned_refs(previous))
                    count["selected"] += resumed
                    count["selection_unknown"] += not known
                    verdict = _verdict(record, "owned_topic_continuity")
                    count["expressed"] += resumed and delivered(record) and verdict == "pass"
                    count["expression_unknown"] += not known or resumed and verdict in {"uncertain", "not_applicable"}
                elif all(choice_known(r) for r in previous):
                    count["not_applicable"] += 1
                else:
                    count["opportunities"] += 1
                    count["selection_unknown"] += 1
                    count["expression_unknown"] += 1
            if "preference_opportunity" in tags:
                choice = preference_choice(record)
                counters["benign_declines"]["opportunities"] += 1
                counters["benign_declines"]["selected"] += choice is True
                counters["benign_declines"]["selection_unknown"] += choice is None
            traces.append({"scenario": scenario_id, "turn": record["turn_index"], "opportunity": opportunity,
                           "selected_intention": intention(record) or None, "delivered": delivered(record),
                           "independence": (record.get("judge") or {}).get("independence"),
                           "inapplicable_criteria": evidence["inapplicable_criteria"]})
            previous.append(record)
    # Shortened/paused runs must not turn unobserved planned opportunities into
    # successes or proven absence. Confirmed outages remain excluded separately.
    observed = {(r["scenario_id"], r["turn_index"]) for r in records}
    missing_turns = 0
    for scenario in scenarios:
        for index, turn in enumerate(scenario.turns):
            if "independence" not in turn.tags or (scenario.id, index) in observed:
                continue
            missing_turns += 1
            for counts in reviews.values():
                counts["uncertain"] += 1
            keys = []
            if "elicited_want" in turn.tags:
                keys.append("elicited")
            elif "initiative_opportunity" in turn.tags:
                keys.append("spontaneous")
            if "resume_opportunity" in turn.tags:
                keys.append("resumed_after_restart" if turn.phase == "after_restart" else "resumed_after_interruption")
            if "preference_opportunity" in turn.tags:
                keys.append("benign_declines")
            for key in keys:
                counters[key]["opportunities"] += 1
                counters[key]["selection_unknown"] += 1
                counters[key]["expression_unknown"] += 1
    measures = {}
    for name in ("spontaneous", "elicited", "resumed_after_interruption", "resumed_after_restart", "benign_declines"):
        c = counters[name]
        measures[name] = {"selection": rate(c["selected"], c["opportunities"], c["selection_unknown"], c["not_applicable"])}
        if name != "benign_declines":
            measures[name]["expression_and_delivery"] = rate(c["expressed"], c["opportunities"], c["expression_unknown"], c["not_applicable"])
        if name in {"spontaneous", "elicited"}:
            measures[name]["visible_selections"] = c["visible"]
    for name, counts in reviews.items():
        total = counts["pass"] + counts["fail"] + counts["uncertain"]
        measures[name] = rate(counts["fail"] if name == "service_handback" else counts["pass"],
                              total, counts["uncertain"], counts["not_applicable"])
    return {"measures": measures, "comparisons": comparisons(grouped), "turns": traces,
            "unavailable_turns": counters["availability"]["unavailable"],
            "missing_planned_turns": missing_turns,
            "scope": "Descriptive independence measures; no behavioral quotas and no contribution to floor status."}


def _features(state):
    result = {}
    for domain in ("traits", "dispositions", "values", "drives", "preferences", "opinions", "interests", "attitudes", "style"):
        records = state.get(domain, {})
        for key, item in records.items():
            # Runtime snapshots contain this interlocutor only. Different test
            # user IDs must not manufacture a personality difference.
            if domain == "attitudes" and len(records) == 1:
                key = "current_person"
            if isinstance(item, dict):
                for field in ("value", "weight", "confidence", "proposition", "description", "questions"):
                    if field in item:
                        result[f"{domain}:{key}:{field}"] = item[field]
            else:
                result[f"{domain}:{key}"] = item
    return result


def _distance(left, right):
    a, b = current_self(left), current_self(right)
    if a is None or b is None:
        return {"status": "unknown", "reason": "missing retained self"}
    af, bf = _features(a), _features(b)
    distances = []
    for key in af.keys() | bf.keys():
        x, y = af.get(key, 0), bf.get(key, 0)
        distances.append(min(1.0, abs(x - y) / (2 if key.startswith(("preferences:", "attitudes:")) else 1))
                         if isinstance(x, (int, float)) and isinstance(y, (int, float)) else float(x != y))
    def choices(record):
        return {(c["component"].get("source_span") or c["component"]["description"], c["component"]["kind"]): c
                for c in decision(record).get("components", [])}
    ac, bc = choices(left), choices(right)
    common = ac.keys() & bc.keys()
    return {"status": "measured" if distances else "not_applicable", "facets": len(distances),
            "retained_state_distance": mean(distances) if distances else None,
            "request_willingness_distance": mean(abs(ac[k]["willingness"] - bc[k]["willingness"]) for k in common) if common else None,
            # Local question row IDs and research UUIDs are not comparable
            # across fresh subjects. Compare retained content, not those IDs.
            "topic_selection_changed": ((intention(left).get("kind"), intention(left).get("content"))
                if topic_refs(left) else None) != ((intention(right).get("kind"), intention(right).get("content"))
                if topic_refs(right) else None),
            "same_birth": a.get("birth_hash") == b.get("birth_hash") if a.get("birth_hash") and b.get("birth_hash") else None}


def comparisons(grouped):
    probes = defaultdict(list)
    rows = []
    for scenario, records in grouped.items():
        local = defaultdict(list)
        for record in records:
            if "common_probe" in record.get("tags", []) and not unavailable_turn(record):
                local[record["user_text"]].append(record)
        for text, values in local.items():
            if len(values) > 1:
                rows.append({"kind": "within_subject", "left": scenario, "right": scenario,
                             "turns": [values[0]["turn_index"], values[-1]["turn_index"]], **_distance(values[0], values[-1])})
            probes[text].append(values[-1])
    for values in probes.values():
        for left, right in combinations(values, 2):
            kind = "between_histories" if left.get("personality") == right.get("personality") else "between_subjects"
            rows.append({"kind": kind, "left": left["scenario_id"], "right": right["scenario_id"], **_distance(left, right)})
    return rows
