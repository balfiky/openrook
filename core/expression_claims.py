"""Conservative local proposition checks; unfamiliar meanings remain semantic.

These checks bind subjects and polarity, rather than assigning a judgment to a
particular adjective or amount. They neither select willingness nor rewrite prose.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Predication:
    subject: str
    predicate: tuple[str, ...]
    negative: bool
    claim: str


_CLAUSES = re.compile(r"(?<=[.!?])\s+|[,;\n]+|\b(?:and|but|yet)\b", re.I)
_COPULA = re.compile(
    r"^(?P<subject>.+?)(?:\s+(?P<verb>is|are|am|was|were)|(?P<contract>'s|'re|'m))\s+"
    r"(?P<predicate>.+)$", re.I,
)
_NEGATION = re.compile(r"^(?:not|never)\s+(?P<predicate>.+)$", re.I)
_QUALIFIER = re.compile(r"^(?:exactly|really)\s+", re.I)
_REFERENCE = re.compile(r"^(?:it|this|that|which|this amount|that amount)$", re.I)
_COUNTS = {word: n for n, word in enumerate(
    ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"))}
_COUNTED_REQUEST = re.compile(
    r"\b(?:suggest|list|give|provide|name|offer)\s+(?:(?:me|us)\s+)?"
    r"(?P<count>\d+|" + "|".join(_COUNTS) + r")\s+(?:\w+\s+){0,2}(?P<kind>\w+s)\b", re.I)


def counted_items(user: str, response: str) -> tuple[int, list[str]] | None:
    """Recognize explicit small enumerations; unfamiliar output stays semantic.

    Lists, emphasized names and quoted items provide structural evidence. This
    does not judge the quality or suitability of their contents.
    """
    request = _COUNTED_REQUEST.search(user)
    if not request:
        return None
    raw = request["count"].casefold()
    count = int(raw) if raw.isdigit() else _COUNTS[raw]
    if not 2 <= count <= 10:
        return None
    items = re.findall(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)(.+)$", response, re.M)
    if not items:
        items = re.findall(r'\*([^*\n]+)\*|["“]([^"”\n]+)["”]', response)
        items = [a or b for a, b in items]
    return count, items


def supported_preference_recall(user: str, response: str, memories) -> bool:
    """Exact saved preference content can disprove a claim of absent recall.

    Only a direct affirmative recall, with the preference's complete wording,
    qualifies. Paraphrases and unsupported memories remain semantic questions.
    """
    if re.search(r"\b(?:not|never|don't|can't|cannot)\b", response, re.I):
        return False
    return bool(recalled_fact_spans(user, response, memories, preference_only=True))


def recalled_fact_spans(user: str, response: str, memories, *, preference_only=False) -> tuple[tuple[int, int], ...]:
    """Locate complete, exactly supported facts requested again on this turn.

    Only saved preference/fact evidence qualifies, never an earlier assistant
    reply alone. Keep byte ranges so freshness can still assess adjacent prose.
    Unfamiliar questions and paraphrased facts remain semantic questions.
    """
    if not re.search(r"\b(?:remember|recall|preference)\b", user, re.I):
        return ()
    tokens = list(re.finditer(r"\w+", response))
    words = [m[0].casefold() for m in tokens]
    spans = set()
    for memory in memories:
        text = memory.get("summary", "")
        preference = re.search(r"\bpreference\s*:\s*(.+)", text, re.I)
        if preference:
            text = preference[1]
        elif memory.get("kind") != "preference":
            if (preference_only or re.search(r"\bpreference\b", user, re.I)
                    or memory.get("kind") != "fact"):
                continue
        fact = re.findall(r"\w+", text.casefold())
        if len(fact) < 4:
            continue
        for start in range(len(words) - len(fact) + 1):
            if words[start:start + len(fact)] == fact:
                spans.add((tokens[start].start(), tokens[start + len(fact) - 1].end()))
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = merged[-1][0], max(end, merged[-1][1])
        else:
            merged.append((start, end))
    return tuple(merged)


def contradictory_predicates(text: str) -> list[Predication]:
    """Find opposite assertions of the same predicate about the same subject.

    An adjacent relative pronoun or subjectless continuation inherits its
    antecedent. Explicit subject changes, questions, quotations, conditions and
    workload/comparison qualifiers do not establish this local contradiction.
    """
    prior: dict[tuple[str, str, tuple[str, ...]], bool] = {}
    subject = ""
    tense = "present"
    findings = []
    for clause in _CLAUSES.split(text.replace("’", "'")):
        clause = clause.strip()
        if not clause:
            continue
        if re.search(r"[?\"`]|\b(?:if|unless|perhaps|maybe|for|than|necessarily)\b", clause, re.I):
            subject = ""
            continue
        parsed = _COPULA.fullmatch(clause.rstrip(".!"))
        if parsed:
            current = parsed["subject"].casefold().strip()
            if not _REFERENCE.fullmatch(current):
                subject = current
            tense = "past" if (parsed["verb"] or "").casefold() in {"was", "were"} else "present"
            predicate = parsed["predicate"]
        elif _NEGATION.match(clause):
            predicate = clause
        else:
            # A short elliptical judgment can continue the previous subject;
            # unrelated complete prose must not lend it an old antecedent.
            subject = ""
            continue
        negative = _NEGATION.fullmatch(predicate.rstrip(".!"))
        if negative:
            predicate = negative["predicate"]
        predicate = _QUALIFIER.sub("", predicate)
        predicate = re.sub(r"\s+(?:either|too)$", "", predicate.rstrip(".!"), flags=re.I)
        words = tuple(re.findall(r"\w+", predicate.casefold()))
        if not subject or not words:
            continue
        key = subject, tense, words
        if key in prior and prior[key] != bool(negative):
            findings.append(Predication(subject, words, bool(negative), clause))
        prior[key] = bool(negative)
    return findings
