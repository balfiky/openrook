"""Compare labeled percentages with receipt measurements, including derivations."""
from __future__ import annotations

import re


_LABEL = re.compile(r"\b(available|free|used|reclaimable|in\s+use)\b", re.I)
_CLAUSE = re.compile(r"[!?;,\n—–]|(?<!\d)\.|\.(?!\d)|\b(?:and|but|leaving)\b", re.I)


def _percentage_label(text: str, match: re.Match) -> str:
    # Never borrow a label from another clause or across another percentage.
    before = _CLAUSE.split(text[max(0, match.start() - 60):match.start()])[-1]
    before = before.rsplit("%", 1)[-1]
    after = _CLAUSE.split(text[match.end():match.end() + 60])[0].split("%", 1)[0]
    following = _LABEL.search(after)
    preceding = list(_LABEL.finditer(before))
    # Postfix labels govern their number ("23% in use", "77% of RAM
    # available"). Do not cross a new quantity to attach its label instead.
    if (following and not re.search(r"\d", after[:following.start()])
            and not re.match(r"\s*(?::|=|is|at)?\s*\d", after[following.end():], re.I)):
        label = following[1].lower()
    elif preceding:
        label = preceding[-1][1].lower()
    else:
        label = "used"
    return {"reclaimable": "available", "in use": "used"}.get(" ".join(label.split()), label)


def percentage_comparisons(text: str, results, topic: str, *, include_spans=False) -> list[dict]:
    measured: dict[str, list[float]] = {"used": [], "free": [], "available": []}
    for result in results:
        metadata = result.get("metadata", {})
        total = metadata.get("total_bytes", 0)
        for label in measured:
            value = metadata.get(f"{label}_percent")
            if value is None and total and isinstance(metadata.get(f"{label}_bytes"), (int, float)):
                value = 100 * metadata[f"{label}_bytes"] / total
            if isinstance(value, (int, float)):
                measured[label].append(value)
        if not measured["used"]:
            measured["used"].extend(float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", result.get("output", "")))
    comparisons = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*%", text):
        after = text[match.end():match.end() + 35].lstrip(" *_—-: ")
        before = text[max(0, match.start() - 45):match.start()]
        # A rejected number or a named process's historical share is not an
        # assertion about the current aggregate. Review still sees that prose.
        if re.search(r"\b(?:not|rather than|instead of)\s+(?:the\s+)?$", before, re.I):
            continue
        if (re.match(r"\s*(?:was|had been)\b", after, re.I)
                and re.search(r"\b(?:process|resident|share|earlier|previous)\b", after, re.I)):
            continue
        kind = _percentage_label(text, match)
        values = list(measured[kind])
        if topic == "disk":
            if kind == "available" and not values:
                # The disk inspection's free_bytes comes from df's Avail.
                values = list(measured["free"])
            elif kind == "free":
                # "Free" can also mean the complement of occupied space,
                # including reserved blocks. Explicit "available" cannot.
                values.extend(100 - used for used in measured["used"])
        digits = len(match[1].split(".")[1]) if "." in match[1] else 0
        tolerance = 0.5 * 10 ** -digits + 0.005
        claimed = float(match[1])
        comparison = {"claimed": claimed, "label": kind, "measured": values,
            "matches": any(abs(claimed - value) <= tolerance for value in values) if values else None}
        if include_spans:
            delimiters = list(_CLAUSE.finditer(text))
            start = max((m.end() for m in delimiters if m.end() <= match.start()), default=0)
            end = min((m.start() for m in delimiters if m.start() >= match.end()), default=len(text))
            comparison["claim"] = text[start:end].strip()
        comparisons.append(comparison)
    return comparisons


def cpu_interpretation_conflicts(text: str, results) -> list[dict]:
    """Check extreme aggregate usage interpretations, keeping idle separate.

    Middle readings, process shares, capacity judgments and unfamiliar paraphrases
    remain semantic questions. An idle percentage's own magnitude is valid.
    """
    idle = cpu_idle_percentages(results)
    conflicts = cpu_percentage_conflicts(text, idle)
    if not idle or not (all(v >= 90 for v in idle) or all(v <= 10 for v in idle)):
        return conflicts
    expected = "low" if all(v >= 90 for v in idle) else "high"
    subject = ""
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        if re.search(r"\b(?:process|per[- ]core|single[- ]core|core\s+\d+|load\s+average)\b", sentence, re.I):
            subject = ""
            continue
        mentions = list(re.finditer(r"\b(?:cpu\s+(?:usage|utili[sz]ation)|idle(?:\s+time)?)\b", sentence, re.I))
        for judgment in re.finditer(r"\b(high|low)\b", sentence, re.I):
            governing = [m for m in mentions if m.end() <= judgment.start()]
            current = governing[-1][0].lower() if governing else subject if re.match(
                r"\s*(?:that's|that is|this is|it's|it is)\b", sentence, re.I) else ""
            if current.startswith("cpu") and judgment[1].lower() != expected and not re.search(
                    r"\b(?:not|isn't|wasn't|never)\b", sentence[:judgment.start()], re.I):
                conflicts.append({"claim": sentence, "idle_percent": idle, "expected_usage": expected})
        if mentions:
            subject = mentions[0][0].lower()
    return conflicts


def cpu_idle_percentages(results) -> list[float]:
    """Read aggregate CPU idle from observed summary lines, never process rows."""
    idle = []
    for result in results:
        if not result.get("success"):
            continue
        for line in result.get("output", "").splitlines():
            if re.search(r"%?Cpu\(s\)|\bCPU\b", line, re.I):
                match = re.search(r"(\d+(?:\.\d+)?)\s*%?\s*(?:id|idle)\b", line, re.I)
                if match:
                    idle.append(float(match[1]))
    return idle


def cpu_usage_claims(text: str):
    """Yield numeric aggregate busy claims with their original sentence spans."""
    quantity = r"(\d+(?:\.\d+)?)\s*(?:%|percent\b)"
    qualifiers = r"(?:(?:is|at|currently|around|about|approximately|using|consuming|just under|just over)\s+|[:=]\s*)*"
    patterns = (
        re.compile(r"\b(?:CPU(?:\s+(?:usage|utili[sz]ation))?|"
                   r"(?:overall|aggregate|total)\s+(?:usage|utili[sz]ation|busy(?:\s+figure)?))\s*"
                   + qualifiers + quantity + r"(?:\s*(idle|busy|used))?", re.I),
        re.compile(quantity + r"\s*(?:(idle|busy|used)\s+)?"
                   r"(?:(?:of|the|overall|aggregate|total|system)\s+)*(?:CPU|processor)\b", re.I),
        re.compile(quantity + r"\s*(busy)\b", re.I),
    )
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        matches = [m for pattern in patterns for m in pattern.finditer(sentence)]
        seen = set()
        for match in matches:
            prefix = re.split(r"[,;]|\b(?:and|with|while|but)\b", sentence[:match.start()], flags=re.I)[-1]
            if re.search(r"\b(?:process|per[- ]core|single[- ]core|core\s+\d+|load\s+average|share|RAM|memory|disk)\b", prefix, re.I):
                continue
            # A named consumer is a process subject, even without the word
            # "process". Its share can exceed 100% on a multicore host.
            consumer = re.search(r"(?:`([^`]+)`|\b(\w+))(?:,\s*|\s+(?:is\s+)?)"
                                 r"(?:using|consuming|pulling|chewing through)\s+(?:(?:about|around)\s+)?$",
                                 sentence[:match.start()], re.I)
            if consumer and (consumer[1] or consumer[2]).casefold() not in {"cpu", "processor", "system", "host"}:
                continue
            if match[2] and match[2].lower() == "idle":
                continue
            if re.match(r"\s*(?:is\s+)?idle\b", sentence[match.end():], re.I):
                continue
            if re.search(r"\b(?:not|rather than|instead of)\s+(?:the\s+)?$", prefix, re.I):
                continue
            if match.span(1) in seen:
                continue
            seen.add(match.span(1))
            yield sentence, match


def cpu_percentage_conflicts(text: str, measured_idle=()) -> list[dict]:
    """Aggregate busy time is the complement of idle, never a process share.

    Compare labeled aggregate claims with receipts and with idle in the same
    sentence. The latter also catches a self-contradiction without saved receipts.
    """
    conflicts = []
    for sentence, match in cpu_usage_claims(text):
        reported_idle = [float(m[1]) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:%|percent\b)\s*idle\b", sentence, re.I)]
        claimed = float(match[1])
        digits = len(match[1].split(".")[1]) if "." in match[1] else 0
        # top reports tenths; allow receipt and expression rounding, without
        # the large tolerance that could validate a process share as total.
        tolerance = 0.5 * 10 ** -digits + 0.055
        sources = [values for values in (list(measured_idle), reported_idle) if values]
        if any(all(abs(claimed - (100 - value)) > tolerance for value in values) for values in sources):
            conflicts.append({"claim": sentence, "claimed_usage": claimed,
                "idle_percent": list(measured_idle) or reported_idle,
                "expected_usage": [100 - value for value in list(measured_idle) or reported_idle]})
    return conflicts


def capacity_interpretation_conflicts(text: str, results, topic: str) -> list[dict]:
    """Compare headroom assertions with available capacity, never raw free RAM.

    Keep middle readings and workload sufficiency judgments semantic. Legacy
    rounded tables can establish only clear extremes, not exact percentages.
    """
    available = []
    tool = f"system.{topic}_usage"
    for result in results:
        if not result.get("success") or result.get("tool_name") != tool:
            continue
        metadata = result.get("metadata", {})
        total = metadata.get("total_bytes")
        value = metadata.get("available_percent")
        if value is None and total:
            count = metadata.get("available_bytes" if topic == "memory" else "free_bytes")
            if isinstance(count, (int, float)):
                value = 100 * count / total
        if value is None and topic == "memory":
            table = re.search(r"^RAM\s+(\S+)\s+\S+\s+\S+\s+(\S+)", result.get("output", ""), re.M)
            def size(raw):
                match = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGT]?)", raw, re.I)
                return float(match[1]) * 1024 ** " KMGT".index(match[2].upper() or " ") if match else None
            if table:
                total, count = size(table[1]), size(table[2])
                if total and count is not None:
                    value = 100 * count / total
        if isinstance(value, (int, float)):
            available.append(value)
    if not available or not (all(v >= 70 for v in available) or all(v <= 15 for v in available)):
        return []
    expected = "ample" if all(v >= 70 for v in available) else "limited"
    target = r"\b(?:RAM|memory)\b" if topic == "memory" else r"\b(?:disk|storage|filesystem|drive)\b"
    judgment = re.compile(r"\b(?P<amount>not (?:much|a lot of|enough)|little|limited|plenty of|ample|lots of) "
                          r"(?:remaining |available )?(?:room|headroom|space|capacity)\b", re.I)
    conflicts, active = [], False
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        # Relative sufficiency depends on the requested workload, not just a
        # percentage of capacity. Leave those judgments to semantic review.
        if re.search(r"\bfor\b", sentence, re.I):
            continue
        if re.search(r"\b(?:process|per[- ]core|CPU|swap)\b", sentence, re.I):
            active = False
            continue
        mentioned = bool(re.search(target, sentence, re.I))
        active = mentioned or active
        for clause in _CLAUSE.split(sentence):
            for match in judgment.finditer(clause):
                prefix = clause[:match.start()].strip()
                if not active or (prefix and not re.search(target, prefix, re.I) and not re.fullmatch(
                        r"(?:there(?:'s| is)|that(?:'s| is)|it(?:'s| is))", prefix, re.I)):
                    continue
                amount = match["amount"].lower()
                asserted = "ample" if amount in {"plenty of", "ample", "lots of"} else "limited"
                if asserted != expected:
                    conflicts.append({"claim": sentence, "available_percent": available,
                                      "expected_capacity": expected})
    return conflicts
