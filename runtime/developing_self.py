"""Consolidated personality, with an immutable birth and auditable evidence.

This module proposes no actions and generates no dialogue. The Life History
transaction owns its writes; willingness remains the only choice owner.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sqlite3
from dataclasses import asdict, is_dataclass
from typing import Any

from config.loader import SoulConfig


NUMERIC_DOMAINS = {"traits", "dispositions", "values", "drives"}
RECORD_DOMAINS = {"opinions", "interests", "preferences", "attitudes", "style"}


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def bound(value: float, lower: float = 0.0) -> float:
    return max(lower, min(1.0, value))


def birth_state(soul: SoulConfig | dict) -> dict:
    """No implicit trait, value, drive, taste, opinion or style is supplied."""
    payload = asdict(soul) if is_dataclass(soul) else copy.deepcopy(soul)
    preferences = {}
    for field, sign in (("likes", 1.0), ("dislikes", -1.0)):
        for subject in payload.get(field, []):
            preferences[subject] = {"subject": subject, "weight": sign, "confidence": 1.0,
                                    "context": "", "reasons": [], "evidence_refs": ["birth"]}
    style = {key: {"description": payload[key], "weight": 1.0, "confidence": 1.0,
                   "reasons": [], "evidence_refs": ["birth"]}
             for key in ("voice", "relational_stance") if payload.get(key)}
    return {
        "schema_version": 1, "revision": 0, "birth_hash": fingerprint(payload),
        "name": payload.get("name", ""), "identity": payload.get("identity", ""),
        "character_seed": payload.get("character_seed", 0),
        "boundaries": payload.get("boundaries", []), "growth_policy": payload.get("growth_policy", ""),
        "traits": dict(payload.get("initial_traits", {})),
        "dispositions": dict(payload.get("dispositions", {})),
        "values": dict(payload.get("core_values", {})),
        "drives": {name: {"value": value, "baseline": value, "birth_baseline": value}
                   for name, value in payload.get("drives", {}).items()},
        "opinions": {}, "interests": {}, "preferences": preferences, "attitudes": {}, "style": style,
    }


def project_self(state: dict, person_id: str = "") -> dict:
    """A detached per-turn snapshot. Other people's attitudes stay private."""
    result = copy.deepcopy(state)
    attitudes = result["attitudes"]
    result["attitudes"] = {person_id: attitudes[person_id]} if person_id in attitudes else {}
    return result


def soul_from_self(state: dict) -> SoulConfig:
    """Compatibility for consumers of the old soul shape; never reload YAML."""
    style = state.get("style", {})
    preferences = state.get("preferences", {})
    return SoulConfig(
        name=state.get("name", ""), identity=state.get("identity", ""),
        character_seed=state.get("character_seed", 0),
        voice=style.get("voice", {}).get("description", ""),
        relational_stance=style.get("relational_stance", {}).get("description", ""),
        boundaries=list(state.get("boundaries", [])), growth_policy=state.get("growth_policy", ""),
        core_values=dict(state.get("values", {})), initial_traits=dict(state.get("traits", {})),
        dispositions=dict(state.get("dispositions", {})),
        drives={k: v["value"] for k, v in state.get("drives", {}).items()},
        likes=[k for k, v in preferences.items() if v.get("weight", 0) > 0],
        dislikes=[k for k, v in preferences.items() if v.get("weight", 0) < 0],
    )


class DevelopingSelf:
    """A small materialized view and its evidence, in the Life History DB."""

    def __init__(self, connection: sqlite3.Connection, *, preserve_legacy_drive_priors: bool = False):
        self.conn = connection
        self.preserve_legacy_drive_priors = preserve_legacy_drive_priors

    def initialize(self, soul: SoulConfig | dict, *, legacy: bool = False) -> None:
        self.conn.execute("""CREATE TABLE IF NOT EXISTS developing_self (
            id INTEGER PRIMARY KEY CHECK (id = 1), birth_json TEXT NOT NULL,
            state_json TEXT NOT NULL, provenance_json TEXT NOT NULL)""")
        self.conn.execute("""CREATE TABLE IF NOT EXISTS self_evidence (
            id INTEGER PRIMARY KEY, experience_id INTEGER NOT NULL,
            domain TEXT NOT NULL, subject TEXT NOT NULL, fingerprint TEXT NOT NULL,
            material_hash TEXT NOT NULL, proposal_json TEXT NOT NULL, weight REAL NOT NULL,
            UNIQUE(domain, subject, fingerprint), UNIQUE(domain, subject, material_hash))""")
        if self.conn.execute("SELECT 1 FROM developing_self WHERE id = 1").fetchone():
            return
        payload = asdict(soul) if is_dataclass(soul) else copy.deepcopy(soul)
        source = "setup_soul"
        genesis = self.conn.execute("SELECT id, created_at, source_hash, payload_json FROM genesis_provenance WHERE source_kind = 'soul_yaml' ORDER BY id LIMIT 1").fetchone()
        if legacy:
            if genesis and json.loads(genesis["payload_json"]):
                payload = json.loads(genesis["payload_json"])
                source = "legacy_genesis"
            else:
                source = "migration_configuration_birth_unavailable"
        state = self._legacy_state(payload) if legacy else birth_state(payload)
        self.conn.execute("INSERT INTO developing_self VALUES (1, ?, ?, ?)", (
            json.dumps(payload, sort_keys=True), json.dumps(state, sort_keys=True),
            json.dumps({"source": source, "birth_hash": fingerprint(payload),
                        "genesis_record_id": genesis["id"] if genesis else None,
                        "genesis_recorded_at": genesis["created_at"] if genesis else None,
                        "genesis_source_hash": genesis["source_hash"] if genesis else None,
                        "legacy_drive_states": [dict(row) for row in self.conn.execute("SELECT * FROM drive_states")] if legacy else [],
                        "legacy_traits": "replayed with evidence bounds" if legacy else ""}),
        ))
        if legacy:
            self._migrate_traits()

    def read(self) -> dict:
        if not self._has_table("developing_self"):
            payload, _ = self._legacy_birth()
            return self._legacy_state(payload)
        row = self.conn.execute("SELECT state_json FROM developing_self WHERE id = 1").fetchone()
        return json.loads(row[0])

    def birth(self) -> dict:
        if not self._has_table("developing_self"):
            payload, provenance = self._legacy_birth()
            return {"soul": payload, "provenance": provenance}
        row = self.conn.execute("SELECT birth_json, provenance_json FROM developing_self WHERE id = 1").fetchone()
        return {"soul": json.loads(row[0]), "provenance": json.loads(row[1])}

    def _has_table(self, name: str) -> bool:
        return self.conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)).fetchone() is not None

    def _legacy_birth(self) -> tuple[dict, dict]:
        row = self.conn.execute("SELECT id, source_hash, payload_json FROM genesis_provenance WHERE source_kind = 'soul_yaml' ORDER BY id LIMIT 1").fetchone() if self._has_table("genesis_provenance") else None
        payload = json.loads(row["payload_json"]) if row else {}
        return payload, {"source": "legacy_genesis_read_only" if row else "birth_unavailable_read_only",
                         "birth_hash": fingerprint(payload), "legacy_record_id": row["id"] if row else None}

    def _legacy_state(self, payload: dict) -> dict:
        """Safe view before a writable open performs migration; no invented traits."""
        state = birth_state(payload)
        if self._has_table("drive_states"):
            for row in self.conn.execute("SELECT * FROM drive_states"):
                name = row["name"]
                changes = self.conn.execute("SELECT id, delta FROM drive_changes WHERE drive_name = ? AND experience_id IS NOT NULL", (name,)).fetchall() if self._has_table("drive_changes") else []
                authored = name in state["drives"]
                if not authored and not changes and not self.preserve_legacy_drive_priors:
                    continue
                # An old hidden 0.50 is not an authored birth. Retain learned
                # deltas and their sources, without importing that extra prior.
                value = bound(number(row["value"]) if authored or self.preserve_legacy_drive_priors else sum(number(change["delta"]) for change in changes))
                state["drives"][name] = {"value": value, "baseline": value,
                    "birth_baseline": payload.get("drives", {}).get(name),
                    "evidence_refs": [f"legacy_drive_change:{change['id']}" for change in changes] or ["legacy_drive_state"]}
        if self._has_table("beliefs"):
            for row in self.conn.execute("SELECT * FROM beliefs WHERE status = 'active'"):
                state["opinions"][row["key"]] = {"proposition": row["statement"],
                    "confidence": bound(number(row["confidence"])), "reasons": [row["evidence"]] if row["evidence"] else [],
                    "evidence_refs": [f"legacy_belief:{row['id']}"]}
        return state

    def _save(self, state: dict) -> None:
        self.conn.execute("UPDATE developing_self SET state_json = ? WHERE id = 1", (json.dumps(state, sort_keys=True),))

    def candidates(self) -> list[dict]:
        """Tentative interpretations for digestion, never effective identity."""
        if not self._has_table("self_evidence"):
            return []
        candidates = {}
        for row in self.conn.execute("SELECT experience_id, proposal_json FROM self_evidence ORDER BY id DESC LIMIT 64"):
            proposal = json.loads(row["proposal_json"])
            projected = {key: proposal[key] for key in ("domain", "key", "proposition", "questions", "description", "subject", "context") if key in proposal}
            projected["direction"] = 1 if number(proposal.get("delta", 0.1)) >= 0 else -1
            identifier = fingerprint(projected)
            candidate = candidates.setdefault(identifier, {**projected, "reason": proposal["reason"], "evidence_refs": []})
            candidate["evidence_refs"].append(f"experience:{row['experience_id']}")
        return list(candidates.values())

    def _migrate_traits(self) -> None:
        rows = self.conn.execute("""SELECT e.*, x.raw_excerpt FROM evolution_events e
            JOIN experience_events x ON x.id = e.experience_id
            WHERE e.domain = 'self_trait' ORDER BY e.id""").fetchall()
        for row in rows:
            metadata = json.loads(row["metadata_json"] or "{}")
            # Text without a numeric delta or an experience remains a ledger
            # observation. Do not infer numerical personality from prose.
            if "delta" not in metadata or not row["reason"] or not row["raw_excerpt"]:
                continue
            self.consolidate(row["experience_id"], [{"domain": "traits", "key": row["subject"],
                "delta": metadata["delta"], "reason": row["reason"], "confidence": row["confidence"],
                "evidence": row["raw_excerpt"], "evidence_kind": "experience"}],
                text=row["raw_excerpt"], quality=1.0, influence=0.5)

    def consolidate(self, experience_id: int, proposals: list, *, text: str,
                    quality: float, influence: float, person_ids: list[str] | None = None,
                    evidence_material: str | None = None) -> list[dict]:
        """Accumulate independent grounded observations, then move slowly.

        Assertions and generated self-descriptions are not observations. One
        intake supplies at most 0.6 evidence mass per facet; consolidation needs
        mass above 1.0 from distinct material. Duplicate/replayed material cannot
        accumulate. Confidence scales both accumulation and the update rate.
        """
        state = self.read()
        changes = []
        seen = set()
        for raw in proposals:
            if not isinstance(raw, dict):
                continue
            domain, key = str(raw.get("domain", "")), str(raw.get("key", "")).strip()[:160]
            if domain not in NUMERIC_DOMAINS | RECORD_DOMAINS or not key or (domain, key) in seen:
                continue
            if (domain in NUMERIC_DOMAINS or "delta" in raw) and not number(raw.get("delta")):
                continue
            seen.add((domain, key))
            if domain == "dispositions" and key not in {"pride", "patience", "forgiveness", "assertiveness"}:
                continue
            if domain == "attitudes" and key not in (person_ids or []):
                continue
            evidence = str(raw.get("evidence") or "").strip()
            reason = str(raw.get("reason") or "").strip()[:900]
            if (raw.get("evidence_kind") not in {"experience", "observed_outcome", "interaction"}
                    or not evidence or evidence not in (evidence_material if evidence_material is not None else text)
                    or not reason):
                continue
            if domain == "opinions" and not str(raw.get("proposition") or "").strip():
                continue
            if domain == "interests" and (not isinstance(raw.get("questions"), list)
                    or not any(q.strip() for q in raw["questions"] if isinstance(q, str))):
                continue
            confidence = bound(number(raw.get("confidence")))
            weight = min(0.6, confidence * bound(quality))
            if weight <= 0:
                continue
            # A renamed title or occurrence ID cannot turn the same material
            # into independent corroboration. Quotes are also deduplicated.
            evidence_hash = fingerprint(" ".join(evidence.casefold().split()))
            proposal = {**raw, "reason": reason, "evidence": evidence[:2000]}
            inserted = self.conn.execute("""INSERT OR IGNORE INTO self_evidence
                (experience_id, domain, subject, fingerprint, material_hash, proposal_json, weight)
                VALUES (?, ?, ?, ?, ?, ?, ?)""", (experience_id, domain, key, evidence_hash,
                                               fingerprint(" ".join(text.casefold().split())),
                                               json.dumps(proposal), weight)).rowcount
            if not inserted:
                continue
            rows = self.conn.execute("SELECT proposal_json, weight, experience_id FROM self_evidence WHERE domain = ? AND subject = ?",
                                     (domain, key)).fetchall()
            direction = 1 if number(raw.get("delta", 0.1)) >= 0 else -1
            compatible = [r for r in rows if self._same_candidate(json.loads(r[0]), raw, direction)]
            mass = sum(r[1] for r in compatible)
            if mass <= 1.0 or len({r[2] for r in compatible}) < 2:
                continue
            rate = min(0.03, 0.04 * confidence * bound(quality) * (0.5 + 0.5 * bound(influence)))
            before = copy.deepcopy(state[domain].get(key))
            if domain in NUMERIC_DOMAINS:
                delta = max(-rate, min(rate, number(raw.get("delta")) * confidence * bound(quality)))
                if domain == "drives":
                    drive = state[domain].setdefault(key, {"value": 0.0, "baseline": 0.0, "birth_baseline": None})
                    drive["value"] = bound(drive["value"] + delta)
                    drive["baseline"] = bound(drive["baseline"] + delta)
                else:
                    state[domain][key] = bound(number(state[domain].get(key)) + delta)
            else:
                self._update_record(state, domain, key, raw, compatible, rate, direction)
            after = state[domain].get(key)
            if after != before:
                changes.append({"domain": domain, "key": key, "before": before, "after": copy.deepcopy(after),
                                "reason": reason, "confidence": confidence, "evidence": evidence,
                                "evidence_refs": [f"experience:{r[2]}" for r in compatible]})
        if changes:
            state["revision"] += 1
            self._save(state)
        return changes

    @staticmethod
    def _same_candidate(previous: dict, current: dict, direction: int) -> bool:
        if current.get("domain") == "opinions":
            return previous.get("proposition") == current.get("proposition") and (1 if number(previous.get("delta", 0.1)) >= 0 else -1) == direction
        if current.get("domain") == "style" and previous.get("description") != current.get("description"):
            return False
        if current.get("domain") == "interests" and previous.get("questions") != current.get("questions"):
            return False
        return (1 if number(previous.get("delta", 0.1)) >= 0 else -1) == direction

    @staticmethod
    def _update_record(state: dict, domain: str, key: str, proposal: dict,
                       evidence: list, rate: float, direction: int) -> None:
        record = state[domain].get(key)
        refs = [f"experience:{r[2]}" for r in evidence]
        reasons = list(dict.fromkeys(json.loads(r[0])["reason"] for r in evidence))[-8:]
        if domain == "opinions":
            proposition = str(proposal["proposition"]).strip()[:1200]
            if record and record["proposition"] != proposition:
                if direction < 0:
                    return  # Evidence against an unowned rival is not against this opinion.
                # Counterevidence first weakens the retained position. A rival
                # with repeated support replaces it only after that confidence
                # has fallen; pressure cannot instantly replace its reasons.
                record["confidence"] = bound(record["confidence"] - rate)
                record["counterevidence_refs"] = refs[-12:]
                record["counterevidence_reasons"] = reasons
                if record["confidence"] > rate * 2:
                    return
                record = None
            if record is None:
                if direction < 0:
                    return
                record = {"proposition": proposition, "confidence": 0.0}
                state[domain][key] = record
            record["confidence"] = bound(record["confidence"] + direction * rate)
            if direction < 0:
                record["counterevidence_refs"] = refs[-12:]
                record["counterevidence_reasons"] = reasons
                return
        else:
            if record is None:
                record = {"weight": 0.0, "confidence": 0.0}
                state[domain][key] = record
            delta = max(-rate, min(rate, number(proposal.get("delta", direction * rate))))
            record["weight"] = bound(record.get("weight", 0.0) + delta, -1.0)
            record["confidence"] = bound(record.get("confidence", 0.0) + rate)
            for field in ("description", "context", "subject"):
                if proposal.get(field):
                    description = str(proposal[field])[:900]
                    if domain == "style" and field == "description" and record.get(field) and record[field] != description:
                        tendencies = record.setdefault("developments", [])
                        tendency = next((item for item in tendencies if item["description"] == description), None)
                        if tendency is None:
                            tendency = {"description": description, "weight": 0.0}
                            tendencies.append(tendency)
                        tendency["weight"] = bound(tendency["weight"] + delta, -1.0)
                        tendency["evidence_refs"] = refs[-12:]
                    else:
                        record[field] = description
            if domain == "interests":
                record["questions"] = list(dict.fromkeys([*record.get("questions", []),
                    *(q[:500] for q in proposal.get("questions", []) if isinstance(q, str) and q.strip())]))[-8:]
        record["reasons"] = list(dict.fromkeys([*record.get("reasons", []), *reasons]))[-8:]
        record["evidence_refs"] = list(dict.fromkeys([*record.get("evidence_refs", []), *refs]))[-12:]
