"""
AI Insurance Automation - simple base project
---------------------------------------------
Features
  1. Policy lookup (in-memory demo data)
  2. Claim submission with automatic decision (approve / review / reject)
  3. Rule-based fraud risk scoring
  4. Premium quote calculator
  5. AI extraction of claim details from free text (Claude, with regex fallback)

Run:  uvicorn app:app --reload
Docs: http://127.0.0.1:8000/docs
"""
import json
import os
import re
import uuid
from datetime import date, datetime
from enum import Enum
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="AI Insurance Automation", version="0.1.0")

# --------------------------------------------------------------------------
# Demo data (replace with a real database such as PostgreSQL / SQLite)
# --------------------------------------------------------------------------
POLICIES = {
    "POL-1001": {
        "holder": "Ravi Kumar",
        "type": "health",
        "coverage_limit": 500_000,
        "deductible": 5_000,
        "start_date": date(2024, 1, 1),
        "end_date": date(2026, 12, 31),
        "active": True,
    },
    "POL-2002": {
        "holder": "Anita Sharma",
        "type": "auto",
        "coverage_limit": 300_000,
        "deductible": 10_000,
        "start_date": date(2026, 10, 25),  # very new policy -> fraud signal
        "end_date": date(2027, 10, 24),
        "active": True,
    },
    "POL-3003": {
        "holder": "John Mathew",
        "type": "home",
        "coverage_limit": 2_000_000,
        "deductible": 25_000,
        "start_date": date(2022, 1, 1),
        "end_date": date(2025, 12, 31),  # expired
        "active": False,
    },
}
CLAIMS: dict[str, dict] = {}

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class Decision(str, Enum):
    AUTO_APPROVE = "AUTO_APPROVE"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    REJECT = "REJECT"


class ClaimIn(BaseModel):
    policy_id: str
    incident_date: date
    claim_amount: float = Field(gt=0)
    description: str = Field(min_length=5)


class QuoteIn(BaseModel):
    policy_type: str = Field(description="health | auto | home | life")
    age: int = Field(ge=18, le=90)
    coverage_amount: float = Field(gt=0)
    smoker: bool = False
    claims_last_3_years: int = Field(default=0, ge=0)


class ExtractIn(BaseModel):
    text: str = Field(min_length=10)


# --------------------------------------------------------------------------
# Fraud scoring (0-100). Swap in an ML model later.
# --------------------------------------------------------------------------
SUSPICIOUS_WORDS = ["cash only", "no witness", "urgent", "lost receipt", "total loss", "stolen overnight"]


def fraud_score(claim: ClaimIn, policy: dict, prior_claims: int) -> tuple[int, list[str]]:
    score, reasons = 0, []

    days_since_start = (claim.incident_date - policy["start_date"]).days
    if days_since_start < 30:
        score += 30
        reasons.append("Incident within 30 days of policy start")

    if claim.claim_amount > 0.8 * policy["coverage_limit"]:
        score += 25
        reasons.append("Claim exceeds 80% of coverage limit")

    if prior_claims >= 2:
        score += 20
        reasons.append(f"{prior_claims} previous claims on this policy")

    hits = [w for w in SUSPICIOUS_WORDS if w in claim.description.lower()]
    if hits:
        score += 10 * len(hits)
        reasons.append(f"Suspicious wording: {', '.join(hits)}")

    if claim.claim_amount % 10_000 == 0 and claim.claim_amount >= 50_000:
        score += 5
        reasons.append("Suspiciously round claim amount")

    return min(score, 100), reasons


# --------------------------------------------------------------------------
# Decision engine
# --------------------------------------------------------------------------
def decide(claim: ClaimIn, policy: dict, prior_claims: int) -> dict:
    reasons: list[str] = []

    if not policy["active"] or not (policy["start_date"] <= claim.incident_date <= policy["end_date"]):
        return {"decision": Decision.REJECT, "payout": 0, "fraud_score": 0,
                "reasons": ["Policy not active on the incident date"]}

    if claim.claim_amount <= policy["deductible"]:
        return {"decision": Decision.REJECT, "payout": 0, "fraud_score": 0,
                "reasons": ["Claim amount is below the deductible"]}

    score, fraud_reasons = fraud_score(claim, policy, prior_claims)
    reasons += fraud_reasons

    payout = min(claim.claim_amount - policy["deductible"], policy["coverage_limit"])

    if score >= 60:
        decision = Decision.MANUAL_REVIEW
        reasons.append("High fraud risk - sent to investigator")
    elif score >= 30 or claim.claim_amount > 100_000:
        decision = Decision.MANUAL_REVIEW
        reasons.append("Moderate risk or high value - needs human review")
    else:
        decision = Decision.AUTO_APPROVE
        reasons.append("Low risk - approved automatically")

    return {"decision": decision, "payout": round(payout, 2), "fraud_score": score, "reasons": reasons}


# --------------------------------------------------------------------------
# AI extraction (Claude if ANTHROPIC_API_KEY is set, else regex fallback)
# --------------------------------------------------------------------------
def regex_extract(text: str) -> dict:
    policy = re.search(r"POL-\d+", text, re.I)
    amount = re.search(r"(?:rs\.?|inr|₹|\$)\s?([\d,]+(?:\.\d+)?)", text, re.I)
    d = re.search(r"\d{4}-\d{2}-\d{2}", text)
    return {
        "policy_id": policy.group(0).upper() if policy else None,
        "incident_date": d.group(0) if d else None,
        "claim_amount": float(amount.group(1).replace(",", "")) if amount else None,
        "description": text.strip()[:300],
        "source": "regex",
    }


def ai_extract(text: str) -> dict:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return regex_extract(text)
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"),
            max_tokens=500,
            system=(
                "Extract insurance claim details from the user's text. Reply with ONLY a JSON "
                'object: {"policy_id": str|null, "incident_date": "YYYY-MM-DD"|null, '
                '"claim_amount": number|null, "description": str}. No markdown.'
            ),
            messages=[{"role": "user", "content": text}],
        )
        raw = msg.content[0].text.strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw).strip()
        data = json.loads(raw)
        data["source"] = "claude"
        return data
    except Exception as e:  # network / parse errors -> graceful fallback
        data = regex_extract(text)
        data["warning"] = f"AI extraction failed, used regex: {e}"
        return data


# --------------------------------------------------------------------------
# Premium quote
# --------------------------------------------------------------------------
BASE_RATES = {"health": 0.020, "auto": 0.035, "home": 0.012, "life": 0.008}


def calc_premium(q: QuoteIn) -> dict:
    if q.policy_type not in BASE_RATES:
        raise HTTPException(400, f"policy_type must be one of {list(BASE_RATES)}")
    premium = q.coverage_amount * BASE_RATES[q.policy_type]
    factors = {}

    age_factor = 1.0 if q.age < 30 else 1.2 if q.age < 45 else 1.5 if q.age < 60 else 2.0
    factors["age"] = age_factor
    smoker_factor = 1.4 if q.smoker and q.policy_type in ("health", "life") else 1.0
    factors["smoker"] = smoker_factor
    claims_factor = 1 + 0.1 * min(q.claims_last_3_years, 5)
    factors["claims_history"] = claims_factor

    premium *= age_factor * smoker_factor * claims_factor
    return {"annual_premium": round(premium, 2), "monthly_premium": round(premium / 12, 2), "factors": factors}


# --------------------------------------------------------------------------
# API routes
# --------------------------------------------------------------------------
@app.get("/")
def root():
    return {"app": "AI Insurance Automation", "docs": "/docs"}


@app.get("/policies/{policy_id}")
def get_policy(policy_id: str):
    p = POLICIES.get(policy_id)
    if not p:
        raise HTTPException(404, "Policy not found")
    return {"policy_id": policy_id, **p}


@app.post("/claims")
def submit_claim(claim: ClaimIn):
    policy = POLICIES.get(claim.policy_id)
    if not policy:
        raise HTTPException(404, "Policy not found")

    prior = sum(1 for c in CLAIMS.values() if c["policy_id"] == claim.policy_id)
    result = decide(claim, policy, prior)

    claim_id = f"CLM-{uuid.uuid4().hex[:8].upper()}"
    record = {
        "claim_id": claim_id,
        "submitted_at": datetime.utcnow().isoformat(),
        **claim.model_dump(mode="json"),
        **result,
    }
    CLAIMS[claim_id] = record
    return record


@app.post("/claims/from-text")
def claim_from_text(body: ExtractIn):
    """Paste an email / message; AI extracts fields. Review before submitting to /claims."""
    return ai_extract(body.text)


@app.get("/claims")
def list_claims(decision: Optional[Decision] = None):
    items = list(CLAIMS.values())
    if decision:
        items = [c for c in items if c["decision"] == decision]
    return items


@app.get("/claims/{claim_id}")
def get_claim(claim_id: str):
    if claim_id not in CLAIMS:
        raise HTTPException(404, "Claim not found")
    return CLAIMS[claim_id]


@app.post("/quote")
def quote(q: QuoteIn):
    return calc_premium(q)
