# AI Insurance Automation (base project)

## Setup
```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key    # optional, enables AI text extraction
uvicorn app:app --reload
```
Open http://127.0.0.1:8000/docs and try the endpoints.

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | /policies/{id} | Look up a policy (try POL-1001) |
| POST | /claims | Submit a claim, get auto decision + fraud score |
| POST | /claims/from-text | Extract claim fields from free text (AI) |
| GET | /claims | List claims (filter by `decision`) |
| POST | /quote | Premium calculator |

## Sample claim
```json
{"policy_id": "POL-1001", "incident_date": "2026-11-01",
 "claim_amount": 45000, "description": "Hospitalised for appendix surgery"}
```

## Next steps
Database (PostgreSQL), authentication, document/image upload and OCR, a trained fraud model,
a React dashboard for adjusters, email/SMS notifications.
