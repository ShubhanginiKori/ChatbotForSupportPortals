# Acme RAG Chatbot Support Runbook

## Service Summary
- **Service name:** Acme Assist RAG Chatbot (customer self-service & internal knowledge queries).
- **Stack overview:** FastAPI gateway (`main.py`) → Retriever (`utils/retriever.py`) → Embedding store (`store/`) → LLM hosted in `models/`.
- **Critical data:** Support playbooks in `docs/`, user tickets (Zendesk), and inference telemetry in DataDog (dashboard `RAG/Assist`).

## Tier-1 Responsibilities
1. Keep chatbot uptime ≥ 99.5% and median latency ≤ 1.5s.
2. Ensure all ingestion jobs succeed nightly (02:00 UTC) and confirm doc freshness.
3. Serve as first responder for tickets tagged `product:rag-chatbot`.
4. Communicate incidents in Slack `#rag-ops` within 5 minutes of detection.

## Triage Checklist
1. Gather ticket details: user ID, prompt text, timestamp, and screenshot when available.
2. Verify status endpoint: `curl https://chatbot.acme.com/health`.
3. Compare reported issue to current incidents board (Statuspage).
4. Pull recent logs: `tail -n 200 logs/app.log | rg "<user_id>"`.
5. Reproduce in staging: `python main.py --prompt "<prompt>" --env staging`.
6. Decide severity (P1–P3) and follow escalation matrix below.

## Troubleshooting Procedures
| Scenario | Diagnostic Steps | Fix Actions |
| --- | --- | --- |
| **Bot returns "I cannot answer that" for known policy** | Check ingestion job status in Airflow (`KB-Ingest`). Validate doc exists in `store/index.json`. | Run `python utils/ingest.py --recreate`, then `python utils/refresh_embeddings.py`. Notify QA to spot-check. |
| **Latency > 3s** | Inspect DataDog `Latency` graph, confirm load vs worker count in `main.py`. | Scale API pods via `kubectl scale deploy/rag-api --replicas=<n>`. If model saturated, switch to `models/v2` by toggling `MODEL_NAME` env var. |
| **Hallucinated responses** | Review retrieval scores in logs (`score=<value>`). If <0.3, similarity threshold too low. | Update `MIN_SCORE` constant in `utils/retriever.py`, redeploy. Flag conversation to content team for manual remediation. |
| **Regional users cannot connect** | Ping regional edge: `curl https://<region>.chatbot.acme.com/health`. Check CDN status. | Fail over DNS via Akamai playbook, alert network ops if >10 min impact. |

### Escalation Matrix
- **P1 (total outage, security incident):** Page ML On-Call (`+1-650-555-1000`), call Network Ops if region-specific.
- **P2 (accuracy degradation, ingestion failure):** Notify Product Owner (Mia Ortiz) via Slack DM and open Jira `RAG-INC`.
- **P3 (single customer impact):** Own to resolution within 1 business day, update Zendesk ticket hourly if open.

## Knowledge Base Hygiene
1. **Document intake:** Store net-new PDFs in `docs/supporting/<YYYY-MM-DD>-<title>.pdf`.
2. **Validation:** Run `python utils/ingest.py --dry-run` first; confirm summary output lists new doc names.
3. **Embedding refresh:** `python utils/refresh_embeddings.py --source docs/supporting`.
4. **Audit:** Every Friday, export KB diffs `python utils/kb_audit.py --week <ISO-week>` and file summary in Confluence.


## Regional Support Contacts
| Region | Hours (local) | Hotline | Email / Slack |
| --- | --- | --- | --- |
| Americas (AMER) | 24/7 | +1-415-555-0110 | support-amer@acme.com / Slack `#rag-ops-amer` |
| Europe, Middle East, Africa (EMEA) | 07:00–22:00 CET | +49-30-555-0200 | support-emea@acme.com / Slack `#rag-ops-emea` |
| Asia-Pacific (APAC) | 06:00–23:00 SGT | +65-555-0770 | support-apac@acme.com / Slack `#rag-ops-apac` |
| After-hours Backup | PagerDuty escalation | +1-650-555-9999 | On-call alias `rag-oncall@acme.com` |

## Communication Templates
- **Customer update:**
  “Hi <name>, our RAG assistant experienced <issue>. Mitigation is in progress (tracking ID <INC#>), next update by <time>. Thank you for your patience.”
- **Internal status post:**
  “RAG Chatbot incident <INC#>: symptom <summary>. Impacted % of traffic: <value>. Mitigation owner: <name>. ETA for resolution: <time>. Tracking doc: <link>.”

Keep this runbook accessible in your Zendesk macro panel. When procedures change, submit a PR updating this file and mention `@rag-ops-leads` for approval.
