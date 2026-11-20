"""Vespera's (fictional) product documentation, used as the assistant's
retrieval source. Synthetic data only."""

DOCUMENTS = [
    {"id": "free-plan", "text": "Vespera Free plan: 60 API requests per minute, community forum support only."},
    {"id": "pro-plan", "text": "Vespera Pro plan: includes SSO (single sign-on) for enterprise identity providers, email support with 1 business day response time."},
    {"id": "enterprise-plan", "text": "Vespera Enterprise plan: 24/7 phone support, a dedicated account manager, and custom API rate limits negotiated per contract."},
    {"id": "data-retention", "text": "Vespera retains ticket data for 90 days on all plans, after which it is permanently deleted."},
]
