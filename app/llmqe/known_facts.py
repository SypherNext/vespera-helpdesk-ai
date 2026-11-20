"""Ground-truth facts about Vespera's (fictional) products, used to check
the assistant's answers against reality. Synthetic data only -- no real
customer or company information."""

VESPERA_FACTS = {
    "free_plan_rate_limit": "60 requests per minute",
    "pro_plan_sso": True,
    "enterprise_phone_support": True,
    "pro_plan_phone_support": False,
    "data_retention_days": 90,
}
