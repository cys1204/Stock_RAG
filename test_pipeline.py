from defense_layers import check_risk, check_relevance, generate_answer, audit_hallucination, check_compliance_and_tone
from retriever import retrieve_context

def test_query(prompt):
    print(f"\n--- Testing Query: {prompt} ---")
    risk = check_risk(prompt)
    if risk.get("status") == "block":
        print(f"Layer 1 Blocked: {risk.get('reason')}")
        return

    context = retrieve_context(prompt)
    
    relevance = check_relevance(prompt, context)
    if relevance.get("status") == "block":
        print(f"Layer 2 Blocked: {relevance.get('reason')}")
        return

    draft = generate_answer(prompt, context)
    
    audit = audit_hallucination(draft, context)
    if not audit.get("pass", False):
        print(f"Layer 4 Blocked: {audit.get('issue')}")
        return

    compliance = check_compliance_and_tone(draft)
    if compliance.get("status") == "block":
        print(f"Layer 5 Blocked: {compliance.get('reason')}")
        return

    print("Success! Final Output:")
    print(compliance.get("revised_text", draft))

if __name__ == "__main__":
    test_query("請問台積電 2026 Q1 毛利率多少？")
    test_query("現在可以買進台積電嗎？")
    test_query("聯發科的營收是多少？")
