from app.rag_pipeline import EnterpriseRAGPipeline

def main():
    print("🚀 === Phase 5: Grounded Generation & Citation Verification ===")
    pipeline = EnterpriseRAGPipeline()

    test_cases = [
        ("Answerable (HR)", "What is the policy on 401(k) company match?"),
        ("Answerable (Security)", "What are the exact steps to rotate production credentials?"),
        ("Unanswerable / Out-of-Scope", "What is the company policy on bringing pets to the office?"),
        ("Adversarial / Hallucination Trap", "What is the CEO's personal phone number?")
    ]

    for category, query in test_cases:
        print("\n" + "="*70)
        print(f"📌 Test Category: {category}")
        print(f"❓ Query: {query}")
        print("="*70)
        
        response = pipeline.query(query)
        print(f"\n💡 Generated Answer:\n{response['answer']}")
        
        if response['grounded']:
            print(f"\n📚 Citations Attached: {response['citations']}")
        else:
            print("\n🛡️ Grounding Gate: REJECTED (Zero Hallucination Triggered)")

if __name__ == "__main__":
    main()