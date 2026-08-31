from verifier import extract_ana_verify_all

def run_evidence_agent(text: str) -> dict:
    return {'results': extract_and_verify_all(text)}
