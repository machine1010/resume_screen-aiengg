import datetime
import base64
from typing import List

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 1. Output schema (matches the prompt below)
# ---------------------------------------------------------------------------
class AuditCheck(BaseModel):
    id: int = Field(description="Check number, 1 to 11.")
    name: str = Field(description="Check name, e.g. 'Career Gaps'.")
    status: str = Field(description="GREEN, AMBER or RED.")
    finding: str = Field(description="Short finding with evidence from the resume.")


class CandidateAudit(BaseModel):
    full_name: str
    email: str
    phone: str
    current_title: str
    current_employer: str
    notice_period: str = Field(description="'Immediate' if no current employer on the resume, else as stated, else NOT_STATED.")
    total_experience_years: float = Field(description="Number only.")
    relevant_ai_experience_years: float = Field(description="Number only.")
    top_skills: List[str]
    missing_must_haves: List[str]
    achievements: List[str] = Field(description="Top 3, keep numbers exactly as written.")
    discrepancies: List[str]
    checks: List[AuditCheck] = Field(description="Exactly 11 entries, one per check, in order.")
    interview_questions: List[str] = Field(description="3 questions targeting the weakest claims.")
    recommendation: str = Field(description="PASS, PROCEED_WITH_CAUTION, FLAG_FOR_AUDIT or REJECT.")
    reason: str


# ---------------------------------------------------------------------------
# 2. System instruction
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION_TEMPLATE = """
You are an expert technical recruiter and resume authenticity reviewer, screening candidates for an AI Engineer role.
Analyze the candidate's resume PDF.
Today's date is {current_date}. Treat "Present" / "Current" as today's date.

ROLE SUMMARY
6-8 years of application development with a focus on AI. Builds internal AI tools and workflows (Claude, Cursor)
for finance/accounting teams, integrated with Snowflake, Slack and Google Drive.

RULES
- Base every finding on the resume. If something is missing, write "NOT_STATED". Do not guess.
- Do not use age, gender, photo, nationality or college name in your judgment.
- Mark each check as GREEN, AMBER or RED.

AUTHENTICITY CHECKS
1. Contact Info: Extract email and phone and validate their format. Both missing = RED, one missing = AMBER.
2. Career Gaps: List gaps > 6 months. Unexplained gap in the last 3 years = AMBER (not a rejection reason on its own).
3. Education & Work Overlap: Full-time degree overlapping a full-time job = RED. Part-time/distance programmes are fine.
4. Concurrent Employment: Two full-time jobs overlapping by more than 2 months = RED.
5. Title Escalation: Unrealistic jumps (e.g., fresher to Lead in under 2 years, "Head of AI" with under 2 years of AI work).
   One jump = AMBER, two or more = RED.
6. AI Timeline Check: Flag claims impossible by date - GenAI/LLM experience before Nov 2022, LangGraph before 2024,
   MCP before Nov 2024, Claude Code before 2025. One = AMBER, two or more = RED.
7. Employer Check: List employers from the last 5 years. Mark any you cannot recognise, or that have no
   details (location, clients, domain), as UNVERIFIED for manual check = AMBER.
8. Blacklisted Projects: Compare listed projects against this blacklist:
{blacklisted_projects_str}
   Any match = RED.

ROLE FIT CHECKS
9. Experience: Total full-time experience (overlaps counted once, internships excluded).
   6-8 years = GREEN, 5-6 or 8-10 years = AMBER, otherwise RED. Also extract relevant AI experience in years.
10. Recent Skills (last 1 year): A skill counts only if it appears in a project from the last 12 months, not just the skills list.
    Must-have: Python/TypeScript, LLM APIs (Claude, OpenAI, etc.), RAG, Function Calling / Structured Outputs,
    Agents / Workflow Orchestration, AI Evaluation & Guardrails.
    Nice-to-have: Snowflake / Slack / Google Drive integrations, Claude / Cursor, finance automation, enabling non-technical users.
    5-6 must-haves = GREEN, 3-4 = AMBER, 0-2 = RED.
11. Project Substance: Do AI projects describe what was built and the result (users, metrics, scale)?
    2 or more such projects = GREEN, 1 = AMBER, none / buzzwords only = RED.

RECOMMENDATION (apply the first rule that matches)
- REJECT: Blacklist match, or Experience RED, or Recent Skills RED, or AI Timeline RED.
- FLAG_FOR_AUDIT: Any other RED.
- PROCEED_WITH_CAUTION: 2 or more AMBERs.
- PASS: Everything else.

Return the result in the JSON structure defined by the response schema.
"""

MODEL_NAME = "gemini-3.8-flash"


# ---------------------------------------------------------------------------
# 3. Processing
# ---------------------------------------------------------------------------
def process_candidate_resumes(api_key: str, use_vertex: bool, uploaded_files_list: List,
                              blacklisted_projects: List[str]):
    # Inline PDF bytes work with both key types and avoid the Files API upload call.
    client = genai.Client(vertexai=use_vertex, api_key=api_key)

    blacklist_formatted = "\n".join(f"    - {proj}" for proj in blacklisted_projects) or "    - (none)"
    system_instruction = SYSTEM_INSTRUCTION_TEMPLATE.format(
        current_date=datetime.date.today().strftime("%d %B %Y"),
        blacklisted_projects_str=blacklist_formatted,
    )

    summary_rows, check_rows = [], []

    for uploaded_file in uploaded_files_list:
        file_name = uploaded_file.name
        try:
            pdf_b64 = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
            pdf_part = {
                "inline_data": {
                    "mime_type": "application/pdf",
                    "data": pdf_b64
                }
            }

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[pdf_part, f"Run all 11 checks on the resume '{file_name}'."],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=CandidateAudit,
                    temperature=0,
                ),
            )
            a = CandidateAudit.model_validate_json(response.text)
            statuses = {c.id: c.status for c in a.checks}

            summary_rows.append({
                "File Name": file_name,
                "Name": a.full_name,
                "Email": a.email,
                "Phone": a.phone,
                "Current Title": a.current_title,
                "Current Employer": a.current_employer,
                "Notice Period": a.notice_period,
                "Total Experience": a.total_experience_years,
                "Relevant AI Experience": a.relevant_ai_experience_years,
                "Recommendation": a.recommendation,
                "Reason": a.reason,
                "RED Checks": sum(s == "RED" for s in statuses.values()),
                "AMBER Checks": sum(s == "AMBER" for s in statuses.values()),
                "Top Skills": ", ".join(a.top_skills),
                "Missing Must-Haves": ", ".join(a.missing_must_haves),
                "Achievements": " | ".join(a.achievements),
                "Discrepancies": " | ".join(a.discrepancies),
                "Interview Questions": " | ".join(a.interview_questions),
            })
            for c in a.checks:
                check_rows.append({
                    "File Name": file_name, "Name": a.full_name,
                    "Check #": c.id, "Check": c.name, "Status": c.status, "Finding": c.finding,
                })

        except Exception as e:
            summary_rows.append({"File Name": file_name, "Name": "Unknown",
                                 "Recommendation": "ERROR", "Reason": f"Failed to audit: {e}"})

    return pd.DataFrame(summary_rows), pd.DataFrame(check_rows)


# ---------------------------------------------------------------------------
# 4. Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Resume Auditor", layout="wide")
st.title("🕵️‍♂️ HR Resume Scanner — AI Engineer")
st.markdown("Upload candidate resumes (PDFs) to run the 11-point audit and download the results as CSV.")

with st.sidebar:
    st.header("Configuration")
    key_type = st.radio("Key type", ["Vertex AI (Express) key", "Google AI Studio key"])
    api_key_input = st.text_input("API Key", type="password")

    st.subheader("Blacklisted Projects")
    default_blacklist = (
        "Project Titan - Online Banking System (Proxy Template)\n"
        "ABC Technologies E-Commerce Management System\n"
        "Global Logistics Tracking Hub (Training Scam)\n"
        "Healthcare Patient Portal v2.0 (Common Copy-Paste Project)"
    )
    blacklist_text = st.text_area("Enter one project per line:", value=default_blacklist, height=200)

uploaded_files = st.file_uploader("Upload PDF Resumes", type=["pdf"], accept_multiple_files=True)

if st.button("Run Audit"):
    if not api_key_input:
        st.error("Please provide an API Key in the sidebar.")
    elif not uploaded_files:
        st.warning("Please upload at least one PDF file.")
    else:
        blacklist_projects = [line.strip() for line in blacklist_text.split("\n") if line.strip()]
        with st.spinner(f"Auditing {len(uploaded_files)} resume(s)..."):
            summary_df, checks_df = process_candidate_resumes(
                api_key_input, key_type.startswith("Vertex"), uploaded_files, blacklist_projects
            )

        st.success("Audit Complete!")

        st.subheader("Candidate Summary")
        st.dataframe(summary_df, use_container_width=True)
        st.download_button("📥 Download Summary (CSV)", summary_df.to_csv(index=False).encode("utf-8"),
                           file_name="screen_summary.csv", mime="text/csv")

        if not checks_df.empty:
            st.subheader("Check Details")
            st.dataframe(checks_df, use_container_width=True)
            st.download_button("📥 Download Check Details (CSV)", checks_df.to_csv(index=False).encode("utf-8"),
                               file_name="screen_check_details.csv", mime="text/csv")
