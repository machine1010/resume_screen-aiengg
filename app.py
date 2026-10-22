import datetime
from typing import List, Literal

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 1. Pydantic Output Schema
# ---------------------------------------------------------------------------
class AuditCheckItem(BaseModel):
    check_name: str = Field(description="Exact check name from the prompt.")
    outcome: str = Field(description="Short finding with evidence (dates, role or project name).")
    flag_status: Literal["PASS", "PROCEED_WITH_CAUTION", "FLAG_FOR_AUDIT"]


class CandidateResumeAudit(BaseModel):
    candidate_name: str = Field(description="The full name of the candidate extracted from the resume.")
    checks: List[AuditCheckItem] = Field(description="Exactly 12 entries, one per check, in order.")


# ---------------------------------------------------------------------------
# 2. System Instruction Template
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION_TEMPLATE = """
You are an expert HR Forensic Auditor and Resume Authenticity Specialist, screening candidates for an AI Engineer role.
Analyze the candidate's resume PDF. Extract the candidate's full name.
Today's date is {current_date}. Treat "Present" / "Current" as today's date.

ROLE SUMMARY
6-8 years of application development with a focus on AI. Builds internal AI tools and workflows (Claude, Cursor)
for finance/accounting teams, integrated with Snowflake, Slack and Google Drive.

RULES
- Base every finding on the resume. If something is missing, write "NOT_STATED". Do not guess.
- Do not use age, gender, photo, nationality or college name in your judgment.
- Flag Status must be ONLY one of: PASS, PROCEED_WITH_CAUTION, FLAG_FOR_AUDIT.
- In Findings / Outcome, give a short finding with evidence (dates, role or project name).

AUDIT CHECKS TO EXECUTE (use these exact check names, in this order):
1. Contact Info Verification: Extract email and phone and validate their format.
   Both present = PASS, one missing = PROCEED_WITH_CAUTION, both missing = FLAG_FOR_AUDIT.
2. Career Gap Analysis: Identify gaps > 6 months and report the most recent one.
   No gap or explained gap = PASS, unexplained gap in the last 3 years = PROCEED_WITH_CAUTION.
3. Education & Work Experience Overlap: Full-time degree overlapping a full-time job = FLAG_FOR_AUDIT.
   Part-time/distance programmes = PASS. Programme mode not stated = PROCEED_WITH_CAUTION.
4. Company-to-Company Tenure Overlap: Two full-time jobs overlapping by more than 2 months = FLAG_FOR_AUDIT.
5. Career Trajectory & Title Escalation: Unrealistic jumps (e.g., fresher to Lead in under 2 years,
   "Head of AI" with under 2 years of AI work). One jump = PROCEED_WITH_CAUTION, two or more = FLAG_FOR_AUDIT.
6. AI Timeline Check: Flag claims impossible by date - GenAI/LLM experience before Nov 2022, LangGraph before 2024,
   MCP before Nov 2024, Claude Code before 2025. One = PROCEED_WITH_CAUTION, two or more = FLAG_FOR_AUDIT.
7. Employer Check: List employers from the last 5 years. Any you cannot recognise, or with no details
   (location, clients, domain) = PROCEED_WITH_CAUTION for manual check.
8. Blacklisted Project Audit: Compare listed projects against this blacklist:
{blacklisted_projects_str}
   Any match = FLAG_FOR_AUDIT.
9. Experience Range Filter (6 to 8 Years): Total full-time experience (overlaps counted once, internships excluded).
   6-8 years = PASS, 5-6 or 8-10 years = PROCEED_WITH_CAUTION, otherwise FLAG_FOR_AUDIT.
   State total years and relevant AI years in the finding.
10. Recent Skills Application Check (Last 1 Year): A skill counts only if it appears in a project from the last 12 months,
    not just the skills list. Must-have: Python/TypeScript, LLM APIs (Claude, OpenAI, etc.), RAG,
    Function Calling / Structured Outputs, Agents / Workflow Orchestration, AI Evaluation & Guardrails.
    Nice-to-have: Snowflake / Slack / Google Drive integrations, Claude / Cursor, finance automation.
    5-6 must-haves = PASS, 3-4 = PROCEED_WITH_CAUTION, 0-2 = FLAG_FOR_AUDIT.
    In the finding, list the top skills found and the must-haves missing.
11. Project Substance & Achievements: Do AI projects describe what was built and the result (users, metrics, scale)?
    2 or more such projects = PASS, 1 = PROCEED_WITH_CAUTION, none / buzzwords only = FLAG_FOR_AUDIT.
    In the finding, list the top 3 achievements (keep numbers as written).
12. Overall Candidate Risk Score & Recommendation:
    - FLAG_FOR_AUDIT: any check above is FLAG_FOR_AUDIT.
    - PROCEED_WITH_CAUTION: 2 or more checks are PROCEED_WITH_CAUTION.
    - PASS: everything else.
    In the finding, give the deciding reasons in 1-2 sentences.
"""

MODEL_NAME = "gemini-2.5-flash"
OUTPUT_COLUMNS = ["Candidate Name", "File Name", "Check Name", "Findings / Outcome", "Flag Status"]


# ---------------------------------------------------------------------------
# 3. Processing Function (inline PDF bytes - no file upload API needed)
# ---------------------------------------------------------------------------
def process_candidate_resumes(api_key: str, use_vertex: bool, uploaded_files_list: List,
                              blacklisted_projects: List[str]) -> pd.DataFrame:
    client = genai.Client(vertexai=use_vertex, api_key=api_key)

    blacklist_formatted = "\n".join(f"    - {proj}" for proj in blacklisted_projects) or "    - (none)"
    system_instruction = SYSTEM_INSTRUCTION_TEMPLATE.format(
        current_date=datetime.date.today().strftime("%d %B %Y"),
        blacklisted_projects_str=blacklist_formatted,
    )

    rows = []
    for uploaded_file in uploaded_files_list:
        file_name = uploaded_file.name
        try:
            pdf_part = types.Part.from_bytes(data=uploaded_file.getvalue(), mime_type="application/pdf")
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[pdf_part, f"Run all 12 audit checks on the resume '{file_name}'."],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=CandidateResumeAudit,
                    temperature=0,
                ),
            )
            audit = CandidateResumeAudit.model_validate_json(response.text)
            for check in audit.checks:
                rows.append({
                    "Candidate Name": audit.candidate_name,
                    "File Name": file_name,
                    "Check Name": check.check_name,
                    "Findings / Outcome": check.outcome,
                    "Flag Status": check.flag_status,
                })
        except Exception as e:
            rows.append({
                "Candidate Name": "Unknown",
                "File Name": file_name,
                "Check Name": "Processing Error",
                "Findings / Outcome": f"Failed to audit: {e}",
                "Flag Status": "FLAG_FOR_AUDIT",
            })

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


# ---------------------------------------------------------------------------
# 4. Streamlit User Interface
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Resume Auditor", layout="wide")
st.title("🕵️‍♂️ HR Resume Scanner — AI Engineer")
st.markdown("Upload candidate resumes (PDFs) to run the 12-point audit and download the results as a CSV.")

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
        with st.spinner(f"Auditing {len(uploaded_files)} resume(s)... This may take a minute."):
            result_df = process_candidate_resumes(
                api_key_input, key_type.startswith("Vertex"), uploaded_files, blacklist_projects
            )

        st.success("Audit Complete!")
        st.subheader("Audit Results Preview")
        st.dataframe(result_df, use_container_width=True)
        st.download_button(
            label="📥 Download Full Report (CSV)",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="screen_results.csv",
            mime="text/csv",
        )
