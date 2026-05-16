from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from google import genai
import fitz  # PyMuPDF
import os
import json
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

@app.post("/analyze")
async def analyze_resume(
    file: UploadFile = File(...),
    job_description: str = Form("")
):
    file_bytes = await file.read()

    if file.filename.endswith(".pdf"):
        resume_text = extract_text_from_pdf(file_bytes)
    else:
        resume_text = file_bytes.decode("utf-8")

    jd_section = f"\n\nJob Description:\n{job_description}" if job_description.strip() else ""

    prompt = f"""
You are an expert resume reviewer and ATS specialist. Analyze the following resume and return a JSON response only.

Resume:
{resume_text}
{jd_section}

Return ONLY a valid JSON object with this exact structure:
{{
  "score": <number 0-100>,
  "score_title": "<one line rating like 'Strong Resume!' or 'Needs Work'>",
  "score_description": "<2 sentence summary of overall quality>",
  "found_skills": ["skill1", "skill2"],
  "missing_skills": ["skill1", "skill2"],
  "ats_checks": [
    {{"label": "Contact Info", "pass": true}},
    {{"label": "LinkedIn", "pass": false}},
    {{"label": "GitHub", "pass": true}},
    {{"label": "Projects Section", "pass": true}},
    {{"label": "Skills Section", "pass": true}},
    {{"label": "Certifications", "pass": false}},
    {{"label": "Education", "pass": true}},
    {{"label": "Experience/Internship", "pass": false}}
  ],
  "jd_match": <number 0-100 or null if no JD provided>,
  "jd_matched_skills": ["skill1", "skill2"],
  "suggestions": [
    {{"type": "good", "text": "something positive"}},
    {{"type": "warn", "text": "something to improve"}},
    {{"type": "bad", "text": "something missing"}}
  ]
}}

Be honest, specific, and helpful. Return ONLY the JSON, no markdown, no explanation.
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    result_text = response.text.strip()

    if result_text.startswith("```"):
        result_text = result_text.split("```")[1]
        if result_text.startswith("json"):
            result_text = result_text[4:]

    result = json.loads(result_text)
    return result