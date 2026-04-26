import os
import io
import uvicorn
import PyPDF2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import our ML functions
from ml_core import calculate_ats_score, analyze_skills

app = FastAPI(title="Rizzume ML Engine")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health Check Route (GET and HEAD for UptimeRobot/BetterStack)
@app.api_route("/", methods=["GET", "HEAD"])
def health_check():
    return {"status": "Alive", "message": "Rizzume ML Engine is running perfectly! 🚀"}

@app.post("/api/extract")
async def extract_data(
    file: UploadFile = File(...), 
    job_description: str = Form("Software Engineer")
):
    try:
        print(f"📥 Receiving file: {file.filename} for job: {job_description}")
        
        pdf_bytes = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        
        resume_text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                resume_text += extracted + "\n"
                
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not read text from this PDF.")

        print("🧠 Analyzing text with ML Brain...")
        
        ats_score = calculate_ats_score(resume_text, job_description)
        skills = analyze_skills(resume_text, job_description)
        
        return {
            "status": "success",
            "data": {
                "content": resume_text,
                "ml_analysis": {
                    "ats_score": ats_score,
                    "matched_skills": skills["matched"],
                    "missing_skills": skills["missing"]
                }
            }
        }
        
    except Exception as e:
        print(f"❌ Error during extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Render assigns a dynamic port, so we must use os.environ.get
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting ML API on Port {port}...")
    uvicorn.run("main:app", host="0.0.0.0", port=port)
