from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io
import PyPDF2

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

# Health Check Route for UptimeRobot
@app.get("/")
def health_check():
    return {"status": "Alive", "message": "Rizzume ML Engine is running perfectly! 🚀"}

@app.post("/api/extract")
async def extract_data(
    file: UploadFile = File(...), 
    job_description: str = Form("Software Engineer") # Default fallback
):
    try:
        print(f"📥 Receiving file: {file.filename} for job: {job_description}")
        
        # 1. Read the PDF File from Buffer
        pdf_bytes = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        
        # 2. Extract Text from PDF Pages
        resume_text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                resume_text += extracted + "\n"
                
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not read text from this PDF.")

        print("🧠 Analyzing text with ML Brain...")
        
        # 3. Run ML Analysis
        ats_score = calculate_ats_score(resume_text, job_description)
        skills = analyze_skills(resume_text, job_description)
        
        # 4. Return EXACT format expected by Node.js analyze.js
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
    print("🚀 Starting ML API on Port 8000...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
