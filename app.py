import os
import requests
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field 
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader

# Load environment variables
load_dotenv()

# Initialize components
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # <-- allow all origins
    allow_credentials=True,         # <-- allow cookies / auth headers
    allow_methods=["*"],            # <-- allow GET, POST, PUT, etc.
    allow_headers=["*"],            # <-- allow all headers
)

# --- Pydantic Models ---
class BasicInfo(BaseModel):
    city: str
    state: str
    residentialAddress1: str
    residentialAddress2: str
    zipCode: str

class EducationInfo(BaseModel):
    educationLevel: str
    schoolName: str
    startYear: str
    endYear: str
    country: str
    schoolAddress: str

class ProfessionInfo(BaseModel):
    profession: str
    skillLevel: str
    experience: str
    additionalSkills: List[str]
    portfolio: str

class Profile(BaseModel):
    firstName: str
    lastName: str
    country: str
    basicInfo: BasicInfo
    educationInfo: List[EducationInfo]
    professionInfo: ProfessionInfo

#class BudgetRange(BaseModel):
    #min: int
    #max: int

class Job(BaseModel):
    id: str = Field(alias='_id')
    title: str
    budgetMin: int
    budgetMax: int
    keySkills: List[str]
    projectType: str

class RequestData(BaseModel):
    profile: Profile
    jobs: List[Job]

# --- Helper Functions ---
def extract_resume_text(url: str) -> str:
    """Extract text from PDF or DOC resume"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Determine file type
        file_ext = url.split('.')[-1].lower()
        temp_file = f"temp.{file_ext}"
        
        with open(temp_file, 'wb') as f:
            f.write(response.content)
        
        if file_ext == 'pdf':
            loader = PyPDFLoader(temp_file)
        elif file_ext in ['doc', 'docx']:
            loader = UnstructuredWordDocumentLoader(temp_file)
        else:
            raise ValueError("Unsupported file format")
        
        pages = loader.load()
        os.remove(temp_file)
        return " ".join([p.page_content for p in pages])
    
    except Exception as e:
        print(f"Error processing resume: {str(e)}")
        return ""

def process_profile(profile: Profile) -> List[str]:
    """Process profile data into text chunks"""
    # Basic Info
    base_info = f"""
    Name: {profile.firstName} {profile.lastName}
    Country: {profile.country}
    Location: {profile.basicInfo.city}, {profile.basicInfo.state}
    Address: {profile.basicInfo.residentialAddress1} {profile.basicInfo.residentialAddress2}
    Zip Code: {profile.basicInfo.zipCode}
    """
    
    # Education Info
    education_text = "\n".join([
        f"Education: {edu.educationLevel} at {edu.schoolName} ({edu.startYear}-{edu.endYear})"
        for edu in profile.educationInfo
    ])
    
    # Professional Info
    profession_text = f"""
    Profession: {profile.professionInfo.profession}
    Experience: {profile.professionInfo.experience}
    Skills: {', '.join(profile.professionInfo.additionalSkills)}
    """
    
    # Resume Text
    resume_text = extract_resume_text(profile.professionInfo.portfolio)
    
    # Combine and split
    full_text = "\n".join([base_info, education_text, profession_text, resume_text])
    return text_splitter.split_text(full_text)

def process_job(job: Job) -> str:
    """Convert job data to text"""
    return f"""
    Job Title: {job.title}
    Budget: ${job.budgetMin}-${job.budgetMax}
    Required Skills: {job.keySkills}
    Project type: {job.projectType}
    """

# --- API Endpoint ---
@app.post("/match-jobs", response_model=List[str])
async def match_jobs(request_data: RequestData):
    # Process profile into chunks
    profile_chunks = process_profile(request_data.profile)
    
    # Generate profile embeddings
    profile_embeddings = [embeddings.embed_query(chunk) for chunk in profile_chunks]
    
    matches = []
    for job in request_data.jobs:
        # Process job data
        job_text = process_job(job)
        job_embedding = embeddings.embed_query(job_text)
        
        # Calculate max similarity across chunks
        max_similarity = max([
            cosine_similarity([p_embed], [job_embedding])[0][0]
            for p_embed in profile_embeddings
        ])
        
        if max_similarity >= 0.65:
            matches.append(job.id)
    
    return matches

# --- Run Server ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)