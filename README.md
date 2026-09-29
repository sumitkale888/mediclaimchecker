Medical Claim Checker
A Python FastAPI project that checks medical claims using RAG.
The first pipeline searches 14K+ medical facts using ChromaDB.
The second pipeline searches PubMed for relevant research papers.
The retrieved evidence is given to Groq LLaMA to generate the final result.
How to run
1. Clone the project.
2. Open the project in VS Code.
3. Create a virtual environment:
   python -m venv venv
4. Activate it:
   venv\Scripts\activate
5. Install the required packages:
   pip install -r requirements.txt
6. Create a .env file in the project folder.
7. Add your Groq API key:
   GROQ_API_KEY=your_key
8. Add your NCBI email:
   NCBI_EMAIL=your_email
9. Add your NCBI API key:
   NCBI_API_KEY=your_key
10. Add:
    NCBI_TOOL=MedicalClaimChecker
11. Make sure the medical facts dataset is available.
12. Make sure the ChromaDB data is available.
13. Start the server:
    uvicorn app.main:app --reload
14. Open http://localhost:8000/docs
15. Use POST /api/v1/verify for the medical-facts pipeline.
16. Enter a claim in the request body.
17. Use POST /api/v1/pubmed/verify for the PubMed pipeline.
18. Enter a medical claim in the request body.,
19. Pipeline 1 retrieves relevant facts from ChromaDB.
20. Pipeline 2 retrieves relevant research papers from PubMed and sends the evidence to the LLM.