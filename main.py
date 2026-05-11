from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import io
from ela import perform_ela
from benfords import perform_benfords
from metadata import perform_metadata
from nlp import perform_nlp
from gnn import perform_gnn

app = FastAPI(
    title="TrustNet API",
    description="AI-powered real-time financial document fraud detection system.",
    version="1.0.0"
)

# Mount the static directory to serve frontend assets (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    # Serve the frontend UI
    return FileResponse("static/index.html")

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Read the file content into memory (we will need this for the different layers)
    content = await file.read()
    file_size = len(content)
    
    # --- Layer 2: Error Level Analysis (ELA) ---
    ela_score = perform_ela(content)

    # --- Layer 3: Benford's Law ---
    benfords_result = perform_benfords(content, filename=file.filename)
    benfords_score = benfords_result["score"]

    # --- Layer 4: Metadata Extraction ---
    metadata_result = perform_metadata(content, filename=file.filename)
    metadata_score = metadata_result["score"]

    # --- Layer 5: NLP Text Anomaly Scoring ---
    nlp_result = perform_nlp(content, filename=file.filename)
    nlp_score  = nlp_result["score"]

    # --- Layer 6: Graph Network Analysis ---
    gnn_result = perform_gnn(content, filename=file.filename)
    gnn_score  = gnn_result["score"]

    # --- Unified Risk Score (weighted average of all five layers) ---
    active_scores = [ela_score, benfords_score, metadata_score, nlp_score, gnn_score]
    unified_risk_score = int(sum(active_scores) / len(active_scores))

    # --- Build explanation ---
    if unified_risk_score >= 70:
        explanation = "HIGH RISK: Multiple fraud signals detected. Manual review strongly recommended."
    elif unified_risk_score >= 40:
        explanation = "MEDIUM RISK: Some anomalies detected. Further investigation advised."
    else:
        explanation = "LOW RISK: No strong fraud signals detected across active analysis layers."

    response = {
        "filename": file.filename,
        "file_size_bytes": file_size,
        "unified_risk_score": unified_risk_score,
        "explanation": explanation,
        "layer_scores": {
            "ela": ela_score,
            "benfords_law": {
                "score": benfords_score,
                "digit_count": benfords_result["digit_count"],
                "chi_square": benfords_result["chi_square"],
                "message": benfords_result["message"]
            },
            "metadata": {
                "score":   metadata_score,
                "flags":   metadata_result["flags"],
                "message": metadata_result["message"],
                "details": metadata_result["details"],
            },
            "nlp": {
                "score":   nlp_score,
                "flags":   nlp_result["flags"],
                "message": nlp_result["message"],
                "details": nlp_result["details"],
            },
            "graph": {
                "score":   gnn_score,
                "flags":   gnn_result["flags"],
                "message": gnn_result["message"],
                "details": gnn_result["details"],
            }
        }
    }

    return JSONResponse(content=response)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
