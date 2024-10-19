from fastapi import File, UploadFile, Form


@app.post("/search")
async def search(text: str = Form(...), file: UploadFile = File(...), component: str = Form(...)):
    if file is None:
        return {"text": text, "file": "No file uploaded"}
    pass


@app.post("/upload")
async def upload(text: str = Form(...), file: UploadFile = File(...), token_id: str = Form(...)):
    content = await file.read()
    pass
