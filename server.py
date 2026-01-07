from fastapi import FastAPI
from models.models import UserRequest

# To run the server:
# uvicorn server:app --reload
# or
# python server.py

app = FastAPI(
    title="Tandemn Server",
    description="API for receiving job requests",
)


@app.post("/request")
async def submit_request(request: UserRequest):
    """
    Submit a job request.
    
    Receives a UserRequest with job configuration and returns
    confirmation of receipt.
    """
    print(type(request))
    print(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=26336)