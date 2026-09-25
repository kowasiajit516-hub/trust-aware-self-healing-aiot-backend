from fastapi import APIRouter, HTTPException, status
from google.oauth2 import id_token
from google.auth.transport import requests
import os

router = APIRouter()

@router.post("/auth/google")
async def google_auth(payload: dict):
    token = payload.get("credential") or payload.get("token")
    if not token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing Google token in request payload"
        )

    try:
        # Verify the ID token against Google's OAuth servers
        client_id = os.getenv("GOOGLE_CLIENT_ID")
        id_info = id_token.verify_oauth2_token(token, requests.Request(), client_id)
        
        # User verified successfully
        user_email = id_info.get("email")
        user_name = id_info.get("name")
        return {"status": "success", "email": user_email, "name": user_name}

    except ValueError as e:
        # Token invalid or expired
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid Google token: {str(e)}"
        )
    except Exception as e:
        # Print exception to console and return HTTP 500 with detail
        print(f"OAuth error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication failed: {str(e)}"
        )