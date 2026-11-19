
import os
import secrets
from pathlib import Path
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from jose import jwt, JWTError
from pwdlib import PasswordHash

load_dotenv(Path(__file__).resolve().parent / ".env")

SECRET_KEY = os.getenv("SECRET_KEY")

if not SECRET_KEY:
    # Avoid a known fallback secret; sessions last only until this process restarts.
    SECRET_KEY = secrets.token_urlsafe(48)

ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7

password_hash = PasswordHash.recommended()


# ==========================================
# PASSWORD FUNCTIONS
# ==========================================

def hash_password(password: str) -> str:

    return password_hash.hash(password)


def verify_password(
    password: str,
    hashed_password: str
) -> bool:

    return password_hash.verify(
        password,
        hashed_password
    )


# ==========================================
# TOKEN FUNCTIONS
# ==========================================

def create_access_token(user_id: int) -> str:

    expires = datetime.now(timezone.utc) + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "sub": str(user_id),
        "exp": expires
    }

    return jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=ALGORITHM
    )


def decode_access_token(token: str):

    try:

        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        user_id = payload.get("sub")

        if user_id is None:
            return None

        return int(user_id)

    except (JWTError, ValueError):

        return None
