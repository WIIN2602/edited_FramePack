from supabase import create_client, Client
from dotenv import load_dotenv
import os
from datetime import datetime
# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
def add_user(session_id: str):
    now = datetime.now()
    supabase.table("ai_backend_users").insert({
        "session_id": session_id,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat()
    }).execute()

def add_or_update_section(session_id: str, status: str):
    """
    Ensures one row per session_id.
    Updates status if session_id exists.
    Inserts new row with session_id and status if not.
    """

    # Check if session_id already exists
    existing = supabase.table("status").select("*")\
        .eq("session_id", session_id)\
        .execute()

    if existing.data:
        # Update status
        update_response = supabase.table("status").update({
            "processing_session": status
        }).eq("session_id", session_id).execute()
    else:
        insert_response = supabase.table("status").insert({
            "session_id": session_id,
            "processing_session": status
        }).execute()
def update_status(session_id: str, status: str):
    """
    Ensures one row per session_id.
    Updates status if session_id exists.
    Inserts new row with session_id and status if not.
    """

    # Check if session_id already exists
    existing = supabase.table("status").select("*")\
        .eq("session_id", session_id)\
        .execute()

    if existing.data:
        # Update status
        update_response = supabase.table("status").update({
            "status": status
        }).eq("session_id", session_id).execute()
    else:
        insert_response = supabase.table("status").insert({
            "session_id": session_id,
            "status": status
        }).execute()


def log_to_supabase(session_id: str, log_type: str, message: str):
    now = datetime.now().isoformat()
    supabase.table("session_logs").insert({
        "session_id": session_id,
        "log_type": log_type,
        "message": message,
        "created_at": now
    }).execute()

def insert_result(session_id: str, transcript: str, summerized: str, audio_url: str):
    now = datetime.now()
    supabase.table("result").insert({
        "session_id": session_id,
        "transcribe_text": transcript,
        "sumarized_text": summerized,
        "audio_url": audio_url,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat()
    }).execute()


