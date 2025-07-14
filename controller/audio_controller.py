import os
import json

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi import File, UploadFile, Form, Query, HTTPException
from typing import List, Optional, Union
from services import *
from utils import *
import asyncio
from dbcontroller.supabase import SupabaseConnector
from services.audio_service import generate_audio

audioStream = APIRouter()

# Create a Supabase connector instance
supabase = SupabaseConnector()


@audioStream.post("/upload-script")
async def upload_script(
    session_id: str = Form(...),
    script: List[UploadFile] = File(...),
):
    # Log the start of script upload
    supabase.add_or_update_section(session_id, "processing", "uploading_script")
    
    await upload_script_svc(session_id, script)

    # Update status after successful upload
    supabase.add_or_update_section(session_id, "processing", "script_uploaded")
    return JSONResponse(content={"message": "script uploaded successfully"})


@audioStream.post("/summarize-script")
def summarize_script(session_id: str = Form(...)):
    # Log the start of the summarization process
    supabase.add_or_update_section(session_id, "processing", "summarizing_script")
    
    sum_text = summarize_svc(session_id)

    # Mark script as summarized in Supabase
    supabase.add_or_update_section(session_id, "processing", "script_summarized")
    return JSONResponse(content={"message": "script summarized successfully", "sumarized_text": sum_text})

@audioStream.post("/transcribe-audio/")
async def transcribe_audio(session_id: str = Query(...), language: str = 'th', diarize: bool = Query(...)):
    try:
        # Log the start of the transcription process
        supabase.add_or_update_section(session_id, "processing", "transcribing_audio")
        
        transcript = await transcribe_svc_v2(session_id, lang=language, diarize=diarize)

        # Update status after successful transcription
        supabase.add_or_update_section(session_id, "processing", "audio_transcribed")
        return JSONResponse(content={"message": "Transcription successful", "transcript": transcript})
    except FileNotFoundError as e:
        supabase.log_to_supabase(
            session_id, "error", f"File not found: {str(e)}")
        return JSONResponse(status_code=404, content={"message": str(e)})
    except Exception as e:
        supabase.log_to_supabase(
            session_id, "error", f"Transcription failed: {str(e)}")
        return JSONResponse(status_code=500, content={"message": str(e)})


@audioStream.get("/generate-voice/")
def generate_voice(
    session_id: str = Query(...),
    voice_id: str = Query(...),
    original_script: bool = Query(...)
):
    audio_path, audio_url = gen_voice_svc(session_id, voice_id, original_script)
    return FileResponse(audio_path, media_type="audio/wav", filename=f"{session_id}.wav")


@audioStream.get("/get-audio-duration/")
def get_audio_duration(session_id: str = Query(None)):
    try:
        audio_path = os.path.join(SERVICE_DIR, VOICE_DIR, f"{session_id}.wav")

        if not os.path.exists(audio_path):
            supabase.log_to_supabase(
                session_id, "error", "Audio file not found")
            return JSONResponse(status_code=404, content={"message": "Audio file not found"})

        audio_duration = get_audio_length_svc(audio_path)

        # Store duration in Supabase
        supabase.add_sum_duration(session_id, audio_duration)

        return JSONResponse(content={"message": audio_duration})
    except FileNotFoundError as e:
        supabase.log_to_supabase(
            session_id, "error", f"File not found: {str(e)}")
        return JSONResponse(status_code=404, content={"message": str(e)})
    except Exception as e:
        supabase.log_to_supabase(
            session_id, "error", f"Failed to get audio duration: {str(e)}")
        return JSONResponse(status_code=500, content={"message": str(e)})


@audioStream.post("/summarize-audio")
async def summarize_audio(
    background_tasks: BackgroundTasks,
    audios: List[UploadFile] = File(..., description="Required video files"),
    scripts: Optional[Union[List[UploadFile], None]] = File(
        None, description="Optional script files"),
    language: str = Form(...)
):
    print("[INFO]: started session")
    session_id = create_session_svc()
    supabase.update_status(session_id, "processing")
    supabase.add_or_update_section(session_id, "processing", "session_created")
    print(f"[INFO]: created session: {session_id}")

    print("[INFO]: started upload")
    supabase.add_or_update_section(session_id, "processing", "uploading_audio")
    await upload_audios_svc(session_id, audios)
    supabase.add_or_update_section(session_id, "processing", "audio_uploaded")
    print("[INFO]: uploaded audio")

    print("[INFO]: audios:", audios)
    print("[INFO]: scripts:", scripts)

    task = []

    if scripts:
        supabase.add_or_update_section(session_id, "processing", "uploading_script")
        task.append(asyncio.create_task(upload_script_svc(session_id, script=scripts)))
    else:
        supabase.add_or_update_section(session_id, "processing", "transcribing_audio")
        task.append(asyncio.create_task(transcribe_svc_v2(session_id, lang=language, diarize=False)))

    # task.append(asyncio.create_task(asyncio.to_thread(chunk_svc, session_id)))
    transcript = await asyncio.gather(*task)
    transcript = ' '.join(transcript[0]['data'])

    print("[INFO]: transcript:", transcript)

    print("[INFO]: started ASR & chunk")
    supabase.add_or_update_section(session_id, "processing", "start_ASR_&_chunk")
    await asyncio.gather(*task)
    supabase.add_or_update_section(session_id, "processing", "end_ASR_&_chunk")
    print("[INFO]: finished ASR & chunk")

    print("[INFO]: started LLM summarize")
    supabase.add_or_update_section(session_id, "processing", "start_summarize_svc")
    summerized = summarize_svc(session_id)
    supabase.done_llm(session_id)  # Mark script as summarized in Supabase
    supabase.add_or_update_section(session_id, "processing", "end_summarize_svc")
    print("[INFO]: done LLM summarized")

    print("[INFO]: started voice API")
    supabase.add_or_update_section(session_id, "processing", "start_gen_voice_svc")
    audio_path, audio_url = gen_voice_svc(session_id, 544, False)
    supabase.add_or_update_section(session_id, "processing", "end_gen_voice_svc")
    print("[INFO]: done voice API")

    # Store results in Supabase
    supabase.insert_result(session_id, transcript, summerized, audio_url)
    
    supabase.add_or_update_section(session_id, "processing", "cleaning_up")
    background_tasks.add_task(clear_svc, session_id=session_id, remove_voice=False)

    match summerized:
        case 1:
            supabase.update_status(session_id, "fail")
            supabase.log_to_supabase(
                session_id, "error", f"Summary with id {session_id} not found")
            raise HTTPException(
                status_code=404, detail=f"Summary with id {session_id} not found.")
        case 2:
            supabase.update_status(session_id, "fail")
            supabase.log_to_supabase(session_id, "error", f"Scene not found")
            raise HTTPException(
                status_code=404, detail=f"Scene named scene_id.mp4 not found.")
        case 3:
            supabase.update_status(session_id, "fail")
            supabase.log_to_supabase(
                session_id, "error", f"Path does not exist")
            raise HTTPException(
                status_code=404, detail=f"Path to scene_id.mp4 does not exist.")
        case 4:
            supabase.update_status(session_id, "fail")
            supabase.log_to_supabase(
                session_id, "error", "Failed to create AudioFileClip")
            raise HTTPException(
                status_code=400, detail="Fail to create AudioFileClip.")
        case _:
            return JSONResponse(
                content={
                    "message": "script summarized successfully",
                    "session_id": session_id,
                    "transcribe_text": transcript,
                    "sumarized_text": summerized,
                    "audio_url": audio_url
                }
            )


@audioStream.get("/audio/{session_id}")
async def get_audio_media_result(session_id: str):
    """
    Retrieve a video or voice file based on session_id.
    - `file_type`: "video" or "voice"
    - Stream the file in the browser instead of downloading.
    """
    print(f"Checking for session_id: {session_id}")
    media_files = os.path.join(SERVICE_DIR, VOICE_DIR, f"{session_id}.wav")
    extension = ".wav"
    media_type = "audio/wav"

    print(f"[INFO] Looking in file: {media_files}")

    if not os.path.exists(media_files):
        raise HTTPException(
            status_code=400, detail=f"No media files found in {media_files}")

    return FileResponse(media_files, media_type=media_type, headers={"Content-Disposition": "attachment"})

@audioStream.post("/audio/transcibe_audio")
async def get_transcibe_audio(
    background_tasks: BackgroundTasks,
    audios: List[UploadFile] = File(..., description="Required video files"),
    language: str = Form(...)
):
    print("[INFO]: started session")
    session_id = create_session_svc()
    print(f"[INFO]: created session: {session_id}")

    print("[INFO]: started upload")
    # await upload_audios_svc(session_id, audios)
    await upload_audios_svc(session_id, audios)
    print("[INFO]: uploaded audio")

    task = []

    task.append(asyncio.create_task(transcribe_svc_v2(
        session_id, lang=language, diarize=False)))

    # task.append(asyncio.create_task(asyncio.to_thread(chunk_svc, session_id)))
    transcript = await asyncio.gather(*task)
    transcript = ' '.join(transcript[0]['data'])

    print("[INFO]: transcript:", transcript)
    background_tasks.add_task(clear_svc, session_id=session_id, remove_voice=False)
    supabase.add_or_update_section(session_id, "processing", "end_transcibe_audio")
    return JSONResponse(
        content={
            "message": "transcribe_audio successfully",
            "session_id": session_id,
            "transcribe_text": transcript,
        }
    )


@audioStream.post("/script/summarize-script")
async def summarize_text(
    background_tasks: BackgroundTasks,
    scripts: Union[List[UploadFile],
                   UploadFile] = File(..., description="Required script file(s)"),
    language: str = Form(...),
):
    print("[INFO]: started session")
    session_id = create_session_svc()
    print(f"[INFO]: created session: {session_id}")

    print("[INFO]: scripts:", scripts)

    task = []
    if isinstance(scripts, list):
        task.append(asyncio.create_task(
            upload_script_svc(session_id, script=scripts)))
        transcript = await asyncio.gather(*task)
        transcript = ' '.join(transcript[0]['data'])
    else:
        task.append(asyncio.create_task(
            upload_script_svc(session_id, script=[scripts])))
        transcript = await asyncio.gather(*task)
    print("[INFO]: transcript:", transcript)

    print("[INFO]: started LLM summarize")
    summerized = summarize_svc(session_id)
    print("[INFO]: done LLM summarized")

    if summerized == 1:
        raise HTTPException(status_code=404, detail=f"Summary with id {session_id} not found.")
    supabase.add_or_update_section(session_id, "processing", "end_summarize_text")
    return {
        "message": "script summarized successfully",
        "session_id": session_id,
        "transcribe_text": transcript,
        "sumarized_text": summerized,
    }


@audioStream.post("/script/botnoi-voice")
async def gen_botnoi_voice(
    voice_id: str = Form(...),
    script: str = Form(...),
):
    print("[INFO]: started session")
    session_id = create_session_svc()
    print(f"[INFO]: created session: {session_id}")
    print("[INFO]: Generating Botnoi voice from user input script")
    audio_data = generate_audio(
        text=script, voice_id=voice_id, session_id=session_id)

    if not audio_data:
        raise HTTPException(
            status_code=500, detail="Failed to generate audio.")
    audio_path, audio_url = audio_data
    return FileResponse(audio_path, media_type="audio/wav", filename=f"{session_id}.wav")
