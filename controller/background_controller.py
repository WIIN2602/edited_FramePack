import os
from fastapi.responses import FileResponse, JSONResponse
import asyncio
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, BackgroundTasks, Request
from typing import List, Optional, Union

import supabase
from services import *
from dbcontroller.supabase import SupabaseConnector
from fastapi import APIRouter
import traceback

backgroundStream = APIRouter()

# Create a Supabase connector instance
supabase = SupabaseConnector()

async def background_audio_to_summary_script(session_id: str,scripts: List[UploadFile],background_tasks: BackgroundTasks,language: str = Form(...)):
    try:
        task = []

        if scripts:
            task.append(asyncio.create_task(upload_script_svc(session_id, script=scripts)))
        else:
            task.append(asyncio.create_task(transcribe_svc_v2(session_id, lang=language, diarize=False)))
        # task.append(asyncio.create_task(asyncio.to_thread(chunk_svc, session_id)))
        transcript = await asyncio.gather(*task)
        transcript = ' '.join(transcript[0]['data'])
        print("[INFO]: transcript:", transcript)
        print("[INFO]: started ASR & chunk")
        await asyncio.gather(*task)
        print("[INFO]: finished ASR & chunk")
        print("[INFO]: started LLM summarize")
        summerized = summarize_svc(session_id)
        print("[INFO]: done LLM summarized")
        print("[INFO]: started voice API")
        audio_path, audio_url = gen_voice_svc(session_id, 544, False)
        print("[INFO]: done voice API")
        background_tasks.add_task(clear_svc, session_id=session_id, remove_voice=False)

        supabase.insert_result(session_id, transcript, summerized, audio_url)
        supabase.update_status(session_id, "success")  
        
    except Exception as e:
        error_message = f"[ERROR] background_audio_to_summary_script full_process failed: {str(e)}"
        print(error_message)
        traceback.print_exc()
        supabase.log_to_supabase(session_id, "error", error_message)
        supabase.update_status(session_id, "fail")


async def background_video_to_summary_video(session_id: str, scripts: List[UploadFile], language: str):
    try:
        task = []
        if scripts is None:
            task.append(asyncio.create_task(transcribe_svc_v2(session_id, lang=language, diarize=False)))
        else:
            task.append(asyncio.create_task(upload_script_svc(session_id, script=scripts)))

        task.append(asyncio.create_task(asyncio.to_thread(chunk_svc, session_id)))
        await asyncio.gather(*task)
        summarize_svc(session_id)
        await get_vqa_svc(session_id)
        audio_path, audio_url = gen_voice_svc(session_id, 149, False)
        auto_match_list = auto_match_svc(session_id)
        selected_list = [scene["id"] for scene in auto_match_list]
        select_scenes_svc(session_id, selected_list)
        
        ret, scene_id = render_video_svc(session_id, True)
        #clear_svc(session_id)
        if ret in [1, 2, 3, 4]:
            supabase.update_status(session_id, "fail")
            return

        supabase.update_status(session_id, "success")

    except Exception as e:
        error_message = f"[ERROR] background_video_to_summary_video failed: {str(e)}"
        print(error_message)
        traceback.print_exc()
        supabase.log_to_supabase(session_id, "error", error_message)
        supabase.update_status(session_id, "fail")


@backgroundStream.post("/background/video/summary-video")
async def video_to_summary_video(
    background_tasks: BackgroundTasks,
    videos: List[UploadFile] = File(...),
    scripts: Optional[Union[List[UploadFile], None]] = File(None, description="Optional script files"),
    language: str = Form(...)
):  
    session_id = create_session_svc()
    supabase.update_status(session_id, "processing")
    print("[DEBUG] uploading videos:", videos)
    await upload_videos_svc(session_id, videos)
    # background_tasks.add_task(lambda: asyncio.run(await full_process(session_id, videos, scripts, language)))
    background_tasks.add_task(background_video_to_summary_video,session_id =session_id, scripts=scripts, language=language)

    # await full_process(session_id, videos, scripts, language)
    return {"session_id": session_id}   


@backgroundStream.post("/background/audio/summary-script")
async def audio_to_summary_script(
    background_tasks: BackgroundTasks, 
    audios: List[UploadFile] = File(..., description="Required video files"), 
    scripts: Optional[Union[List[UploadFile], None]] = File(None, description="Optional script files"), 
    language: str = Form(...)
):
    print("[INFO]: started session")
    session_id = create_session_svc()
    print(f"[INFO]: created session: {session_id}")
    print("[INFO]: started upload")
    await upload_audios_svc(session_id, audios)

    # upload_audios_svc(session_id, audios)
    print("[INFO]: uploaded audio")
    background_tasks.add_task(
        background_audio_to_summary_script,
        session_id,
        scripts,
        background_tasks,
        language
    )

    return {"session_id": session_id}

#-------------------Endpoint Category---------------------------#

async def background_audio_to_summary_script_category(
    session_id: str,
    scripts: List[UploadFile],
    background_tasks: BackgroundTasks,
    language: str,
    category: str  
):  
    """
    Process audio summarization using category-specific prompt for captioning and scene selection.

    This function handles script upload or audio transcription, performs summarization using
    the specified category prompt, generates voice output, and prepares the result for final use.

    Parameters:
        session_id (str): Unique identifier for the user session.
        scripts (List[UploadFile]): Optional uploaded script files. If not provided, transcription will be used.
        background_tasks (BackgroundTasks): FastAPI background task handler.
        language (str): Language code for transcription (e.g., 'th', 'en').
        category (str): Prompt category for LLM summarization (e.g., 'news', 'normal', 'movie').

    Returns:
        None
    """
    try:
        task = []

        if scripts:
            task.append(asyncio.create_task(upload_script_svc(session_id, script=scripts)))
        else:
            task.append(asyncio.create_task(transcribe_svc_v2(session_id, lang=language, diarize=False)))

        transcript = await asyncio.gather(*task)
        transcript = ' '.join(transcript[0]['data'])
        print("[INFO]: transcript:", transcript)

        print("[INFO]: started ASR & chunk")
        await asyncio.gather(*task)
        print("[INFO]: finished ASR & chunk")

        print(f"[INFO]: started LLM summarize with category '{category}'")
        summerized = summarize_svc(session_id, category=category)  
        print("[INFO]: done LLM summarized")

        print("[INFO]: started voice API")
        audio_path, audio_url = gen_voice_svc(session_id, 544, False)
        print("[INFO]: done voice API")

        background_tasks.add_task(clear_svc, session_id=session_id, remove_voice=False)
        supabase.insert_result(session_id, transcript, summerized, audio_url)
        supabase.update_status(session_id, "success")  

    except Exception as e:
        error_message = f"[ERROR] background_audio_to_summary_script_category failed: {str(e)}"
        print(error_message)
        traceback.print_exc()
        supabase.log_to_supabase(session_id, "error", error_message)
        supabase.update_status(session_id, "fail")

async def background_video_to_summary_video_category(
    session_id: str,
    scripts: List[UploadFile],
    language: str,
    category: str
):
    """
    Process video summarization using category-specific prompt for captioning and scene selection.

    Parameters:
    session_id (str): Unique identifier for the user session.
    scripts (List[UploadFile]): Optional uploaded scripts to use instead of transcription.
    language (str): Language code for transcription if no script is provided.
    category (str): Prompt category for LLM summarization (e.g., "news", "normal").

    Returns:
    None
    """
    try:
        print("[INFO] Started background_video_to_summary_video_category")
        print(f"[INFO] session_id: {session_id}")
        print(f"[INFO] prompt category: {category}")

        task = []
        if scripts is None:
            print("[INFO] No script uploaded. Starting transcription...")
            task.append(asyncio.create_task(transcribe_svc_v2(session_id, lang=language, diarize=False)))
        else:
            print("[INFO] Script uploaded. Uploading script instead of transcribing...")
            task.append(asyncio.create_task(upload_script_svc(session_id, script=scripts)))

        print("[INFO]: started ASR & chunk")
        task.append(asyncio.create_task(asyncio.to_thread(chunk_svc, session_id)))
        await asyncio.gather(*task)
        print("[INFO]: finished ASR & chunk")

        print(f"[INFO]: started LLM summarize with category '{category}'")
        summarize_svc(session_id, category=category)
        print("[INFO]: done LLM summarized")

        print("[INFO] Starting VQA...")
        await get_vqa_svc(session_id)
        print("[INFO] finished VQA ")

        print("[INFO]: started voice API")
        audio_path, audio_url = gen_voice_svc(session_id, 149, False)
        print("[INFO] finished Voiceover generated.")

        print("[INFO] Running auto-match for scenes...")
        auto_match_list = auto_match_svc(session_id)
        selected_list = [scene["id"] for scene in auto_match_list]
        select_scenes_svc(session_id, selected_list)
        print(f"[INFO] Selected scenes: {selected_list}")

        print("[INFO] Rendering final video...")
        ret, scene_id = render_video_svc(session_id, True)

        if ret in [1, 2, 3, 4]:
            print("[ERROR] Render failed.")
            supabase.update_status(session_id, "fail")
            return

        supabase.update_status(session_id, "success")
        print("[INFO] background_video_to_summary_video_category completed successfully.")

    except Exception as e:
        error_message = f"[ERROR] background_video_to_summary_video_category failed: {str(e)}"
        print(error_message)
        traceback.print_exc()
        supabase.log_to_supabase(session_id, "error", error_message)
        supabase.update_status(session_id, "fail")

@backgroundStream.post("/background/audio/summary-script-category")
async def audio_to_summary_script_category(
    background_tasks: BackgroundTasks,
    audios: List[UploadFile] = File(..., description="Required audio files"), 
    scripts: Optional[List[UploadFile]] = File(None),
    language: str = Form(...),
    category: str = Form(...)  
):
    session_id = create_session_svc()
    await upload_audios_svc(session_id, audios)

    background_tasks.add_task(
        background_audio_to_summary_script_category,  
        session_id, scripts,background_tasks ,language, category
    )

    return {"session_id": session_id}


@backgroundStream.post("/background/video/summary-video-category")
async def video_to_summary_video_category(
    background_tasks: BackgroundTasks,
    videos: List[UploadFile] = File(...),
    scripts: Optional[Union[List[UploadFile], None]] = File(None, description="Optional script files"),
    language: str = Form(...),
    category: str = Form(...)
):  
    session_id = create_session_svc()
    supabase.update_status(session_id, "processing")

    print("[DEBUG] session_id:", session_id)
    print("[DEBUG] uploading videos:", videos)
    print("[DEBUG] using prompt category:", category)

    await upload_videos_svc(session_id, videos)
    # background_tasks.add_task(lambda: asyncio.run(await full_process(session_id, videos, scripts, language)))
    background_tasks.add_task(
        background_video_to_summary_video_category,
        session_id=session_id,
        scripts=scripts,
        language=language,
        category=category
    )

    # await full_process(session_id, videos, scripts, language)
    return {"session_id": session_id}

@backgroundStream.get("/status/{session_id}")
async def check_processing_status(session_id: str, request: Request):
    """
    Check the processing status for a session.
    Returns status information and relevant media URLs.
    """
    print(f"[DEBUG] check_processing_status: Checking status for session {session_id}")
    
    # Get base URL from request headers
    host = request.headers.get('x-forwarded-host', request.headers.get('host', 'localhost'))
    proto = request.headers.get('x-forwarded-proto', 'http')
    base_url = f"{proto}://{host}"
    
    try:
        # Rest of the code remains the same until the response building sections
        status_response = supabase.client.table("status").select("*").eq("session_id", session_id).execute()
        
        if not status_response.data:
            print(f"[WARNING] check_processing_status: No status record found for session {session_id}")
            return JSONResponse(
                status_code=404,
                content={
                    "code" : 100,
                    "session_id": session_id,
                    "status": "unknown",
                    "message": "No status information found for this session"
                }
            )
        
        status_data = status_response.data[0]
        status = status_data.get("status", "unknown")
        print(f"[DEBUG] check_processing_status: Status for session {session_id} is '{status}'")
        
        if status == "processing":
            print(f"[DEBUG] check_processing_status: Session {session_id} is still processing")
            progress = status_data.get("progress", 0)
            return JSONResponse(
                status_code=200,
                content={
                    "code" : 200,
                    "session_id": session_id,
                    "status": "processing",
                    "message": "Your request is still being processed",
                    "progress": progress
                }
            )
        
        elif status == "fail":
            print(f"[DEBUG] check_processing_status: Session {session_id} processing failed")
            logs_response = supabase.client.table("session_logs").select("*").eq("session_id", session_id).eq("log_type", "error").execute()
            error_logs = logs_response.data if logs_response.data else []
            
            print(f"[DEBUG] check_processing_status: Found {len(error_logs)} error logs")
            return JSONResponse(
                status_code=100,
                content={
                    "code": 100,
                    "session_id": session_id,
                    "status": "fail",
                    "message": "Processing failed",
                    "errors": [log.get("message") for log in error_logs]
                }
            )
        
        elif status == "success":
            print(f"[DEBUG] check_processing_status: Session {session_id} processing succeeded")
            
            response_data = {
                "code"  : 200,  
                "session_id": session_id,
                "status": "success",
                "message": "Processing completed successfully"
            }
            
            summary_video = supabase.find_summary(session_id)
            if summary_video:
                print(f"[DEBUG] check_processing_status: Found summary video with UID {summary_video['uid']}")
                response_data["video_url"] = f"{base_url}/video/{session_id}"
            
            result_response = supabase.client.table("result").select("*").eq("session_id", session_id).execute()
            result_data = result_response.data[0] if result_response.data else None
            
            if result_data and result_data.get("audio_url"):
                print(f"[DEBUG] check_processing_status: Found audio URL in result data")
                response_data["audio_url"] = result_data.get("audio_url")
            else:
                print(f"[DEBUG] check_processing_status: Using default audio URL path")
                response_data["audio_url"] = f"{base_url}/audio/{session_id}"
            
            if result_data:
                print(f"[DEBUG] check_processing_status: Adding text results from result data")
                if result_data.get("transcribe_text"):
                    response_data["transcribe_text"] = result_data.get("transcribe_text")
                if result_data.get("sumarized_text"):
                    response_data["summary_text"] = result_data.get("sumarized_text")
            
            print(f"[DEBUG] check_processing_status: Returning success response with {len(response_data)} fields")
            return JSONResponse(
                status_code=200,
                content=response_data
            )
        
        print(f"[DEBUG] check_processing_status: Unknown status '{status}' for session {session_id}")
        return {
            "session_id": session_id,
            "status": status,
            "message": "Unknown status"
        }
    
    except Exception as e:
        error_message = f"[ERROR] check_processing_status: Failed to check status for session {session_id}: {str(e)}"
        print(error_message)
        traceback.print_exc()
        
        return JSONResponse(
            status_code=500,
            content={
                "session_id": session_id,
                "status": "error",
                "message": "Failed to retrieve status information",
                "error": str(e)
            })