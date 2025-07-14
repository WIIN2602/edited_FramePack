import os
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from services import *
import asyncio
from dbcontroller.supabase import SupabaseConnector
from utils.api_config import SERVICE_DIR, VID_SUM_DIR
# from dbcontroller.mongo import DBConnector
videoStream = APIRouter()


# Create a Supabase connector instance
supabase = SupabaseConnector()


@videoStream.get("/create-session")
async def create_session():
    session_id = create_session_svc()
    # Update with initial status in Supabase
    supabase.update_status(session_id, "session_created")
    return JSONResponse(content={"session_id": session_id})

@videoStream.post("/upload-videos")
async def upload_videos(
    session_id: str = Form(...), videos: List[UploadFile] = File(...)
):
    # Log the start of the upload process
    supabase.add_or_update_section(session_id, "processing", "uploading_videos")
    supabase.log_to_supabase(session_id, "info", f"Uploading {len(videos)} videos")
    
    await upload_videos_svc(session_id, videos)
    
    # Update status after successful upload
    supabase.add_or_update_section(session_id, "processing", "videos_uploaded")
    return JSONResponse(content={"message": "Videos uploaded successfully"})

@videoStream.post("/chunk-videos")
def chunk_videos(session_id: str = Form(...)):
    # Log the start of the chunking process
    supabase.add_or_update_section(session_id, "processing", "chunking_videos")
    
    chunk_svc(session_id)
    
    # Update status after successful chunking
    supabase.add_or_update_section(session_id, "processing", "videos_chunked")
    return JSONResponse(content={"message": "Videos chunked successfully"})

# @videoStream.post("/get-vqa")
# async def get_vqa(session_id: str = Form(...)):
#     await get_vqa_svc(session_id)
#     return JSONResponse(content={"message": "Captioned successfully"})


# @videoStream.get("/auto-match/")
# def auto_match(session_id: str = Query(None)):
#     # Log the start of the auto-matching process
#     supabase.add_or_update_section(session_id, "processing", "auto_matching_scenes")
    
#     auto_match_list = auto_match_svc(session_id)
    
#     # Update status after successful auto-matching
#     supabase.add_or_update_section(session_id, "processing", "scenes_auto_matched")
#     return JSONResponse(content={"scenes": auto_match_list})

# @videoStream.get("/search-scenes/")
# def search_scenes(session_id: str, search: str = Query(None)):   
#     try:
#         # Log the search query
#         supabase.log_to_supabase(session_id, "info", f"Searching scenes with query: {search}")
        
#         scene_list = search_svc(session_id, search)
#         return JSONResponse(content={"scenes": scene_list})
#     except Exception as e:
#         supabase.log_to_supabase(session_id, "error", f"Search failed: {str(e)}")
#         return JSONResponse(content={"message": str(e)}, status_code=500)

# @videoStream.post("/select-scenes")
# def select_scenes(session_id: str = Form(...), scenes: List[str] = File(...)):
#     # Log the scene selection
#     supabase.add_or_update_section(session_id, "processing", "selecting_scenes")
#     supabase.log_to_supabase(session_id, "info", f"Selected {len(scenes)} scenes")
    
#     select_scenes_svc(session_id, scenes)
    
#     # Update the selected scenes in Supabase
#     supabase.update_summary(session_id, scenes)
#     supabase.add_or_update_section(session_id, "processing", "scenes_selected")
#     return JSONResponse(content={"message": "Scenes selected successfully"})

# @videoStream.post("/render-video")
# def render_video(session_id: str, replace: bool = Query(False), selected_scenes: str = None):
#     """
#     Render a video based on selected scenes
#     """
#     print(f"[DEBUG] render_video: Starting for session {session_id}")
#     print(f"[DEBUG] render_video: Parameters - replace={replace}, selected_scenes={selected_scenes}")
    
#     # Log the start of the rendering process
#     try:
#         supabase.add_or_update_section(session_id, "processing", "rendering_video", progress=75)
#         print(f"[DEBUG] render_video: Updated section status to 'rendering_video'")
#     except Exception as e:
#         print(f"[ERROR] render_video: Failed to update status in Supabase: {str(e)}")
    
#     # Call the service function to render the video
#     print(f"[DEBUG] render_video: Calling render_video_svc for session {session_id}")
#     ret, scene_id = render_video_svc(session_id, replace)
#     print(f"[DEBUG] render_video: render_video_svc returned code {ret} with info {scene_id}")
    
#     # Handle the return values
#     match ret:
#         case 1:
#             error_msg = f"Summary with id {session_id} not found."
#             print(f"[ERROR] render_video: {error_msg}")
#             supabase.log_to_supabase(session_id, "error", error_msg)
#             raise HTTPException(status_code=404, detail=error_msg)
#         case 2:
#             error_msg = f"No valid scenes found for session {session_id}"
#             print(f"[ERROR] render_video: {error_msg}")
#             supabase.log_to_supabase(session_id, "error", error_msg)
#             raise HTTPException(status_code=404, detail=error_msg)
#         case 3:
#             error_msg = f"Source video {scene_id} not found"
#             print(f"[ERROR] render_video: {error_msg}")
#             supabase.log_to_supabase(session_id, "error", error_msg)
#             raise HTTPException(status_code=404, detail=error_msg)
#         case 4:
#             error_msg = "Failed to create VideoFileClip"
#             print(f"[ERROR] render_video: {error_msg}")
#             supabase.log_to_supabase(session_id, "error", error_msg)
#             raise HTTPException(status_code=400, detail=error_msg)
#         case _:
#             # Update status after successful rendering
#             print(f"[DEBUG] render_video: Successfully rendered video for session {session_id}")
#             supabase.add_or_update_section(session_id, "processing", "video_rendered", progress=100)
#             supabase.update_status(session_id, "success")
#             return FileResponse(ret, media_type="video/mp4", filename=scene_id)
        
# @videoStream.get("/summary-duration/")
# def get_summary_duration(session_id: str = Query(None)):
#     duration = get_sum_duration_svc(session_id)
    
#     # Store the duration in Supabase
#     supabase.add_sum_duration(session_id, duration)
    
#     return JSONResponse(content={"message": duration})
        
@videoStream.delete("/end-session")
def end_session(session_id: str):
    # Log the end of the session
    supabase.log_to_supabase(session_id, "info", "Session ended, clearing data")
    
    clear_svc(session_id)
    
    # Update status in Supabase
    supabase.update_status(session_id, "cleaning_up")
    return JSONResponse(content={"message": "Your data has been cleared"})

@videoStream.delete("/clear-expired")
def clear_expired(session_id: str):
    supabase.add_or_update_section(session_id, "processing", "clearing_expired_sessions")
    
    count = delete_expired()
    
    supabase.log_to_supabase(session_id, "info", f"Cleared {count} expired sessions")
    supabase.add_or_update_section(session_id, "processing", "expired_sessions_cleared")
    return JSONResponse(content={"message": f"{count} userdata has been cleared"})

# @videoStream.post("/gradio-demo")
# async def gradio_demo(background_tasks: BackgroundTasks, videos: List[UploadFile] = File(...), scripts: List[UploadFile] = File(None), language: str = Form(...)):
#     print("[INFO]: started session")
#     session_id = create_session_svc()
#     supabase.update_status(session_id, "session_created")
#     print(f"[INFO]: created session: {session_id}")

#     print("[INFO]: started upload")
#     supabase.add_or_update_section(session_id, "processing", "uploading_videos")
#     await upload_videos_svc(session_id, videos)
#     supabase.add_or_update_section(session_id, "processing", "videos_uploaded")
#     print("[INFO]: uploaded video")

#     print("[INFO]: videos:", videos)
#     print("[INFO]: scripts:", scripts)

#     task = []
#     if scripts is None:
#         supabase.add_or_update_section(session_id, "processing", "transcribing_audio")
#         task.append(asyncio.create_task(transcribe_svc_v2(session_id, lang=language, diarize=False)))
#     else:
#         supabase.add_or_update_section(session_id, "processing", "uploading_script")
#         task.append(asyncio.create_task(upload_script_svc(session_id, script=scripts)))
    
#     supabase.add_or_update_section(session_id, "processing", "chunking_video")
#     task.append(asyncio.create_task(asyncio.to_thread(chunk_svc, session_id)))
#     print("[INFO]: started ASR & chunk")
#     supabase.add_or_update_section(session_id, "processing", "start_ASR_&_chunk")
#     await asyncio.gather(*task)
#     supabase.add_or_update_section(session_id, "processing", "end_ASR_&_chunk")
#     print("[INFO]: finished ASR & chunk")
    
#     print("[INFO]: started LLM summarize")
#     supabase.add_or_update_section(session_id, "processing", "start_summarize_svc")
#     summarize_svc(session_id)
#     supabase.done_llm(session_id)  # Mark script as summarized in Supabase
#     supabase.add_or_update_section(session_id, "processing", "end_summarize_svc")
#     print("[INFO]: done LLM summarized")

#     print("[INFO]: started VQA")
#     supabase.add_or_update_section(session_id, "processing", "generating_captions")
#     await get_vqa_svc(session_id)
#     supabase.add_or_update_section(session_id, "processing", "captions_generated")
#     print("[INFO]: gotten VQA")

#     print("[INFO]: started voice API")
#     supabase.add_or_update_section(session_id, "processing", "start_gen_voice_svc")
#     audio_path, audio_url = gen_voice_svc(session_id, 149, False)
#     supabase.add_or_update_section(session_id, "processing", "end_gen_voice_svc")
#     print("[INFO]: done voice API")

#     print("[INFO]: started VQAfusing")
#     supabase.add_or_update_section(session_id, "processing", "auto_matching_scenes")
#     auto_match_list = auto_match_svc(session_id)
#     supabase.add_or_update_section(session_id, "processing", "scenes_matched")
#     print("[INFO]: done VQAfusing")

#     selected_list = [scene["id"] for scene in auto_match_list]
#     supabase.update_summary(session_id, selected_list)  # Store selected scenes in Supabase
#     select_scenes_svc(session_id, selected_list)
#     background_tasks.add_task(clear_svc, session_id=session_id)
    
#     print("[INFO]: started render concat videos")
#     supabase.add_or_update_section(session_id, "processing", "rendering_video")
#     ret, scene_id = render_video_svc(session_id, True)
    
#     match ret:
#         case 1:
#             supabase.update_status(session_id, "fail")
#             supabase.log_to_supabase(session_id, "error", f"Summary with id {session_id} not found")
#             raise HTTPException(
#                 status_code=404, detail=f"Summary with id {session_id} not found.")
#         case 2:
#             supabase.update_status(session_id, "fail")
#             supabase.log_to_supabase(session_id, "error", f"Scene named {scene_id}.mp4 not found")
#             raise HTTPException(
#                 status_code=404, detail=f"Scene named {scene_id}.mp4 not found.")
#         case 3:
#             supabase.update_status(session_id, "fail")
#             supabase.log_to_supabase(session_id, "error", f"Path to {scene_id}.mp4 does not exist")
#             raise HTTPException(
#                 status_code=404, detail=f"Path to {scene_id}.mp4 does not exist.")
#         case 4:
#             supabase.update_status(session_id, "fail")
#             supabase.log_to_supabase(session_id, "error", "Failed to create VideoFileClip")
#             raise HTTPException(
#                 status_code=400, detail="Fail to create VideoFileClip.")
#         case _:
#             supabase.update_status(session_id, "success")
#             return FileResponse(
#                 ret, media_type="video/mp4", filename=scene_id)

# @videoStream.get("/video/{session_id}")
# async def get_video_media_result(session_id: str):
#     """
#     Retrieve a video or voice file based on session_id.
#     - `file_type`: "video" or "voice"
#     - Stream the file in the browser instead of downloading.
#     """
#     print(f"Checking for session_id: {session_id}")
#     summed = supabase.find_summary(session_id)

#     ## check uid
#     media_files = os.path.join(SERVICE_DIR, VID_SUM_DIR, f"{summed['uid']}.mp4")
#     extension = ".mp4"
#     media_type = "video/mp4"

#     print(f"[INFO] Looking in file: {media_files}") 
    
#     if not os.path.exists(media_files):
#         raise HTTPException(status_code=400, detail=f"No media files found in {media_files}")

#     return FileResponse(media_files, media_type=media_type, headers={"Content-Disposition": "attachment"})
