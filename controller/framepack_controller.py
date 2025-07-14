import os
import time
import math
import uuid
import argparse
import traceback
import asyncio

from typing import List

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse

from services import *
from utils import *
from dbcontroller.supabase import SupabaseConnector

FramePackStream = APIRouter()

# Create a Supabase connector instance
supabase = SupabaseConnector()

@FramePackStream.post("/generate-image-prompt")
async def generate_imgprompt(session_id: str = Form(...)):
    
    try:
        supabase.add_or_update_section(session_id, "processing", "converting summary text into image prompt")
        img_prompt = gen_ImgPrompt(session_id)
        supabase.add_or_update_section(session_id, "processing", "converted summary text into image prompt")
        
        return JSONResponse(content={"message": "Generate prompt for generate image successfully", "image_prompt": img_prompt})
    except ValueError as e:
        supabase.add_or_update_section(session_id, "failed", "Session id not found")
        print(f"[ERROR] /generate-image-prompt failed: {str(e)}")
        return JSONResponse(status_code=404, content={"error": str(e)})
    except Exception as e:
        supabase.add_or_update_section(session_id, "failed", "unexpected error during image prompt generation")
        print(f"[ERROR] /generate-image-prompt failed: {str(e)}")
        return JSONResponse(status_code=500, content={"error": "Unexpected error during image prompt generation"})

@FramePackStream.post("/generate_image")
async def generate_image(session_id: str = Form(...)):
    supabase.add_or_update_section(session_id, "processing", "start generating image")
    gen_img = gen_Image(session_id)

    if gen_img:
        supabase.add_or_update_section(session_id, "processing", "generating image successfully")
        return JSONResponse(status_code=200, content={
            "message": "Image generated successfully",
            "image_path": gen_img
        })
    else:
        supabase.add_or_update_section(session_id, "failed", "generating image failed")
        return JSONResponse(status_code=500, content={
            "error": "Image generation returned no result"
        })

@FramePackStream.post("/generate-framepack-prompt")
async def generate_FPprompt(session_id: str = Form(...)):
    try:
        supabase.add_or_update_section(session_id, "processing", "converting summary text into framepack prompt")
        fp_prompt = gen_FramePackPrompt(session_id)
        supabase.add_or_update_section(session_id, "processing", "converted summary text into framepack prompt")
        return JSONResponse(content={"message": "Generate prompt for generate framepack successfully", "framepack_prompt": fp_prompt})
    except ValueError as e:
        supabase.add_or_update_section(session_id, "failed", "Session id not found")
        print(f"[ERROR] /generate-image-prompt failed: {str(e)}")
        return JSONResponse(status_code=404, content={"error": str(e)})
    except Exception as e:
        supabase.add_or_update_section(session_id, "failed", "unexpected error during framepack prompt generation")
        print(f"[ERROR] /generate-image-prompt failed: {str(e)}")
        return JSONResponse(status_code=500, content={"error": "Unexpected error during framepack prompt generation"})  

@FramePackStream.post("/generate_video")
async def generate_video(
    session_id: str = Form(...),
    total_second_length: int = Form(..., description="Total video length ,unit: seconds (recommended 3)"),
    latent_window_size: int = Form(..., description="The window size of the latent used to generate the frame (recommended 5)")
):
    try:
        supabase.add_or_update_section(session_id, "processing", "start framepack")
        run_fp = run_Framepack(session_id, total_second_length, latent_window_size)
        return JSONResponse(content={"message": "Generate video by framepack model successfully", "outputpath": run_fp})
    except ValueError as e:
        supabase.add_or_update_section(session_id, "failed", "Session id not found or invalid")
        print(f"[ERROR] /generate-image-prompt failed: {str(e)}")
        return JSONResponse(status_code=404, content={"error": str(e)})

    except Exception as e:
        supabase.add_or_update_section(session_id, "failed", "Unexpected error during video generation")
        print(f"[ERROR] /generate-image-prompt failed: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})