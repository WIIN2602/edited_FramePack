import os
import io
import re  # regax
import ast
import cv2
import json
import shutil
import base64
import asyncio
import hashlib
import platform
import subprocess
import traceback

from moviepy import *
import tempfile
from pydub import AudioSegment
from io import BytesIO
from utils import GPT_TOKEN, BERT_KEY, SERVICE_DIR, VID_SRC_DIR, VID_SCN_DIR, VID_SUM_DIR, SCRIPT_DIR, VOICE_DIR, CACHE_DIR, CAPTION_MODEL, FUSING_MODEL
from uuid import uuid4
from typing import List
from PIL import Image
from math import ceil
from openai import OpenAI, AsyncOpenAI
from .audio_service import gen_voice_svc
from schemas.model import SceneUploadEntry
from scenedetect import detect, ContentDetector, FrameTimecode, HashDetector
from sentence_transformers import SentenceTransformer
from fastapi import UploadFile
from tqdm import tqdm

# from services.audio_service import gen_voice_svc
# from dbcontroller.supabase_controller import *

# from dbcontroller.mongo import DBConnector
# mongo = DBConnector()

from dbcontroller.supabase import SupabaseConnector
supabase = SupabaseConnector()

client = OpenAI(api_key=GPT_TOKEN)              # For Fusing
client_async = AsyncOpenAI(api_key=GPT_TOKEN)   # For Caption
bert = SentenceTransformer(BERT_KEY, trust_remote_code=True)

with io.open('prompts/Caption_prompt.txt', mode="r", encoding="utf-8") as f:  # Thai prompt
    CaptionPrompt = f.read()

# with io.open('prompts/CaptionENG_prompt.txt', mode="r", encoding="utf-8") as f:  # English prompt
#     CaptionPrompt = f.read()

with io.open('prompts/Fusing_prompt.txt', mode="r", encoding="utf-8") as f:
    FusingPrompt = f.read()

# ------------------CREATE SESSION ID------------------#


def create_session_svc() -> str:
    """
    Creates session id and store on DB.

    Returns:
    str: Random uuid4 string.  
    """
    session_id = str(uuid4())
    # mongo.add_user(session_id)
    supabase.add_user(session_id)
    supabase.add_or_update_section(session_id, "processing", "end_create_session_svc")
    return session_id

# ------------------VIDEO PRE-PROCESS------------------#


def md5_storage(file_path: str, chunk_size: int = 8192) -> str:
    """
    Calculate hash of a file in disk memory.

    Parameters:
    file_path (str): The system path to the file.
    chunk_size (int): Hash chunk size.

    Returns:
    str: Hash string.
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


async def md5_upload(upload_file: UploadFile, chunk_size: int = 8192) -> str:
    """
    Calculate hash of an UploadFile.

    Parameters:
    upload_file (UploadFile): UploadFile from an API endpoint.
    chunk_size (int): Hash chunk size.

    Returns:
    str: Hash string.
    """
    hash_md5 = hashlib.md5()
    while True:
        chunk = await upload_file.read(chunk_size)
        if not chunk:
            break
        hash_md5.update(chunk)
    await upload_file.seek(0)  # Reset the file pointer after reading
    return hash_md5.hexdigest()

# Function to compare MD5 of physical file and UploadFile


async def compare_md5(file_path: str, upload_file: UploadFile) -> bool:
    """
    Compare hash of static file and UploadFile.

    Parameters:
    file_path (str): The system path to the file.
    upload_file (UploadFile): UploadFile from an API endpoint.

    Returns:
    bool: True if the hashes are identical, False if not.
    """
    disk_md5 = md5_storage(file_path)
    upload_md5 = await md5_upload(upload_file)
    return disk_md5 == upload_md5


def get_fps(video_path: str) -> float:
    """
    Get framerate of the video.

    Parameters:
    video_path (str): The system path to the video.

    Returns:
    float: The framerate of the video.
    """
    data = cv2.VideoCapture(video_path)
    if not data.isOpened():
        data.release()
        raise TypeError("Unable to access video path: ", video_path)
    fps = data.get(cv2.CAP_PROP_FPS)
    data.release()
    return fps


def get_duration(video_path: str) -> float:
    """
    Get duration in seconds of video.

    Parameters:
    video_path (str): The system path to the video.

    Returns:
    float: Rounded video duration in seconds.
    """
    data = cv2.VideoCapture(video_path)
    if not data.isOpened():
        data.release()
        raise TypeError("Unable to access video path: ", video_path)
    frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = data.get(cv2.CAP_PROP_FPS)
    seconds = float(round(frames / fps))
    data.release()
    return seconds


def fps_cap(video_uuid: str, session_id: str):
    """
    Re-encode an MP4 for a video above 30FPS (SceneDetect error if >30FPS).

    Parameters:
    video_uuid (str): The video filename, no extension.
    session_id (str): The user's session id.
    """
    vid_path = os.path.join(SERVICE_DIR, VID_SRC_DIR,
                            session_id, f'{video_uuid}.mp4')
    cache_path = os.path.join(SERVICE_DIR, CACHE_DIR,
                              session_id, f'{video_uuid}.mp4')
    if not os.path.exists(os.path.dirname(cache_path)):
        os.mkdir(os.path.dirname(cache_path))

    ffmpeg_command = f'ffmpeg -y -i "{vid_path}" -filter:v fps=30 -preset ultrafast -crf 28 "{cache_path}"'
    result = subprocess.call(
        ffmpeg_command, shell=platform.system() != 'Windows')
    if result == 0:
        # move 30FPS video from cache/ to sources/ (replace)
        if platform.system() == 'Windows':
            subprocess.call(f'move "{cache_path}" "{vid_path}"', shell=True)
        else:
            subprocess.call(f'mv "{cache_path}" "{vid_path}"', shell=True)
        return

    else:
        raise TypeError("FFmpeg command failed.")

# -----------------USER UPLOADS VIDEOS-----------------#


async def upload_videos_svc(session_id: str, videos: List[UploadFile]):
    """
    Check if user has uploaded any of these video before.
    Store each video a user uploaded onto storage and add metadata to DB.
    Check the framerate and re-encode to 30FPS if it exceeded.

    Parameters:
    session_id (str): The user's session id.
    videos (List[UploadFile]): List of UploadFile from API endpoint.
    """
    supabase.add_or_update_section(session_id, "processing", "start_upload")

    # mongo.update_used(session_id)
    supabase.update_used(session_id)

    user_path = os.path.join(SERVICE_DIR, VID_SRC_DIR, session_id)
    if not os.path.exists(user_path):
        os.makedirs(user_path, exist_ok=True)

    vid_infos = []
    for video in videos:
        name, ext = os.path.splitext(video.filename)
        # Generate a random UID first - we'll use this consistently
        random_uid = str(uuid4())

        dupe = supabase.match_fname(session_id, "".join([name, ext]))
        print
        if dupe:
            i = 1
            dpath = os.path.join(user_path, f"{dupe[0]['filename']}")
            same = await compare_md5(file_path=dpath, upload_file=video)
            print("[DEBUG] duplicate file found: ", dpath, same)
            if same:
                continue
            else:
                while len(supabase.match_fname(session_id, name + f"_({i}")) != 0:
                    i += 1
                name = name + f"_({i})"

        # Now save the file with the UUID instead of the original filename
        save_path = os.path.join(user_path, f'{name}{ext}')
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        fps = get_fps(save_path)
        if fps > 30:
            # Use random_uid for fps_cap function
            fps_cap(random_uid, session_id)

        # Store both the UID and original filename for reference
        vid_infos.append({
            "uid": random_uid,
            "chunked": False,
            "session_id": session_id,
            "filename": "".join([name, ext])
        })

    if vid_infos:
        # mongo.add_source(vid_infos)
        supabase.add_or_update_section(session_id, "processing", "end_upload_videos_svc")
        supabase.add_source(vid_infos)
    return

# --------------------SCENES DETECT--------------------#


def pyscene_detect(video_path: str, video_uuid: str, session_id: str) -> List[SceneUploadEntry]:
    """
    Detect jump-cuts using HashDetector and
    store the start/end timestamps of each segments.
    Screenshot and save one image per chunk for image captioning.

    Parameters:
    video_path (str): The system path to the video.
    video_uuid (str): The video filename, no extension.
    session_id (str): The user's session id.

    Returns:
    List[SceneUploadEntry]: List of SceneUploadEntry Pydantic model (schemas/model.py)
    """
    print(
        f"[DEBUG] pyscene_detect: Started processing video {video_uuid} for session {session_id}")
    fps = get_fps(video_path)
    duration = get_duration(video_path)
    img_path = os.path.join(SERVICE_DIR, VID_SCN_DIR, session_id)
    if not os.path.exists(img_path):
        os.makedirs(img_path, exist_ok=True)
        print(f"[DEBUG] Created directory for scene images: {img_path}")

    # scene_list = detect(video_path, ContentDetector(threshold=0.45, min_scene_len=fps*2), show_progress=False)
    scene_list = detect(video_path,
                        HashDetector(threshold=0.45, min_scene_len=fps*2),
                        show_progress=False)
    print(
        f"[DEBUG] Scene detection found {len(scene_list) if scene_list else 0} scenes")

    # Scene creation logic remains the same...
    if not scene_list and duration <= 30:
        scene_list.append([FrameTimecode(0.0, fps),
                           FrameTimecode(duration, fps)])
        print(
            f"[DEBUG] No scenes detected and duration <= 30s, using entire video as one scene")

    elif not scene_list:
        segments = ceil(duration / 30)
        print(
            f"[DEBUG] No scenes detected, splitting into {segments} equal segments")
        for i in range(segments):
            start_time = i * 30
            end_time = min((i + 1) * 30, duration)
            scene_list.append([FrameTimecode(float(start_time), fps),
                               FrameTimecode(float(end_time), fps)])

    for i, frames in enumerate(scene_list):
        if int(frames[1]) - int(frames[0]) < int(fps*2):
            scene_list.pop(i)
            print(
                f"[DEBUG] Removed scene {i} because duration was less than 2 seconds")

    scene_entries = [SceneUploadEntry(uid=str(uuid4()),
                                      source_uid=video_uuid,
                                      timestamp=[float(scene[0]),
                                                 float(scene[1])],
                                      duration=float(scene[1])
                                      - float(scene[0]),
                                      session_id=session_id)
                     for scene in scene_list]
    print(f"[DEBUG] Created {len(scene_entries)} SceneUploadEntry objects")

    # Generate thumbnails for the scenes
    data = cv2.VideoCapture(video_path)
    width = data.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = data.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if width > height:
        reso = (960, 540)
    elif height > width:
        reso = (540, 960)
    else:
        reso = (540, 540)
    print(f"[DEBUG] Capturing thumbnail images at resolution {reso}")

    for entry in scene_entries:
        # 2 seconds after the previous jump-cut.
        frame_number = int(entry.timestamp[0] * fps) + int(fps * 2)
        data.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = data.read()
        if success:
            resized = cv2.resize(frame, reso)
            cv2.imwrite(os.path.join(img_path, f"{entry.uid}.jpg"), resized)
            print(f"[DEBUG] Saved thumbnail for scene {entry.uid}")
        else:
            print(
                f"[WARNING] Failed to capture thumbnail for scene {entry.uid}")

    if data.isOpened():
        data.release()
    print(f"[DEBUG] Marking source video {video_uuid} as chunked")
    # mongo.done_chunk(video_uuid, session_id)
    supabase.done_chunk(video_uuid, session_id)

    # Convert the scene entries to dictionaries but modify for Supabase compatibility
    scene_dicts = []
    for entry in scene_entries:
        # Create a dictionary from the model
        entry_dict = entry.model_dump()

        # Convert timestamp to JSONB format instead of removing it
        if 'timestamp' in entry_dict and entry_dict['timestamp'] is not None:
            # Store as a properly formatted JSON object
            entry_dict['timestamp'] = {
                'start': float(entry_dict['timestamp'][0]),
                'end': float(entry_dict['timestamp'][1])
            }

        # Ensure embedding has at least one dimension
        if not entry_dict.get('embedding') or len(entry_dict.get('embedding', [])) == 0:
            entry_dict['embedding'] = [0.0] * 384

        scene_dicts.append(entry_dict)
        print(
            f"[DEBUG] Prepared scene: uid={entry_dict['uid']}, duration={entry_dict['duration']}")

    try:
        print(
            f"[DEBUG] Attempting to add {len(scene_dicts)} scenes to Supabase")
        supabase.add_scene(scene_dicts)
        print(f"[DEBUG] Successfully added scenes to Supabase")
    except Exception as e:
        print(f"[ERROR] Failed to add scenes to Supabase: {str(e)}")
        # Print detailed error information
        traceback.print_exc()
        # Try adding one scene at a time to identify problematic entries
        for i, entry in enumerate(scene_dicts):
            try:
                print(
                    f"[DEBUG] Trying to add scene {i} individually: {entry['uid']}")
                supabase.add_scene([entry])
                print(
                    f"[DEBUG] Successfully added individual scene {entry['uid']}")
            except Exception as e:
                print(
                    f"[ERROR] Failed to add scene {i} ({entry['uid']}): {str(e)}")

    return scene_entries


def chunk_svc(session_id: str):
    """
    Get start/end segments of each user-uploaded video.
    Screenshot and save one image per chunk for image captioning.
    Store chunk's metadata to DB.

    Parameters:
    session_id (str): The user's session id.
    """
    supabase.add_or_update_section(session_id, "processing", "start-ASR & chunk")
    print(f"[DEBUG] chunk_svc: Started processing for session {session_id}")

    # mongo.update_used(session_id)
    supabase.update_used(session_id)
    videos = supabase.find_source(session_id)
    print(f"[DEBUG] Found {len(videos)} videos for session {session_id}")
    
    vuids = [video for video in videos if video["chunked"] is False]
    print(f"[DEBUG] Found {len(vuids)} videos that need chunking")

    ## check filename
    for vuid in vuids:
        src_path = os.path.join(SERVICE_DIR,
                                VID_SRC_DIR,
                                session_id,
                                f'{vuid["filename"]}')
        print(f"[DEBUG] Processing video: {src_path}")

        try:
            scene_entries = pyscene_detect(
                video_path=src_path,
                video_uuid=vuid["uid"],
                session_id=session_id
            )
            # mongo.add_scene(scene_entries)
            print(
                f"[DEBUG] pyscene_detect returned {len(scene_entries)} scenes")

            # This line is redundant as pyscene_detect already calls add_scene
            # Instead of removing it, let's make sure it has valid data
            scene_dicts = [entry.model_dump() for entry in scene_entries]
            # Ensure embeddings exist for all entries
            for entry in scene_dicts:
                if not entry.get('embedding') or len(entry.get('embedding', [])) == 0:
                    entry['embedding'] = [0.0] * 384

            # Double-check if scenes were actually added to Supabase
            db_scenes = supabase.find_scene_owner(session_id)
            scene_uids = [scene["uid"] for scene in scene_dicts]
            db_scene_uids = [scene["uid"] for scene in db_scenes]

            missing_scenes = [
                uid for uid in scene_uids if uid not in db_scene_uids]
            if missing_scenes:
                print(
                    f"[WARNING] {len(missing_scenes)} scenes were not added to Supabase")
                print(f"[DEBUG] Attempting to add missing scenes manually")
                # Try to add only the missing scenes
                missing_scene_dicts = [
                    s for s in scene_dicts if s["uid"] in missing_scenes]
                try:
                    #
                    supabase.add_scene(missing_scene_dicts)
                    print(
                        f"[DEBUG] Successfully added {len(missing_scene_dicts)} missing scenes")
                except Exception as e:
                    print(f"[ERROR] Failed to add missing scenes: {str(e)}")
            else:
                print(f"[DEBUG] All scenes were successfully added to Supabase")

        except Exception as e:
            print(f"[ERROR] Failed to process video {vuid['uid']}: {str(e)}")
            traceback.print_exc()

    supabase.add_or_update_section(session_id, "processing", "end-ASR & chunk")
    print(f"[DEBUG] chunk_svc: Completed processing for session {session_id}")
    return

# -------------------IMAGE CAPTIONING------------------#


def encode_image(image: Image) -> str:
    """
    Encode image as base64 string.

    Parameters:
    image (Image): PIL image of each chunk.

    Returns:
    str: base64 string for sending to external API.
    """
    im_file = BytesIO()
    image.save(im_file, format='PNG')
    im_bytes = im_file.getvalue()
    return base64.b64encode(im_bytes).decode('utf-8')

# -------------------Encode_Image V2 ------------------#


def encode_image_v2(image: Image) -> str:
    """
    Encode image as a base64 string after resizing to 512x512.

    Parameters:
    image (Image): PIL image object to be encoded.

    Returns:
    str: Base64-encoded string of the resized image in PNG format,
         suitable for sending to external APIs or LLMs.
    """
    w, h = image.size
    if w > 512 or h > 512:
        print(f"[RESIZE] Resizing from {w}x{h} to 512x512")
        image = image.resize((512, 512), Image.LANCZOS)

    im_file = BytesIO()
    image.save(im_file, format='PNG')
    im_bytes = im_file.getvalue()
    return base64.b64encode(im_bytes).decode('utf-8')


async def generate_response(question: str, image: Image):
    """
    Convert PIL image to base64 image.
    Send prompt and base64 image to GPT model.

    Parameters:
    question (str): The image captioning prompt.
    image (Image): PIL Image to be converted and send.

    Returns:
    str: the response content from OpenAI's API
    """
    # base64_image = encode_image(image)
    base64_image = image
    try:
        response = await client_async.chat.completions.create(
            model=CAPTION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=100,
            temperature=0,
            stream=False
        )

        # use hasattr to check an object spacific attribute
        if hasattr(response, "usage") and response.usage:
            print("🧾 Prompt Tokens:", response.usage.prompt_tokens)
            print("🗣️ Completion Tokens:", response.usage.completion_tokens)
            print("📊 Total Tokens:", response.usage.total_tokens)

        return response.choices[0].message.content
    except Exception as e:
        print(f"[ERROR]: Captioning - {e}")
        return ""


def get_vqa(image: Image):
    """
    Do image captioning with OpenAI's API.
    Split the response by "," as delimiter.

    Parameters:
    image (Image): The PIL image to be send.

    Returns:
    list[str]: Response content separated into strings.
    """
    response = generate_response(CaptionPrompt, image=image)
    return response.split(",")


async def get_vqa_async(list_base64_image: list):
    """
    Faster image captioning by AsyncOpenAI.
    Split the response by "," as delimiter.

    Parameters:
    list_base64_image (list): list of base64 images to be send.

    Returns:
    list[str]: list of captions for each images.
    """
    task = []
    for base64_image in list_base64_image:
        task.append(asyncio.create_task(
            generate_response(CaptionPrompt, image=base64_image)))
    output = await asyncio.gather(*task)
    list_caption = []
    print("output vqa:")
    for response in tqdm(output):
        print(response)
        list_caption.append(response.split(","))
    return list_caption

# -------------------TEXT EMBEDDINGS-------------------#


def remove_words(caption: list[str], words: list[str]) -> list[str]:
    """
    Remove unwanted words from the image captions.

    Parameters:
    caption (list[str]): from get_vqa()

    Returns:
    list[str]: captions with words removed.
    """
    temp = caption
    for word in words:
        temp = [w for w in temp if word not in w]
    return temp


def get_embedding(text: str):
    """
    Use BERT to transform string into vector embeddings.

    Parameters:
    text (str): string to embed.
    """
    embedding = bert.encode(text)
    if len(embedding) == 768:
        return embedding[:384].tolist()

    # If embedding is already 384 dimensions
    elif len(embedding) == 384:
        return embedding.tolist()
    # If embedding is a different size, either truncate or pad to 384
    else:
        if len(embedding) > 384:
            return embedding[:384].tolist()
        else:  # len(embedding) < 384
            # Pad with zeros to reach 384 dimensions
            padded_embedding = list(embedding) + [0.0] * (384 - len(embedding))
            return padded_embedding


async def get_vqa_svc(session_id: str):
    """
    Send all snapshots to image captioning model.
    Remove unwanted words and transform into embeddings.
    Upload embeddings (array of float) to DB.

    Parameters:
    session_id (str): The user's session id.
    """
    supabase.add_or_update_section(session_id, "processing", "start-VQA")
    # mongo.update_used(session_id)
    supabase.update_used(session_id)
    docs = supabase.find_scene_owner(session_id)
    scenes = [doc for doc in docs if not doc["caption"]]

    list_base64_image = []
    for scene in scenes:
        name = scene["uid"]
        path = os.path.join(SERVICE_DIR, VID_SCN_DIR, session_id, name)
        image = Image.open(path + '.jpg')
        # base64_image = encode_image(image)  #  เดิม ไม่มี resize → ใช้ token สูง
        base64_image = encode_image_v2(image)  # V2: resized 512x512 → ลด token
        list_base64_image.append(base64_image)
        image.close()

    print(f"[INFO]: len image is {len(list_base64_image)}")
    # list_caption = get_vqa(list_base64_image)
    list_caption = await get_vqa_async(list_base64_image)

    for idx, (scene, caption) in enumerate(zip(scenes, list_caption)):
        name = scene["uid"]
        caption = remove_words(
            caption, ['ไทยรัฐนิวส์โชว์', 'ไทยรัฐ นิวส์โชว์', 'Thairath News Show'])
        caption = ', '.join(item.strip('" ') for item in caption)

        embedding = get_embedding(caption)

        # mongo.add_vqa_embed(name, caption, embedding)
        supabase.add_vqa_embed(name, caption, embedding)
    supabase.add_or_update_section(session_id, "processing", "end-VQA")


def gen_search_pipeline(session_id: str, embedded_caption: list, user_vid: list, limit: int = 5):
    """
    Creates and returns a search pipeline structure
    to be used with Supabase's vector search via SupabaseConnector.search_vector.

    Parameters:
    session_id (str): Session id of user performing the search.
    embedded_caption (list): Embeddings of their search string (vector of 384 dimensions).
    user_vid (list): List of original video UIDs uploaded by this user.
    limit (int): The amount of returned results with the highest similarity values.

    Returns:
    list[dict]: Search pipeline format compatible with SupabaseConnector.search_vector.
    """
    return [
        {
            "$vectorSearch": {
                "index": "ai_backend_scene_videos",
                "queryVector": embedded_caption,
                "path": "embedding",
                "numCandidates": 100,
                "limit": 100,
            }
        },
        {
            "$match": {
                "source_uid": {
                    "$in": user_vid
                },
                "session_id": {
                    "$eq": session_id
                }
            }
        },
        {
            "$limit": limit
        },
        {
            "$project": {
                "_id": 0,
                "uid": 1,
                "duration": 1,
                "caption": 1
            }
        },
    ]

# -------------------PAGE2 DISPLAY---------------------------#


def string_list_to_list(string: str):
    """
    Remove any code block markers (``` or ```).
    Evaluate the cleaned string.

    Parameters:
    string (str): the response content from OpenAI's API.

    Returns:
    str: Literal evaluated string.
    None: If value error or syntax error.
    """
    clean_string = string.strip("```").strip()
    try:
        return ast.literal_eval(clean_string)
    except (ValueError, SyntaxError) as e:
        print(f"Error evaluating the string: {e}")
        return None


def VQAfusing(client, VqaChunk, Script_Pure):
    """
    Send candidates chunk with uid and caption to GPT.
    Send the whole summarized script to GPT.
    Prompt it to return correlated match.

    Parameters:
    client (OpenAI): GPT LLM client.
    VqaChunk (list[dict]): Contains uid, caption, duration of candidate scenes.
    Script_Pure (list[str]): Contains lines of summarized script and TTS .wav duration.

    Returns:
    list: list of scene uids in AI-recommended order to be concatenated.
    """
    print(f"Script_Pure: {Script_Pure}")
    try:
        completion = client.chat.completions.create(
            model=FUSING_MODEL,
            messages=[
                {"role": "system", "content": FusingPrompt},
                {"role": "user", "content": f"timestamp with Tag : {VqaChunk}, news script : {Script_Pure}"}
            ],
            temperature=0
        )

        print(f"Completion content: {completion.choices[0].message.content}")
        result = string_list_to_list(completion.choices[0].message.content)

        if not isinstance(result[0], list):
            result = [result]

        print(f"[DEBUG] VQAfusing result: {result}")
        return result
    except Exception as e:
        print(f"[ERROR]: Fusing - {e}")
        return [[""]]


def auto_match_svc(session_id: str):
    """
    Remove old summary video if it exists.
    Read summarized script line-by-line and store as list[str].
    Transform each line into embeddings and perform DB search.
    Send candidates from search with summarized script to GPT.
    Calculate offset and store on DB, to be used in render_video_svc.

    Parameters:
    session_id (str): The user's session id.

    Returns:
    list: Contains captions, thumbnails, uids, durations of scenes and offset.
    """
    print(f"[DEBUG] auto_match_svc: Starting for session {session_id}")
    supabase.add_or_update_section(session_id, "processing", "start-VQAfusing")

    # mongo.update_used(session_id)
    supabase.update_used(session_id)

    # Remove old summary
    # exist = mongo.find_summary(session_id)
    exist = supabase.find_summary(session_id)
    if exist:
        print(f"[DEBUG] auto_match_svc: Found existing summary, removing it")
        if exist.get('duration'):
            path = os.path.join(SERVICE_DIR, VID_SUM_DIR,
                                exist['uid'] + ".mp4")
            safe_remove(path)
        supabase.del_summary(session_id)

    # Paths for script summary and thumbnail directory
    script_sum_path = os.path.join(
        SERVICE_DIR, SCRIPT_DIR, session_id, f'{session_id}_task1.txt')
    audio_path = os.path.join(SERVICE_DIR, VOICE_DIR, f"{session_id}.wav")
    thumb_dir = os.path.join(SERVICE_DIR, VID_SCN_DIR, session_id)
    print(
        f"[DEBUG] auto_match_svc: Checking paths - Script: {script_sum_path}, Audio: {audio_path}")

    # Ensure script summary exists
    if not os.path.exists(script_sum_path):
        print(
            f"[ERROR] auto_match_svc: Script summary file not found at {script_sum_path}")
        raise FileNotFoundError(
            f"Script summary file for session {session_id} not found.")

    # Read script summary
    with io.open(script_sum_path, mode="r", encoding="utf-8") as f:
        script_sum = f.read().strip().split("\n")

    if not script_sum:
        supabase.add_or_update_section(session_id, "fail", "error-summay-script-not-found-VQAfusing")
        print(f"[ERROR] auto_match_svc: Script summary is empty")
        raise ValueError("Script summary is empty.")

    if not os.path.exists(audio_path):
        supabase.add_or_update_section(session_id, "fail", "error-audio-not-found-VQAfusing")
        print(f"[ERROR] auto_match_svc: Audio file not found at {audio_path}")
        raise FileNotFoundError(
            f"Audio file for session {session_id} not found.")

    print(f"[DEBUG] auto_match_svc: Loading audio file")
    audio = AudioSegment.from_wav(audio_path)

    audio_duration = len(audio) / 1000.0
    print(
        f"[DEBUG] auto_match_svc: Audio duration: {audio_duration:.2f} seconds")

    if script_sum:
        script_sum[-1] = f"{script_sum[-1]} audio duration: {audio_duration:.3f} seconds"

    print(
        f"[DEBUG] auto_match_svc: Generating embeddings for {len(script_sum)} script lines")
    script_sum_embeddings = [get_embedding(
        sentence.strip()) for sentence in tqdm(script_sum)]

    # videos = mongo.find_source(session_id)
    videos = supabase.find_source(session_id)
    user_vid = [video["uid"] for video in videos]

    if not user_vid:
        supabase.add_or_update_section(session_id, "fail", "error-user-not-found-VQAfusing")
        print(f"[ERROR] auto_match_svc: No videos found for session {session_id}")
        raise TypeError("User video not found.")

    print(f"[DEBUG] auto_match_svc: Found {len(user_vid)} user videos")

    all_scenes = []
    for embedded_caption in script_sum_embeddings:
        pipeline = gen_search_pipeline(
            session_id, embedded_caption, user_vid, limit=10)
        # results = mongo.search_vector(pipeline)
        results = supabase.search_vector(pipeline)
        print(
            f"[DEBUG] auto_match_svc: Search returned {len(results)} results")
        for result in results:
            scene = {
                "uid": result["uid"],
                "caption": result["caption"],
                "duration": result["duration"]
            }
            all_scenes.append(scene)

    print(f"[DEBUG] auto_match_svc: Total scenes collected: {len(all_scenes)}")
    print(f"[DEBUG] auto_match_svc: Sending to VQAfusing")

    print("Script sum: ", script_sum)
    print("VQA chunk:", all_scenes)
    selected_scene_uids = VQAfusing(client, all_scenes, script_sum)

    # Initialize list to collect all the matched scenes with the same format
    matched_scenes = []
    total_duration = 0
    selected_scenes = []
    x = 0

    print(
        f"[DEBUG] auto_match_svc: Processing {len(selected_scene_uids)} scene UID lists")
    for scene_uid_list in selected_scene_uids:
        for scene_uid in scene_uid_list:
            # Find the scene details by UID
            scene = next(
                (s for s in all_scenes if s.get("uid") == scene_uid), None)
            if scene:
                total_duration += scene["duration"]
                selected_scenes.append(scene)
                print(
                    f"[DEBUG] auto_match_svc: Added scene {scene_uid}, total duration: {total_duration:.2f}s")

                # Stop if the total duration is equal or slightly exceeds the audio duration
                if total_duration >= audio_duration:  # Initially, Allow a 4-second buffer by "audio_duration - 4"
                    print(
                        f"[DEBUG] auto_match_svc: Total duration ({total_duration:.2f}s) exceeds audio duration ({audio_duration:.2f}s)")
                    break
        if total_duration >= audio_duration:  # Initially, Allow a 4-second buffer by "audio_duration - 4"
            break

    # Trim or adjust the final scene to match the audio duration closely
    if total_duration > audio_duration:
        extra_time = total_duration - audio_duration
        last_scene = selected_scenes[-1]
        print(
            f"[DEBUG] auto_match_svc: Need to trim {extra_time:.2f}s from last scene")
        if last_scene["duration"] > extra_time:
            x += extra_time
            print(f"[DEBUG] auto_match_svc: Set offset to {x:.2f}s")
        else:
            selected_scenes.pop()
            # x += extra_time - last_scene["duration"]  # remove last chunk duration from the offset
            print(
                f"[DEBUG] auto_match_svc: Removed last scene as it was too short to trim")

    auto_scenes = []
    print(
        f"[DEBUG] auto_match_svc: Building final scene list with {len(selected_scenes)} scenes")
    # Build matched_scenes response
    for scene in selected_scenes:
        minute = int(scene["duration"] / 60)
        second = int(scene["duration"] % 60)
        video_duration = f"{minute:02}:{second:02}"
        image = Image.open(os.path.join(thumb_dir, scene["uid"] + '.jpg'))
        # base64_img = encode_image(image)  #  เดิม ไม่มี resize → ใช้ token สูง
        base64_img = encode_image_v2(image)  # V2: resized 512x512 → ลด token
        image.close()
        auto_scenes.append(scene["uid"])
        matched_scenes.append({
            "caption": scene["caption"],
            "image_base64": base64_img,
            "id": scene["uid"],
            "video_duration": video_duration,
            "offset": x
        })

    sum_name = str(uuid4())
    print(
        f"[DEBUG] auto_match_svc: Saving summary with ID {sum_name}, offset {x:.2f}s")

    # mongo.add_offset(sum_name, auto_scenes, session_id, x)
    supabase.add_offset(sum_name, auto_scenes, session_id, x)
    print(
        f"[DEBUG] auto_match_svc: Completed with {len(matched_scenes)} matched scenes")

    supabase.add_or_update_section(session_id, "processing", "end-VQAfusing")
    return matched_scenes

# -------------------STRING SEARCH---------------------------#


def search_svc(session_id: str, search: str = None):
    """
    When there are no search string, returns all scenes of this user.
    With a search string, transform it into embeddings.
    Form a DB search pipeline and form return payload of search results.

    Parameters:
    session_id (str): The user's session id.
    search (str): The string to search for in captions.

    Returns:
    list[dict]: Contains caption, thumbnail, uid, and duration of scenes.
    """
    # mongo.update_used(session_id)
    # videos = mongo.find_source(session_id)

    supabase.update_used(session_id)
    videos = supabase.find_source(session_id)
    user_vid = [video["uid"] for video in videos]
    thumb_dir = os.path.join(SERVICE_DIR, VID_SCN_DIR, session_id)

    if not user_vid:
        raise TypeError("User video not found.")

    scene_list = []
    if not search:
        # scenes = mongo.find_scene_owner(session_id)
        scenes = supabase.find_scene_owner(session_id)

        for scene in scenes:
            minute = int(scene["duration"] / 60)
            second = int(scene["duration"] % 60)
            video_duration = f"{minute:02}:{second:02}"
            image = Image.open(os.path.join(thumb_dir, scene["uid"] + '.jpg'))
            # base64_img = encode_image(image)  #  เดิม ไม่มี resize → ใช้ token สูง
            # V2: resized 512x512 → ลด token
            base64_img = encode_image_v2(image)
            image.close()
            scene_list.append(
                {
                    "caption": scene["caption"],
                    "image_base64": base64_img,
                    "id": scene["uid"],
                    "video_duration": video_duration
                }
            )

    else:
        query = get_embedding(search)
        pipeline = gen_search_pipeline(session_id, query, user_vid)
        # results = mongo.search_vector(pipeline)
        results = supabase.search_vector(pipeline)

        for result in results:
            minute = int(result["duration"] / 60)
            second = int(result["duration"] % 60)
            video_duration = f"{minute:02}:{second:02}"
            image = Image.open(os.path.join(thumb_dir, result["uid"] + '.jpg'))
            # base64_img = encode_image(image)  #  เดิม ไม่มี resize → ใช้ token สูง
            # V2: resized 512x512 → ลด token
            base64_img = encode_image_v2(image)
            image.close()
            scene_list.append(
                {
                    "caption": result["caption"],
                    "image_base64": base64_img,
                    "id": result["uid"],
                    "video_duration": video_duration
                }
            )
    print('search results:', len(scene_list))
    return scene_list


def select_scenes_svc(session_id: str, scenes):
    """
    Add a DB document if no auto_match_svc was done (no audio use-case).
    Add the list of scenes to be concatenated to the DB document.

    Parameters:
    session_id (str): The user's session id.
    scenes (list): list of scene uids to be concatenated.
    """
    supabase.add_or_update_section(session_id, "processing", "start-select_scenes_svc")
    print(f"[DEBUG] select_scenes_svc: Started for session {session_id}")
    print(
        f"[DEBUG] select_scenes_svc: Scenes parameter: {type(scenes)}, {scenes[:5] if isinstance(scenes, list) and len(scenes) > 0 else scenes}")

    # Update used timestamp
    # mongo.update_used(session_id)
    supabase.update_used(session_id)

    # Find existing summary
    # exist = mongo.find_summary(session_id)
    exist = supabase.find_summary(session_id)
    print(
        f"[DEBUG] select_scenes_svc: Existing summary found: {exist is not None}")

    # if exist and exist.get('uid'):
    #     path = os.path.join(SERVICE_DIR, VID_SUM_DIR, exist['uid'] + ".mp4")
    #     safe_remove(path)
    #     mongo.del_summary(session_id)

    # Create a new summary if none exists
    if not exist:
        print("[DEBUG] select_scenes_svc: No existing summary found, creating new one")
        summary_id = str(uuid4())
        print(
            f"[DEBUG] select_scenes_svc: Generated new summary ID: {summary_id}")

        # mongo.add_offset(str(uuid4()), [], session_id, 0.0)
        supabase.add_offset(summary_id, [], session_id, 0.0)
        print("[DEBUG] select_scenes_svc: Added new summary with empty scene list")

    # Validate scenes parameter
    if not isinstance(scenes, list):
        print(
            f"[WARNING] select_scenes_svc: scenes parameter is not a list, converting: {scenes}")
        if isinstance(scenes, str):
            # Try to convert from string to list
            try:
                scenes = json.loads(scenes)
                print(
                    f"[DEBUG] select_scenes_svc: Converted scenes string to list: {scenes[:5] if len(scenes) > 0 else []}")
            except:
                # Extract UUIDs with regex
                uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
                scenes = re.findall(uuid_pattern, scenes)
                print(
                    f"[DEBUG] select_scenes_svc: Extracted {len(scenes)} UUIDs from string")
        else:
            scenes = []
            print(
                f"[ERROR] select_scenes_svc: Could not convert {type(scenes)} to list")

    # Update the summary with selected scenes
    print(
        f"[DEBUG] select_scenes_svc: Updating summary with {len(scenes)} selected scenes")
    try:
        # mongo.update_summary(session_id, scenes)
        supabase.update_summary(session_id, scenes)
        supabase.add_or_update_section(session_id, "processing", "end-select_scenes_svc")
        print(f"[DEBUG] select_scenes_svc: Successfully updated summary with scenes")
    except Exception as e:
        print(f"[ERROR] select_scenes_svc: Failed to update summary: {str(e)}")
        traceback.print_exc()

    print(f"[DEBUG] select_scenes_svc: Completed for session {session_id}")
    return


# -------------------VIDEO MERGE---------------------------#
def render_video_svc(session_id: str, replace: bool = False):
    """
    Get list of selected scenes from DB.
    Create a subclip of them from their original video.
    Apply offset to the last scene in the list.
    Concatenate them together into one video.
    Replace the video's audio with TTS audio if the 'replace' bool is True.
    Write an MP4 video.

    Parameters:
    session_id (str): The user's session id.
    replace (bool): True will replace video's audio with TTS, False will not.

    Returns:
    str: System path to the rendered video.
    str: File name with .mp4 extension.
    """
    supabase.add_or_update_section(session_id, "processing", "start-render_video_svc")
    print(f"[DEBUG] render_video_svc: Starting for session {session_id}, replace={replace}")
    
    # Update used timestamp
    # mongo.update_used(session_id)
    supabase.update_used(session_id)

    # Find summary entry for this session
    # summary = mongo.find_summary(session_id)
    summary = supabase.find_summary(session_id)
    print(f"[DEBUG] render_video_svc: Found summary: {summary is not None}")

    if not summary:
        print(
            f"[ERROR] render_video_svc: No summary found for session {session_id}")
        return 1, session_id

    # video_name = session_id
    video_name = summary['uid']
    output_path = os.path.join(SERVICE_DIR, VID_SUM_DIR, video_name + '.mp4')
    print(f"[DEBUG] render_video_svc: Output path will be {output_path}")

    if os.path.exists(output_path):
        print(f"[DEBUG] render_video_svc: Output file already exists, removing it")
        safe_remove(output_path)

    # Look for selected_scenes in the summary
    if 'selected_scenes' in summary and summary['selected_scenes']:
        print(f"[DEBUG] render_video_svc: Found selected_scenes key in summary")
        selected_scenes = summary['selected_scenes']
    else:
        # For backward compatibility
        print(
            f"[DEBUG] render_video_svc: No selected_scenes found in summary, checking selected_scene")
        return 2, session_id

    print(
        f"[DEBUG] render_video_svc: Found {len(selected_scenes) if isinstance(selected_scenes, list) else 'non-list'} selected scenes")

    # Check if selected_scenes is a list
    if not isinstance(selected_scenes, list):
        print(
            f"[ERROR] render_video_svc: selected_scenes is not a list, it's a {type(selected_scenes)}")
        return 2, session_id

    # Filter out any non-UUID entries
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    valid_scenes = [s for s in selected_scenes if isinstance(
        s, str) and re.match(uuid_pattern, s)]

    if len(valid_scenes) != len(selected_scenes):
        print(
            f"[WARNING] render_video_svc: Filtered out {len(selected_scenes) - len(valid_scenes)} invalid scene IDs")
        if not valid_scenes:
            print(
                f"[ERROR] render_video_svc: No valid scene IDs remain after filtering")
            return 2, session_id

    # selected_scenes = summary['selected_scene']
    # offset = summary.get('offset', 0)

    # Get offset from summary
    offset = summary['offset']
    print(f"[DEBUG] render_video_svc: Using offset of {offset}")

    if 'auto_scenes' in summary and selected_scenes != summary.get('auto_scenes', []):
        print(
            f"[DEBUG] render_video_svc: Selected scenes differ from auto scenes, setting offset to 0")
        offset = 0

    # Find scene information for each selected scene
    sources = []
    stamps = []
    valid_uids = []

    # sources = []
    # stamps = []
    # for uuid in selected_scenes:
    #     temp = mongo.find_single_scene(uuid)
    #     sources.append(temp['source_uid'])
    #     stamps.append(temp['timestamp'])

    # Handle timestamp type issues because of JSONB
    for idx, uuid in enumerate(valid_scenes):
        temp = supabase.find_single_scene(uuid)

        if temp is None:
            print(
                f"[WARNING] render_video_svc: Scene with UUID {uuid} not found, skipping")
            continue

        # Debug the exact format and value of the timestamp field
        print(
            f"[DEBUG] render_video_svc: timestamp type: {type(temp['timestamp'])}, value: {temp['timestamp']}")

        if isinstance(temp['timestamp'], str):
            # If it's stored as a JSON string, parse it
            try:
                timestamp = json.loads(temp['timestamp'])
                print(
                    f"[DEBUG] render_video_svc: Parsed timestamp from JSON string: {timestamp}")
            except:
                # If it's not valid JSON, try to extract numbers using regex
                numbers = re.findall(r'\d+(?:\.\d+)?', temp['timestamp'])
                if len(numbers) >= 2:
                    timestamp = [float(numbers[0]), float(numbers[1])]
                    print(
                        f"[DEBUG] render_video_svc: Extracted timestamp from string using regex: {timestamp}")
                else:
                    print(
                        f"[ERROR] render_video_svc: Cannot parse timestamp string: {temp['timestamp']}")
                    continue
        elif isinstance(temp['timestamp'], list):
            # If it's already a list, use it directly
            timestamp = temp['timestamp']
            print(
                f"[DEBUG] render_video_svc: Timestamp is already a list: {timestamp}")
        elif isinstance(temp['timestamp'], dict):
            # If it's a dictionary (from JSONB), extract values
            if 'start' in temp['timestamp'] and 'end' in temp['timestamp']:
                timestamp = [temp['timestamp']
                             ['start'], temp['timestamp']['end']]
            else:
                # Try to get the first two values
                values = list(temp['timestamp'].values())
                if len(values) >= 2:
                    timestamp = [float(values[0]), float(values[1])]
                else:
                    print(
                        f"[ERROR] render_video_svc: Cannot extract timestamp from dict: {temp['timestamp']}")
                    continue
            print(
                f"[DEBUG] render_video_svc: Extracted timestamp from dictionary: {timestamp}")
        else:
            print(
                f"[ERROR] render_video_svc: Unsupported timestamp format: {type(temp['timestamp'])}")
            continue

        # Ensure timestamp contains valid numbers
        try:
            start_time = float(timestamp[0])
            end_time = float(timestamp[1])
            stamps.append([start_time, end_time])
            ## check filename
            videos = supabase.find_source(temp['session_id'])
            sources.append(videos[0]['filename'])
            valid_uids.append(uuid)
            print(
                f"[DEBUG] render_video_svc: Added valid timestamp: [{start_time}, {end_time}]")
        except (ValueError, IndexError, TypeError) as e:
            print(
                f"[ERROR] render_video_svc: Invalid timestamp values: {timestamp}, error: {str(e)}")
            continue

    if not sources:
        print(
            f"[ERROR] render_video_svc: No valid scenes found for session {session_id}")
        return 2, session_id

    print(f"[DEBUG] render_video_svc: Processing {len(sources)} valid scenes")

    # Create video clips from each scene
    clips = []
    for i, (source, stamp) in enumerate(zip(sources, stamps)):
        ## check filename
        source_path = os.path.join(SERVICE_DIR, VID_SRC_DIR, session_id, f'{source}')
        print(f"[DEBUG] render_video_svc: Processing clip {i+1}/{len(sources)}: {source_path}")
        
        if not os.path.exists(source_path):
            print(
                f"[ERROR] render_video_svc: Source file not found: {source_path}")
            return 3, source

        print(
            f"[DEBUG] render_video_svc: Creating clip with timestamp {stamp[0]} to {stamp[1]}")
        try:
            clip = VideoFileClip(source_path).subclipped(stamp[0], stamp[1])

            if i == len(sources) - 1 and offset != 0:
                print(
                    f"[DEBUG] render_video_svc: This is the last clip, applying offset of {offset}")
                sub_clip = clip.subclipped(0, clip.duration - offset)
                clips.append(sub_clip)
            else:
                clips.append(clip)

            print(
                f"[DEBUG] render_video_svc: Added clip with duration {clip.duration}")
        except Exception as e:
            print(f"[ERROR] render_video_svc: Failed to create clip: {str(e)}")
            return 4, session_id

    if not clips:
        print(f"[ERROR] render_video_svc: No clips were created")
        return 4, session_id

    # Concatenate clips
    try:
        print(f"[DEBUG] render_video_svc: Concatenating {len(clips)} clips")
        final_clip = concatenate_videoclips(clips, method='compose')
        video_duration = final_clip.duration
        minute = int(video_duration / 60)
        second = int(video_duration % 60)
        duration = f"{minute:02}:{second:02}"
        print(f"[DEBUG] render_video_svc: Final video duration: {duration}")

        # Store duration in database
        # mongo.add_sum_duration(session_id, duration)
        supabase.add_sum_duration(session_id, duration)
    except Exception as e:
        print(
            f"[ERROR] render_video_svc: Failed to concatenate clips: {str(e)}")
        return 4, session_id

    # Replace audio if requested
    if replace:
        try:
            print(f"[DEBUG] render_video_svc: Replacing audio with TTS")
            tts_path = os.path.join(
                SERVICE_DIR, VOICE_DIR, f'{session_id}.wav')

            if not os.path.exists(tts_path):
                print(
                    f"[WARNING] render_video_svc: TTS audio file doesn't exist, generating it")
                path, audio_url = gen_voice_svc(session_id, "7")
                print(
                    f"[DEBUG] render_video_svc: Generated voice at path '{path}'")
                tts_path = path

            print(
                f"[DEBUG] render_video_svc: Loading TTS audio from {tts_path}")
            tts_wav = AudioFileClip(tts_path)
            supabase.add_or_update_section(session_id, "processing", "end-render_video_svc")
            final_clip.audio = tts_wav
            print(f"[DEBUG] render_video_svc: Successfully replaced audio")
        except Exception as e:
            print(
                f"[ERROR] render_video_svc: Failed to replace audio: {str(e)}")
            print("[WARNING] render_video_svc: Continuing without audio replacement")
            # Continue without audio replacement

    # Write the final video
    try:
        print(
            f"[DEBUG] render_video_svc: Writing final video to {output_path}")
        final_clip.write_videofile(output_path, audio_codec='aac')
        print(f"[DEBUG] render_video_svc: Successfully wrote video file")

        # Clean up
        print(f"[DEBUG] render_video_svc: Cleaning up resources")
        final_clip.close()
        for clip in clips:
            clip.close()
        if replace and 'tts_wav' in locals() and tts_wav:
            tts_wav.close()
    except Exception as e:
        print(
            f"[ERROR] render_video_svc: Failed to write video file: {str(e)}")
        return 4, session_id

    print(
        f"[DEBUG] render_video_svc: Successfully rendered video for session {session_id}")
    return output_path, video_name + '.mp4'


def get_sum_duration_svc(session_id: str):
    """
    Can be used after render_video_svc.
    Gets the duration of result video from DB.

    Parameters:
    session_id (str): The user's session id.

    Returns:
    str: Duration in "MM:SS" format.
    """
    supabase.add_or_update_section(session_id, "processing", "start_summary-duration")
    exist = supabase.find_summary(session_id)
    # exist = mongo.find_summary(session_id)

    supabase.add_or_update_section(session_id, "processing", "end_summary-duration")
    return exist["duration"]

# -------------------CLEAR---------------------------#


def safe_remove(file_path) -> int:
    """
    Check if a file exists before deletion.

    Parameters:
    file_path: Path to the file.

    Returns:
    int: 1 if deletion succeed, 0 if file does not exist.
    """
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"SUCCESS: Deleted file '{file_path}'")
        return 1
    else:
        print(f"NOT-FOUND: Cannot find file '{file_path}'")
        return 0


def safe_dir_remove(dir_path):
    """
    Check if a directory exists before deletion.

    Parameters:
    dir_path: Path to the directory.

    Returns:
    int: 1 if deletion succeed, 0 if directory does not exist.
    """
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path, ignore_errors=True)
        print(f"SUCCESS: Deleted directory '{dir_path}'")
        return 1
    else:
        print(f"NOT-FOUND: Cannot find directory '{dir_path}'")
        return 0


def clear_svc(session_id: str, remove_voice=True):
    """
    Delete files of the user with input session_id,
    as well as delete their metadata from DB:
     - Summarized video
     - Scenes / Chunks
     - Original videos
     - Script
     - Cached files
     - Text-To-Speech audio

    Parameters:
    session_id (str): The user's session id.
    """
    supabase.add_or_update_section(session_id, "processing", "start-clear_svc")

    # mongo.update_used(session_id)
    supabase.update_used(session_id)
    # videos = mongo.find_source(session_id)
    sources = supabase.find_source(session_id)

    # summary = mongo.find_summary(session_id)
    summary = supabase.find_summary(session_id)
    if summary:
        path = os.path.join(SERVICE_DIR, VID_SUM_DIR, summary['uid'] + ".mp4")
        if safe_remove(path):
            # mongo.del_summary(session_id)
            supabase.del_summary(session_id)

    # scenes = mongo.find_scene_owner(session_id)
    scenes = supabase.find_scene_owner(session_id)
    if scenes:
        if safe_dir_remove(os.path.join(SERVICE_DIR, VID_SCN_DIR, session_id)):
            # mongo.del_scene_owner(session_id)
            supabase.del_scene_owner(session_id)

    if sources:
        if safe_dir_remove(os.path.join(SERVICE_DIR, VID_SRC_DIR, session_id)):
            # mongo.del_sources(session_id)
            supabase.del_sources(session_id)

    # cript = mongo.find_script(session_id)
    script = supabase.find_script(session_id)
    if script:
        if safe_dir_remove(os.path.join(SERVICE_DIR, SCRIPT_DIR, session_id)):
            # mongo.del_script(session_id)
            supabase.del_script(session_id)

    safe_dir_remove(os.path.join(SERVICE_DIR, CACHE_DIR, session_id))

    if remove_voice:
        voice_path = os.path.join(SERVICE_DIR, VOICE_DIR, f'{session_id}.wav')
        safe_remove(voice_path)
    supabase.add_or_update_section(session_id, "success", "end-clear_svc", 100)


def delete_expired():
    src_dir = os.path.join(SERVICE_DIR, VID_SRC_DIR)
    uids = [name for name in os.listdir(src_dir)
            if os.path.isdir(os.path.join(src_dir, name))]

    i = 0
    for uid in uids:
        # if mongo.find_user(uid) is None:
        if supabase.find_user(uid) is None:
            print(f"[INFO]: Deleting data of expired user '{uid}'")
            clear_svc(uid)
            i += 1
    return i
