import os
import json
import traceback
import pytz
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv(override=True)

# Fetch environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")


supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_bangkok_time():
    """
    Returns the current time in Asia/Bangkok timezone (UTC+7) in ISO 8601 format.

    Returns:
        str: Current Bangkok time in ISO 8601 format.
    """
    # Get current UTC time
    utc_now = datetime.utcnow()

    # Define Bangkok timezone
    bangkok_tz = pytz.timezone('Asia/Bangkok')

    # Convert UTC time to Bangkok time
    bangkok_now = utc_now.replace(tzinfo=pytz.UTC).astimezone(bangkok_tz)

    # Return in ISO format
    return bangkok_now.isoformat()


class SupabaseConnector:
    def __init__(self):
        """
        Initializes the SupabaseConnector by loading environment variables 
        and establishing a Supabase client.
        """
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError(
                "Supabase URL or API Key is not properly configured.")

        self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # CRUD Operations 'C': CREATE
    def add_user(self, session_id: str):
        """
        Add a new user to the 'ai_backend_users' table.

        Parameters:
        session_id (str): Random uuid4 string.
        """
        now = get_bangkok_time()
        self.client.table("ai_backend_users").insert({
            "session_id": session_id,
            "created_at": now,
            "updated_at": now
        }).execute()

    def add_source(self, docs: list):
        """
        Add multiple source videos to the 'ai_backend_source_videos' table.

        Parameters:
        docs (list[dict]): List of source video metadata.
        """
        self.client.table("ai_backend_source_videos").insert(docs).execute()

    def add_scene(self, entries: list):
        """
        Add multiple scenes to the 'ai_backend_scene_videos' table.

        Parameters:
        entries (list[dict]): List of scene metadata.
        """
        self.client.table("ai_backend_scene_videos").insert(entries).execute()

    def add_script(self, info: dict):
        """
        Add a script to the 'ai_backend_scripts' table.

        Parameters:
        info (dict): Script metadata.
        """
        self.client.table("ai_backend_scripts").insert(info).execute()

    def add_offset(self, uid: str, auto_scenes: list, session_id: str, offset: float):
        """
        Add a summary video to the 'ai_backend_summary_videos' table.

        Parameters:
        uid (str): UUID for the summary video.
        auto_scenes (list): List of scene UIDs to be concatenated.
        session_id (str): Session ID of the user.
        offset (float): Offset duration for the last scene.
        """
        self.client.table("ai_backend_summary_videos").insert({
            "uid": uid,
            "selected_scenes": auto_scenes,
            "session_id": session_id,
            "offset": offset,
            "created_at": get_bangkok_time()
        }).execute()

    # From supabase_controller.py
    def add_or_update_section(self, session_id: str, status: str, processing_session: str = None, progress: int = 0, create_at: datetime = None):
        """
        Ensures one row per session_id.
        Updates status and related fields if session_id exists.
        Inserts new row with provided parameters if not.

        Parameters:
        session_id (str): Session ID of the user.
        status (str): Status to be stored or updated.
        processing_session (str, optional): Processing session information.
        progress (int, optional): Processing progress percentage (0-100).
        create_at (datetime, optional): Creation timestamp. Defaults to current time.
        """
        if create_at is None:
            create_at = get_bangkok_time()
        elif isinstance(create_at, datetime):
            # Convert datetime to ISO string if it's a datetime object
            create_at = create_at.isoformat()

        # Check if session_id already exists
        existing = self.client.table("status").select("*")\
            .eq("session_id", session_id)\
            .execute()

        if existing.data:
            # Update status
            update_data = {
                "status": status,
                "progress": progress
            }

            # Only add processing_session if provided
            if processing_session is not None:
                update_data["processing_session"] = processing_session

            # Add create_at
            update_data["create_at"] = create_at

            self.client.table("status").update(update_data).eq(
                "session_id", session_id).execute()
        else:
            # Insert new row
            insert_data = {
                "session_id": session_id,
                "status": status,
                "progress": progress,
                "create_at": create_at
            }

            # Only add processing_session if provided
            if processing_session is not None:
                insert_data["processing_session"] = processing_session

            self.client.table("status").insert(insert_data).execute()

    # From supabase_controller.py
    def log_to_supabase(self, session_id: str, log_type: str, message: str):
        """
        Logs a message to the 'session_logs' table.

        Parameters:
        session_id (str): Session ID of the user.
        log_type (str): Type of log (e.g., "info", "error", "warning").
        message (str): Log message content.
        """
        now = datetime.now().isoformat()
        supabase.table("session_logs").insert({
            "session_id": session_id,
            "log_type": log_type,
            "message": message,
            "created_at": now
        }).execute()

    # From supabase_controller.py
    def insert_result(self, session_id: str, transcript: str, summerized: str, audio_url: str):
        """
        Inserts a result into the 'result' table.

        Parameters:
        session_id (str): Session ID of the user.
        transcript (str): Transcribed text from the video.
        summerized (str): Summarized text from the transcription.
        audio_url (str): URL to the generated audio file.
        """
        now = get_bangkok_time()
        self.client.table("result").insert({
            "session_id": session_id,
            "transcribe_text": transcript,
            "sumarized_text": summerized,
            "audio_url": audio_url,
            "created_at": now,
            "updated_at": now
        }).execute()

    def insert_prompt(self, prompt_infos: list):
        """
        Insert a framepack data into the 'framepack_data' table.

        Parameters:
        prompt_infos: list of meta data of framepack [
            uid (str): UUID of the framepack_sceces
            session_id (str): Session ID of the user.
            image_prompt (str): Image prompt from summary text
            framepack_prompt (str): FramePack prompt from image and summary text
            image_name (str): Filename of image
            video_name (str): Filename of video
        ]
        """
        self.client.table("framepack_data").insert(prompt_infos).execute()


    # CRUD Operations 'R': READ
    def match_fname(self, session_id: str, fname: str):
        """
        Find a source video by UID and session ID.

        Parameters:
        session_id (str): Session ID of the user.
        fname (str): Name of the source video.

        Returns:
        dict: Matching source video or None if not found.
        """
        response = self.client.table("ai_backend_source_videos") \
            .select("*") \
            .eq("session_id", session_id) \
            .eq("filename", fname) \
            .execute()
        
        return response.data

    def find_source(self, session_id: str):
        """
        Find all source videos for a session ID.

        Parameters:
        session_id (str): Session ID of the user.

        Returns:
        list: List of source videos.
        """
        response = self.client.table("ai_backend_source_videos").select(
            "*").eq("session_id", session_id).execute()
        return response.data

    def find_scene_owner(self, session_id: str):
        """
        Find all scenes for a session ID.

        Parameters:
        session_id (str): Session ID of the user.

        Returns:
        list: List of scenes.
        """
        response = self.client.table("ai_backend_scene_videos").select(
            "*").eq("session_id", session_id).execute()
        return response.data

    def find_single_scene(self, scene_uuid: str):
        """
        Find a single scene by UUID.

        Parameters:
        scene_uuid (str): UUID of the scene.

        Returns:
        dict: Scene data or None if not found.
        """
        print(
            f"[DEBUG] SupabaseConnector.find_single_scene: Looking for scene {scene_uuid}")
        response = self.client.table("ai_backend_scene_videos").select(
            "*").eq("uid", scene_uuid).execute()

        if response.data:
            print(
                f"[DEBUG] SupabaseConnector.find_single_scene: Found scene with source_uid {response.data[0]['source_uid']}")
            return response.data[0]

        print(
            f"[DEBUG] SupabaseConnector.find_single_scene: No scene found with UUID {scene_uuid}")
        return None

    def find_script(self, session_id: str):
        """
        Find a script by session ID.

        Parameters:
        session_id (str): Session ID of the user.

        Returns:
        dict: Script metadata or None if not found.
        """
        response = self.client.table("ai_backend_scripts").select(
            "*").eq("session_id", session_id).execute()
        return response.data[0] if response.data else None

    def find_summary(self, session_id: str):
        """
        Find a summary video by session ID.

        Parameters:
        session_id (str): Session ID of the user.

        Returns:
        dict: Summary video data or None if not found.
        """
        print(
            f"[DEBUG] SupabaseConnector.find_summary: Looking for session {session_id}")
        response = self.client.table("ai_backend_summary_videos").select(
            "*").eq("session_id", session_id).execute()

        if response.data:
            print(
                f"[DEBUG] SupabaseConnector.find_summary: Found summary with UID {response.data[0]['uid']}")
            # Print schema info for debugging
            print(
                f"[DEBUG] SupabaseConnector.find_summary: Schema keys: {list(response.data[0].keys())}")
            if 'selected_scenes' in response.data[0]:
                scenes = response.data[0]['selected_scenes']
                print(
                    f"[DEBUG] SupabaseConnector.find_summary: selected_scenes has {len(scenes) if isinstance(scenes, list) else 'non-list'} items")
            return response.data[0]

        print(
            f"[DEBUG] SupabaseConnector.find_summary: No summary found for session {session_id}")
        return None

    def find_user(self, session_id: str):
        """
        Find a user by session ID.

        Parameters:
        session_id (str): Session ID of the user.

        Returns:
        dict: User metadata or None if not found.
        """
        response = self.client.table("ai_backend_users").select(
            "*").eq("session_id", session_id).execute()
        return response.data[0] if response.data else None
        
    def search_vector(self, pipeline: list):
        """
        Enhanced vector search function that uses more efficient methods
        while maintaining compatibility with the MongoDB-style pipeline
        generated by gen_search_pipeline().

        This version attempts to use Supabase's pgvector capabilities directly
        when available, with a fallback to client-side similarity calculation.

        Parameters:
        pipeline (list): Search pipeline from gen_search_pipeline() function

        Returns:
        list: Results of top-n similar vectors based on the search criteria
        """
        print(
            f"[DEBUG] SupabaseConnector.search_vector: Processing pipeline: {pipeline}")

        # Extract search parameters from the pipeline
        query_vector = None
        session_id = None
        source_uids = None
        limit = 10
        projection = None

        # Parse the MongoDB-style pipeline to extract parameters
        for stage in pipeline:
            # Extract vector search parameters
            if "$vectorSearch" in stage:
                query_vector = stage["$vectorSearch"]["queryVector"]
                if "limit" in stage["$vectorSearch"]:
                    limit = stage["$vectorSearch"]["limit"]

            # Extract match conditions
            if "$match" in stage:
                if "session_id" in stage["$match"] and "$eq" in stage["$match"]["session_id"]:
                    session_id = stage["$match"]["session_id"]["$eq"]

                if "source_uid" in stage["$match"] and "$in" in stage["$match"]["source_uid"]:
                    source_uids = stage["$match"]["source_uid"]["$in"]

            # Extract limit if specified
            if "$limit" in stage:
                limit = stage["$limit"]

            # Extract projection if specified
            if "$project" in stage:
                projection = stage["$project"]

        # Validate that we have the necessary parameters
        if query_vector is None:
            print(
                "[ERROR] SupabaseConnector.search_vector: No query vector found in pipeline")
            return []

        if session_id is None:
            print(
                "[ERROR] SupabaseConnector.search_vector: No session ID found in pipeline")
            return []

        response = self.client.rpc('search_scene_video', {
            "input_vector": query_vector,
            "input_source_uid": source_uids,
            "input_session_id": session_id,
            "input_limit": limit
        }).execute()
        results = response.data
        print(
            f"[DEBUG] SupabaseConnector.search_vector: Search result: {results}")
        return results

    def find_img_prompt(self, session_id: str):
        """
        Find image prompt by session ID

        Parameters:
            session_id (str): Session ID of the user.

        Returns:
        dict: Image prompt metadata or None if not found.
        """
        response = self.client.table("framepack_data").select(
            "*").eq("session_id", session_id).execute()
        return response.data[0] if response.data else None

    # CRUD Operations 'U': UPDATE
    def update_used(self, session_id: str):
        """
        Update the 'updated_at' field for a user.

        Parameters:
        session_id (str): Session ID of the user.
        """
        self.client.table("ai_backend_users").update({
            "updated_at": get_bangkok_time()
        }).eq("session_id", session_id).execute()

    def done_chunk(self, source_uuid: str, session_id: str):
        """
        Mark a source video as chunked.

        Parameters:
        source_uuid (str): UID of the source video.
        session_id (str): Session ID of the user.
        """
        self.client.table("ai_backend_source_videos").update({"chunked": True}).eq(
            "uid", source_uuid).eq("session_id", session_id).execute()

    def add_vqa_embed(self, scene_uuid: str, caption: str, embed: list):
        """
        Add caption and embedding to a scene.

        Parameters:
        scene_uuid (str): UID of the scene.
        caption (str): Caption for the scene.
        embed (list): Vector embedding for the caption.
        """
        self.client.table("ai_backend_scene_videos").update(
            {"caption": caption, "embedding": embed}).eq("uid", scene_uuid).execute()

    def done_llm(self, session_id: str):
        """
        Mark a script as summarized.

        Parameters:
        session_id (str): Session ID of the user.
        """
        self.client.table("ai_backend_scripts").update(
            {"summed": True}).eq("session_id", session_id).execute()

    def add_sum_duration(self, session_id: str, duration: str):
        """
        Add duration to a summary video.

        Parameters:
        session_id (str): Session ID of the user.
        duration (str): Duration in MM:SS format.
        """
        self.client.table("ai_backend_summary_videos").update(
            {"duration": duration}).eq("session_id", session_id).execute()

    def update_summary(self, session_id: str, scenes: list):
        """
        Update the selected scenes for a summary video.

        Parameters:
        session_id (str): Session ID of the user.
        scenes (list): List of scene UIDs.
        """
        print(
            f"[DEBUG] SupabaseConnector.update_summary: Updating session {session_id} with {len(scenes) if isinstance(scenes, list) else 'non-list'} scenes")

        if not isinstance(scenes, list):
            print(
                f"[WARNING] SupabaseConnector.update_summary: scenes parameter is not a list: {type(scenes)}")
            if isinstance(scenes, str):
                print(
                    f"[WARNING] SupabaseConnector.update_summary: converting string to list: {scenes}")
                try:
                    scenes = json.loads(scenes)
                except:
                    scenes = [scenes]
            else:
                scenes = []

        # Verify scenes is a proper list before update
        print(f"[DEBUG] Full scenes list before update: {scenes}")

        # Convert any non-list scenes to list if needed
        if scenes and not isinstance(scenes[0], list):
            scenes_list = list(scenes)  # Create explicit copy
        else:
            scenes_list = scenes

        try:
            # Explicitly format the update data
            update_data = {
                "selected_scenes": scenes_list,
                "duration": ""
            }
            print(f"[DEBUG] Update data being sent: {update_data}")

            # Execute update with explicit data
            response = self.client.table("ai_backend_summary_videos") \
                .update(update_data) \
                .eq("session_id", session_id) \
                .execute()

            # Verify the update
            updated_record = self.find_summary(session_id)
            print(
                f"[DEBUG] Verification - stored scenes: {updated_record['selected_scenes'] if updated_record else 'No record found'}")

            return response.data

        except Exception as e:
            print(
                f"[ERROR] SupabaseConnector.update_summary: Failed to update summary: {str(e)}")
            traceback.print_exc()
            raise

    # From supabase_controller.py
    def update_status(self, session_id: str, status: str):
        """
        Ensures one row per session_id.
        Updates status if session_id exists.
        Inserts new row with session_id and status if not.
        """
        # Check if session_id already exists
        existing = self.client.table("status").select("*")\
            .eq("session_id", session_id)\
            .execute()

        if existing.data:
            # Update status
            update_response = self.client.table("status").update({
                "status": status
            }).eq("session_id", session_id).execute()
        else:
            insert_response = self.client.table("status").insert({
                "session_id": session_id,
                "status": status
            }).execute()

    def update_script(self, session_id: str, data: dict):
        """
        Update or insert a row into ai_backend_scripts using a dict input.
        """
        existing = self.client.table("ai_backend_scripts").select("*")\
            .eq("session_id", session_id)\
            .execute()

        if existing.data:
            self.client.table("ai_backend_scripts").update(data)\
                .eq("session_id", session_id).execute()
        else:
            data["uid"] = session_id
            data["session_id"] = session_id
            self.client.table("ai_backend_scripts").insert(data).execute()
    
    def update_framepack_prompt(self, session_id: str, data: dict):
        """
        Updates prompt and related fields if session_id exists.

        Parameters:
        session_id (str): Session ID of the user.
        image_prompt (str): Image prompt from summary text
        """
        existing = self.client.table("framepack_data").select("*")\
            .eq("session_id", session_id)\
            .execute()
        if existing.data:
            self.client.table("framepack_data").update(data)\
                .eq("session_id", session_id).execute()
        else:
            data["uid"] = session_id
            data["session_id"] = session_id
            self.client.table("framepack_data").insert(data).execute()

    # CRUD Operations 'D': DELETE
    def del_sources(self, session_id: str):
        """
        Delete all source videos for a session ID.

        Parameters:
        session_id (str): Session ID of the user.
        """
        self.client.table("ai_backend_source_videos").delete().eq(
            "session_id", session_id).execute()

    def del_scene_owner(self, session_id: str):
        """
        Delete all scenes for a session ID.

        Parameters:
        session_id (str): Session ID of the user.
        """
        self.client.table("ai_backend_scene_videos").delete().eq(
            "session_id", session_id).execute()

    def del_script(self, session_id: str):
        """
        Delete a script for a session ID.

        Parameters:
        session_id (str): Session ID of the user.
        """
        self.client.table("ai_backend_scripts").delete().eq(
            "session_id", session_id).execute()

    def del_summary(self, session_id: str):
        """
        Delete a summary video for a session ID.

        Parameters:
        session_id (str): Session ID of the user.
        """
        self.client.table("ai_backend_summary_videos").delete().eq(
            "session_id", session_id).execute()

    def del_user(self, session_id: str):
        """
        Delete a user by session ID.

        Parameters:
        session_id (str): Session ID of the user.
        """
        self.client.table("ai_backend_users").delete().eq(
            "session_id", session_id).execute()
