import os
import dotenv

from pymongo import MongoClient
from datetime import datetime

dotenv.load_dotenv(override=True)

# Database
MONGODB_URI = os.getenv("MONGODB_URL")
DB_NAME = os.getenv("DB_NAME")
USER_COLLECTION = os.getenv("USER_COLLECTION")
SRC_VIDEOS_COLLECTION = os.getenv("SRC_VIDEOS_COLLECTION")
SCENE_VIDEOS_COLLECTION = os.getenv("SCENE_VIDEOS_COLLECTION")
SUMMARY_VIDEOS_COLLECTION = os.getenv("SUMMARY_VIDEOS_COLLECTION")
SCRIPT_COLLECTION = os.getenv("SCRIPT_COLLECTION")

class DBConnector:
    def __init__(self):
        """
        Connect to the MongoDB cluster at MONGODB_URI
        Link the database and 5 collections inside.
        """
        self.client = MongoClient(MONGODB_URI)
        self.db = self.client[DB_NAME]
        self.user_coll = self.db[USER_COLLECTION]
        self.source_coll = self.db[SRC_VIDEOS_COLLECTION]
        self.scene_coll = self.db[SCENE_VIDEOS_COLLECTION]
        self.summary_coll = self.db[SUMMARY_VIDEOS_COLLECTION]
        self.script_coll = self.db[SCRIPT_COLLECTION]

    # CRUD Operations 'C': CREATE
    def add_user(self, session_id: str):
        """
        Add one document into 'User' collection
        With the time of creation in 2 other fields

        Parameters:
        session_id (str): random uuid4() string
        """
        self.user_coll.insert_one({
            "session_id": session_id,
            "createdAt": datetime.now(),
            "updatedAt": datetime.now()
            }
        )

    def add_source(self, docs: list):
        """
        Inserts a list of dict into 'SourceVideo' collection

        Parameters:
        docs (list[dict]): each element stores metadata of original video
        """
        self.source_coll.insert_many(docs)

    def add_scene(self, entries: list):
        """
        Inserts a list of 'SceneUploadEntry' into 'SceneVideo' collection

        Parameters:
        entries (list): each element is our Pydantic BaseModel (see schemas/model.py)
        """
        self.scene_coll.insert_many([entry.model_dump() for entry in entries])

    def add_script(self, info: dict):
        """
        Insert one script metadata into 'Script' collection

        Parameters:
        info (dict): script metadata and session_id it's tied to
        """
        self.script_coll.insert_one(info)

    def add_offset(self, uid: str, auto_scenes: list, session_id: str, offset: float):
        """
        Insert one document into 'SummaryVideo' collection

        Parameters:
        uid (str): uuid4() string that will be result video name
        auto_scenes (list): a list containing uid of scenes to be concatenated,
                            blank if no auto_match_svc
        session_id (str): the owner of this result video
        offset (float): duration to be cut from the end of the last scene
        """
        self.summary_coll.insert_one({"uid": uid,
                                      "auto_scenes": auto_scenes,
                                      "session_id": session_id,
                                      "offset": offset})

    # CRUD Operations 'R': READ
    def match_fname(self, session_id: str, fname: str):
        """
        Find a document with matching 'uid' and 'session_id' in 'SourceVideo' collection

        Parameters:
        session_id (str): uuid4() string representing a user
        fname (str): original video name without file extension

        Returns:
        list: contains all matches or [] if no match
        """
        return list(self.source_coll.find({"uid": fname, "session_id": session_id}))

    def find_source(self, session_id: str):
        """
        Find all document in 'SourceVideo' that belongs to this session_id

        Parameters:
        session_id (str): uuid4() string representing a user

        Returns:
        list: contains all matches or [] if no match
        """
        return list(self.source_coll.find({"session_id": session_id}))
    
    def find_scene_owner(self, session_id: str):
        """
        Find all document in 'SceneVideo' that belongs to this session_id

        Parameters:
        session_id (str): uuid4() string representing a user

        Returns:
        list: contains all matches or [] if no match
        """
        return list(self.scene_coll.find({"session_id": session_id}))
    
    def find_single_scene(self, video_uuid: str):
        """
        Find one document in 'SceneVideo' collection based on uid

        Parameters:
        video_uuid (str): name of the scene (should be uuid4() string)

        Returns:
        Any: a single document that matches
        None: if no match
        """
        return self.scene_coll.find_one({"uid": video_uuid})
    
    def find_script(self, session_id: str):
        """
        Find one document in 'Script' collection based on session_id

        Parameters:
        session_id (str): uuid4() string representing a user

        Returns:
        Any: a single document that matches
        None: if no match
        """
        return self.script_coll.find_one({"session_id": session_id})
    
    def search_vector(self, pipeline: list):
        """
        Uses pipeline to perform Atlas search in 'SceneVideo' collection

        Parameters:
        pipeline (list): search pipeline from def gen_search_pipeline(...)

        Returns:
        list: results of top-n similarities
        """
        return list(self.scene_coll.aggregate(pipeline))
    
    def find_summary(self, session_id: str):
        """
        Find one document in 'SummaryVideo' collection based on session_id

        Parameters:
        session_id (str): uuid4() string representing a user

        Returns:
        Any: a single document that matches
        None: if no match
        """
        return self.summary_coll.find_one({"session_id": session_id})
    
    def find_user(self, session_id: str):
        """
        Find exact user by 'session_id' in 'User' collection

        Parameters:
        session_id (str): uuid4() string representing a user

        Returns:
        Any: a single document that matches
        None: if no match
        """
        return self.user_coll.find_one({"session_id": session_id})

    # CRUD Operations 'U': UPDATE
    def update_used(self, session_id: str):
        """
        Update the time of last usage ("updatedAt" field)

        Parameters:
        session_id (str): ID of the user to update
        """
        self.user_coll.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "updatedAt": datetime.now()
                }
            }
        )

    def done_chunk(self, source_uuid: str, session_id: str):
        """
        Update the "chunked" field of one document
        with a matching 'session_id' and 'source_uuid' in 'SceneVideo'
        indicating that chunk_videos_svc had been done

        Parameters:
        source_uuid (str): name of the source video
        session_id (str): uuid4() string representing a user
        """
        self.source_coll.update_one(
          {"uid": source_uuid,
           "session_id": session_id},
            {
               "$set": {
                    "chunked": True,
                }
            },
     )
        
    def add_vqa_embed(self, scene_uuid: str, caption, embed):
        """
        Add caption and vector embeddings of that caption
        to a document in 'SceneVideo' with a matching 'scene_uuid'

        Parameters:
        scene_uuid (str): name of the scene
        """
        self.scene_coll.update_one(
          {"uid": scene_uuid},
            {
               "$set": {
                    "caption": caption,
                    "embedding": embed
                }
            },
     )
        
    def done_llm(self, session_id: str):
        """
        Update the "summed" field of one document
        with a matching 'session_id' in 'Script'
        indicating that summarize_svc had been done

        Parameters:
        session_id (str): uuid4() string representing a user
        """
        self.script_coll.update_one(
          {"session_id": session_id},
            {
               "$set": {
                    "summed": True
                }
            },
     )
        
    def add_sum_duration(self, session_id: str, duration: str):
        """
        Add duration onto a document in 'SummaryVideo'
        with a matching 'session_id'

        Parameters:
        session_id (str): uuid4() string representing a user
        duration (str): "MM:SS" formatted string
        """
        self.summary_coll.update_one(
          {"session_id": session_id},
            {
               "$set": {
                    "duration": duration
                }
            },
     )
    
    def update_summary(self, session_id: str, scenes):
        """
        Update the "selected_scenes" field of one document
        in 'SummaryVideo' with a matching 'session_id'

        Parameters:
        session_id (str): uuid4() string representing a user
        scenes (list): list of uid of scenes to be concatenated
        """
        self.summary_coll.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "selected_scene": scenes,
                    "duration": ""
                }
            }
        )    

    # CRUD Operations 'D': DELETE
    def del_sources(self, session_id: str):
        """
        Delete every document in 'SourceVideo' tied to this session_id

        Parameters:
        session_id (str): uuid4() string representing a user
        """
        self.source_coll.delete_many({"session_id": session_id})
    
    def del_scene_owner(self, session_id: str):
        """
        Delete every document in 'SceneVideo' tied to this session_id

        Parameters:
        session_id (str): uuid4() string representing a user
        """
        self.scene_coll.delete_many({"session_id": session_id})

    def del_script(self, session_id: str):
        """
        Delete one document in 'Script' tied to this session_id

        Parameters:
        session_id (str): uuid4() string representing a user
        """
        self.script_coll.delete_one({"session_id": session_id})

    def del_summary(self, session_id: str):
        """
        Delete one document in 'SummaryVideo' tied to this session_id

        Parameters:
        session_id (str): uuid4() string representing a user
        """
        self.summary_coll.delete_one({"session_id": session_id})

    def del_user(self, session_id: str):
        """
        Delete one document in 'User' tied to this session_id

        Parameters:
        session_id (str): uuid4() string representing a user
        """
        self.user_coll.delete_one({"session_id": session_id})