# AI Video Summarization (Back-end)

## Introduction
This is an **Application Programming Interface (API)** for AI video summarizaton, built with **FastAPI** with various structured endpoints for ease of access. Built to assist the creation of a short video from full-length videos while maintaining the core concepts. It is capable of transcribing audio, analyzing and extracting the key points of the input video, and reconstructing a shortened output video completed with a **Text-To-Speech (TTS)** voiceover.

## Table of Contents
- [Features](#features)
- [Installation](#quick-start)
    - [Detailed instructions](docs/installation.md)
    - [FFMPEG](docs/installation.md/#download-ffmpeg)
- [Usage](#usage)
    - [Use cases](docs/usage.md/#common-use-cases)
    - [Order of endpoints requesting](docs/usage.md/#order-of-endpoints-operations)
- [API References](docs/api_reference.md)
- [Cloud Database](docs/database.md)
- [Contributions](#contributions)
- [License](#license)

## Features
- **Audio**:
  - Upload and summarize scripts.
  - Transcribe audio files with optional speaker diarization.
  - Generate synthesized audio from summarized scripts.
- **Video**:
  - Session management for multiple user requests.
  - Upload and chunk video files.
  - Automated scene detection by jump-cut
  - Image captioning and conversion to vector embeddings.
  - Similarity search of embeddings stored on database.
  - Automatic chunk selection based on correlation to dubbing audio.
  - Render summarized videos based on chunks concatenation.

## Quick Start
Start by installing all the dependencies available on PyPI.
```pip install -r requirements.txt```

**FFMPEG** is another necessary package that is not available on PyPI, it can be downloaded [here](https://www.ffmpeg.org/download.html) or by following [this guide](docs/installation.md/#download-ffmpeg).

Next, duplicate the file named **.env.example** and renamed it as **.env**. External API keys and customizable model names must be filled in. 

To run the live server, enter this in the terminal
```uvicorn app:app```

The flag ```--reload``` can be used to reload the instance of API automatically when changes are made inside any files.
```uvicorn app:app --reload```

Another flag ```--workers 4``` can be used to allow parallel requests from many end-users.
```uvicorn app:app --workers 4```

## Usage
Without a front-end instance, the endpoints can be accessed locally via
```localhost:port/docs```

## Contributions
Inset information on how others can contribute, report bugs, or request features.

## License
Insert licensing information.
