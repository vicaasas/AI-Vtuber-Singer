import os
import shutil
import json
import requests

from fastapi import FastAPI, Request, WebSocket
from starlette.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response

from .routes import init_client_ws_route, init_webtool_routes
from .service_context import ServiceContext
from .config_manager.utils import Config

from .only_uv5 import uvr
from pydub import AudioSegment
from pytubefix import YouTube

# from .rvc.infer_rvc import rvc_api

import os
import subprocess
import math
import librosa
# from concurrent.futures import ProcessPoolExecutor

import asyncio


class CustomStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if path.endswith(".js"):
            response.headers["Content-Type"] = "application/javascript"
        return response


class AvatarStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        allowed_extensions = (".jpg", ".jpeg", ".png", ".gif", ".svg")
        if not any(path.lower().endswith(ext) for ext in allowed_extensions):
            return Response("Forbidden file type", status_code=403)
        return await super().get_response(path, scope)


class WebSocketServer:
    def __init__(self, config: Config):
        self.app = FastAPI()

        # Add CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        SAVE_FOLDER = os.path.join(BASE_DIR, "downloaded_music")
        os.makedirs(SAVE_FOLDER, exist_ok=True)
        save_root_vocal=save_root_ins=f"{SAVE_FOLDER}/sing_opt"

        # format0="wav"
        # SAVE_FOLDER = "downloaded_music"
        # os.makedirs(SAVE_FOLDER, exist_ok=True)

        active_websockets = set()
        @self.app.websocket("/ws_music")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            active_websockets.add(websocket)
            try:
                while True:
                    await websocket.receive_text()  # 可根據需要處理 client 訊息
            except:
                active_websockets.remove(websocket)

        @self.app.post("/api/music")
        async def get_yt_music(request: Request):
            data = await request.json()
            output_folder = r"C:\Users\victo\Desktop\AI\Open-LLM-VTuber\src\open_llm_vtuber\downloaded_music"
            yt = YouTube(data['url'])
            audio_stream = yt.streams.filter(only_audio=True).first()

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            downloaded_file = audio_stream.download(output_path=output_folder)

            # 轉 mp3
            base, _ = os.path.splitext(downloaded_file)
            mp3_file = base + ".mp3"
            AudioSegment.from_file(downloaded_file).export(mp3_file, format="mp3")
            os.remove(downloaded_file)
            print(f"✅ MP3 saved at: {mp3_file}")
            # uvr(save_root_vocal, mp3_file, save_root_ins, format0)
            # batch_uvr(mp3_file)
            for ws in active_websockets.copy():
                await batch_uvr(mp3_file, ws)

            # rvc_api(dir_input=save_root_vocal, opt_input=save_root_ins)


        def split_audio(path, segment_length=10, output_dir=os.path.join(BASE_DIR, "split_clips")):
            os.makedirs(output_dir, exist_ok=True)
            y, sr = librosa.load(path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            n_segments = math.ceil(duration / segment_length)

            segment_paths = []
            for i in range(n_segments):
                start = i * segment_length
                output_path = os.path.join(output_dir, f"segment_{i:04d}.wav")
                command = [
                    "ffmpeg",
                    "-y",  # overwrite
                    "-i", path,
                    "-ss", str(start),
                    "-t", str(segment_length),
                    output_path,
                ]
                subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                segment_paths.append(output_path)

            return segment_paths

        async def process_and_send(segment_path, idx, websocket):
            loop = asyncio.get_running_loop()

            def blocking_task():
                # vocal_out = f"{save_root_vocal}"
                # inst_out = f"{save_root_ins}"
                uvr(save_root_vocal, segment_path, idx, save_root_ins, format0="wav")
                # return vocal_out, inst_out

            await loop.run_in_executor(None, blocking_task)

            try:
                await websocket.send_json({
                    "play_url": f"/music/sing_opt/vocal_{idx}.wav",
                    "play_background_url": f"/music/sing_opt/instrument_{idx}.wav"
                })
            except:
                print(f"⚠️ 無法傳送第 {idx} 段 WebSocket")

        async def batch_uvr(filename, websocket):
            segment_paths = split_audio(filename, segment_length=10)
            # tasks = []
            # for idx, seg_path in enumerate(segment_paths):
            #     task = asyncio.create_task(process_and_send(seg_path, str(idx), websocket))
            #     tasks.append(task)
            # await asyncio.gather(*tasks)
            for idx, seg_path in enumerate(segment_paths):
                await process_and_send(seg_path, str(idx), websocket)

        @self.app.post("/callback")
        async def receive_callback(request: Request):
            data = await request.json()
            print("✅ 收到 Callback 資料：")
            with open("data.json","w", encoding="utf-8")as f: 
                json.dump(data,f, indent=2, ensure_ascii=False)

            items = data.get("data", {}).get("data", [])
            # downloaded_titles = []
            item=items[0]
            # for item in items:
            title = item.get("title", "untitled").replace(" ", "_")
            audio_url = item.get("audio_url")

            if audio_url:
                filename = os.path.join(SAVE_FOLDER, f"{title}.mp3")
                print(f"🎵 正在下載：{filename}")
                try:
                    response = requests.get(audio_url)
                    if response.status_code == 200:
                        with open(filename, "wb") as f:
                            f.write(response.content)
                        print(f"✅ 下載完成：{filename}")

                        for ws in active_websockets.copy():
                            await batch_uvr(filename, ws)

                    else:
                        print(f"❌ 無法下載音樂，狀態碼：{response.status_code}")
                except Exception as e:
                    print(f"❌ 錯誤：{e}")

            return {"status": "received"}
        
        self.app.mount("/music", StaticFiles(directory=SAVE_FOLDER), name="music")

        # Load configurations and initialize the default context cache
        default_context_cache = ServiceContext()
        default_context_cache.load_from_config(config)


        
        # Include routes
        self.app.include_router(
            init_client_ws_route(default_context_cache=default_context_cache),
        )
        self.app.include_router(
            init_webtool_routes(default_context_cache=default_context_cache),
        )

        # Mount cache directory first (to ensure audio file access)
        if not os.path.exists("cache"):
            os.makedirs("cache")
        self.app.mount(
            "/cache",
            StaticFiles(directory="cache"),
            name="cache",
        )

        # Mount static files
        self.app.mount(
            "/live2d-models",
            StaticFiles(directory="live2d-models"),
            name="live2d-models",
        )
        self.app.mount(
            "/bg",
            StaticFiles(directory="backgrounds"),
            name="backgrounds",
        )
        self.app.mount(
            "/avatars",
            AvatarStaticFiles(directory="avatars"),
            name="avatars",
        )

        # Mount web tool directory separately from frontend
        self.app.mount(
            "/web-tool",
            CustomStaticFiles(directory="web_tool", html=True),
            name="web_tool",
        )

        # Mount main frontend last (as catch-all)
        self.app.mount(
            "/",
            CustomStaticFiles(directory="frontend", html=True),
            name="frontend",
        )

    def run(self):
        pass

    @staticmethod
    def clean_cache():
        """Clean the cache directory by removing and recreating it."""
        cache_dir = "cache"
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir)
