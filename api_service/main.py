from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict
import uuid
import json
import os
import shutil
from datetime import datetime
from enum import Enum
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="InfiniteTalk API", version="1.0.0")

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: int
    created_at: str
    updated_at: str
    result_url: Optional[str] = None
    error_message: Optional[str] = None

tasks_db: Dict[str, Dict] = {}

UPLOAD_DIR = "temp_uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.get("/")
async def root():
    return {"service": "InfiniteTalk API", "status": "running", "version": "1.0.0"}

@app.post("/api/task/create", response_model=TaskResponse)
async def create_task(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    image: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    resolution: str = "480",
    sample_steps: int = 40
):
    task_id = str(uuid.uuid4())
    audio_path = os.path.join(UPLOAD_DIR, f"{task_id}_audio.wav")
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)
    
    input_path = None
    input_type = None
    if image:
        input_type = "image"
        input_path = os.path.join(UPLOAD_DIR, f"{task_id}_input.jpg")
        with open(input_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
    elif video:
        input_type = "video"
        input_path = os.path.join(UPLOAD_DIR, f"{task_id}_input.mp4")
        with open(input_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
    else:
        raise HTTPException(status_code=400, detail="必須提供 image 或 video")
    
    now = datetime.utcnow().isoformat()
    task = {
        "task_id": task_id,
        "status": TaskStatus.PENDING,
        "progress": 0,
        "created_at": now,
        "updated_at": now,
        "audio_path": audio_path,
        "input_path": input_path,
        "input_type": input_type,
        "resolution": resolution,
        "sample_steps": sample_steps,
        "result_url": None,
        "error_message": None
    }
    tasks_db[task_id] = task
    background_tasks.add_task(process_task, task_id)
    return TaskResponse(**task)

@app.get("/api/task/status/{task_id}", response_model=TaskResponse)
async def get_task_status(task_id: str):
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="任務不存在")
    return TaskResponse(**tasks_db[task_id])

@app.get("/api/task/result/{task_id}")
async def get_task_result(task_id: str):
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="任務不存在")
    task = tasks_db[task_id]
    if task["status"] != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="任務尚未完成")
    output_file = os.path.join(OUTPUT_DIR, f"{task_id}_output.mp4")
    if not os.path.exists(output_file):
        raise HTTPException(status_code=404, detail="結果文件不存在")
    return FileResponse(output_file, media_type="video/mp4", filename=f"infinitetalk_{task_id}.mp4")

@app.delete("/api/task/{task_id}")
async def delete_task(task_id: str):
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="任務不存在")
    task = tasks_db[task_id]
    files_to_delete = [task.get("audio_path"), task.get("input_path"), os.path.join(OUTPUT_DIR, f"{task_id}_output.mp4")]
    for file_path in files_to_delete:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.error(f"刪除失敗 {file_path}: {e}")
    del tasks_db[task_id]
    return {"success": True, "message": "任務已刪除"}

@app.get("/api/tasks/list")
async def list_tasks(status: Optional[TaskStatus] = None, limit: int = 100):
    tasks = list(tasks_db.values())
    if status:
        tasks = [t for t in tasks if t["status"] == status]
    tasks.sort(key=lambda x: x["created_at"], reverse=True)
    return {"total": len(tasks), "tasks": [TaskResponse(**t) for t in tasks[:limit]]}

async def process_task(task_id: str):
    task = tasks_db[task_id]
    try:
        task["status"] = TaskStatus.PROCESSING
        task["progress"] = 10
        task["updated_at"] = datetime.utcnow().isoformat()
        input_json = [{"audio": task["audio_path"], "input": task["input_path"], "type": task["input_type"]}]
        input_json_path = os.path.join(UPLOAD_DIR, f"{task_id}_input.json")
        with open(input_json_path, "w") as f:
            json.dump(input_json, f)
        output_path = os.path.join(OUTPUT_DIR, task_id)
        cmd = ["/workspace/infinitetalk-env/bin/python", "generate_infinitetalk.py", "--ckpt_dir", "weights/Wan2.1-I2V-14B-480P", "--wav2vec_dir", "weights/chinese-wav2vec2-base", "--infinitetalk_dir", "weights/InfiniteTalk/single/infinitetalk.safetensors", "--input_json", input_json_path, "--size", f"infinitetalk-{task['resolution']}", "--sample_steps", str(task["sample_steps"]), "--mode", "streaming", "--motion_frame", "9", "--num_persistent_param_in_dit", "0", "--save_file", output_path]
        task["progress"] = 30
        logger.info(f"開始處理任務 {task_id}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd="/workspace/InfiniteTalk")
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise Exception(f"生成失敗: {stderr}")
        generated_file = f"{output_path}.mp4"
        final_output = os.path.join(OUTPUT_DIR, f"{task_id}_output.mp4")
        if os.path.exists(generated_file):
            shutil.move(generated_file, final_output)
        else:
            raise Exception("找不到生成的影片")
        task["status"] = TaskStatus.COMPLETED
        task["progress"] = 100
        task["result_url"] = f"/api/task/result/{task_id}"
        task["updated_at"] = datetime.utcnow().isoformat()
        logger.info(f"任務 {task_id} 完成")
    except Exception as e:
        logger.error(f"任務 {task_id} 失敗: {e}")
        task["status"] = TaskStatus.FAILED
        task["error_message"] = str(e)
        task["updated_at"] = datetime.utcnow().isoformat()
