from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
import os
import sys
import asyncio
import httpx
import json
import time
import mimetypes
import subprocess
import tempfile
import re
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse
from urllib.parse import urljoin

load_dotenv()

EXPECTED_EMAIL = os.getenv("EXPECTED_EMAIL")
MY_SECRET = os.getenv("MY_SECRET")
AIPIPE_API_KEY = os.getenv("AIPIPE_API_KEY")
AIPIPE_BASE_URL = os.getenv("AIPIPE_BASE_URL", "")

class SubmissionRetryableError(Exception):
    pass

class Payload(BaseModel):
    email: str
    secret: str
    url: str

app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid JSON format or missing required fields."},
    )

async def install_package(package_name):
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pip", "install", package_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()
        return process.returncode == 0
    except Exception:
        return False

async def execute_generated_code(code_str, timeout=30):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp:
        tmp.write(code_str)
        tmp_path = tmp.name

    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable, tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        
        stdout_dec = stdout.decode().strip()
        stderr_dec = stderr.decode().strip()
        
        if process.returncode != 0:
            if "ModuleNotFoundError" in stderr_dec:
                match = re.search(r"No module named '([^']+)'", stderr_dec)
                if match:
                    module_name = match.group(1)
                    if await install_package(module_name):
                        return await execute_generated_code(code_str, timeout)
            return False, f"Error: {stderr_dec}"
        
        return True, stdout_dec

    except asyncio.TimeoutError:
        return False, "Execution timed out."
    except Exception as e:
        return False, str(e)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

async def perform_coding_task(task_description, context_str, start_time):
    messages = [
        {
            "role": "system", 
            "content": f"""You are a Python coding expert. Write a script to solve the following task.
            Task: {task_description}
            
            Context Data (URLs/Paths): {context_str}
            
            Requirements:
            1. Write code to fetch any required files (CSV, etc.) from the provided URLs.
            2. Perform the required operation (sum, process, etc.).
            3. Print the FINAL RESULT to stdout.
            4. If dealing with CSVs without known headers, either inspect headers first or use column indices (0, 1, etc.).
            5. The code will be executed on a local machine. Assume internet access.
            6. Output ONLY valid Python code inside markdown blocks (```python ... ```).
            """
        }
    ]

    async with httpx.AsyncClient() as client:
        while time.time() - start_time < 180:
            current_elapsed = time.time() - start_time
            remaining_timeout = max(10.0, 180 - current_elapsed)

            try:
                response = await client.post(
                    f"{AIPIPE_BASE_URL.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {AIPIPE_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-5-nano",
                        "messages": messages
                    },
                    timeout=min(60.0, remaining_timeout)
                )
                response.raise_for_status()
                ai_content = response.json()["choices"][0]["message"]["content"]
                
                code_match = re.search(r"```python\n(.*?)```", ai_content, re.DOTALL)
                if not code_match:
                    code_match = re.search(r"```\n(.*?)```", ai_content, re.DOTALL)
                
                if code_match:
                    code = code_match.group(1)
                    success, output = await execute_generated_code(code, timeout=30)
                    
                    if success:
                        return output
                    else:
                        messages.append({"role": "assistant", "content": ai_content})
                        messages.append({"role": "user", "content": f"The code execution failed with this error:\n{output}\n\nPlease fix the code and try again."})
                else:
                    messages.append({"role": "assistant", "content": ai_content})
                    messages.append({"role": "user", "content": "I could not find a python code block. Please provide the python code in ```python ... ``` format."})
            
            except Exception as e:
                print(f"Coding agent error: {e}")
                break
                
    return "Failed to generate valid solution."

async def _process_data_attempt(payload: Payload, start_time: float):
    elapsed_time = time.time() - start_time
    if elapsed_time >= 180:
        raise HTTPException(status_code=504, detail="Process execution time exceeded 180 seconds limit.")
    
    print(f"Processing request for email: {payload.email}, URL: {payload.url} (Attempt Time Left: {180 - elapsed_time:.2f}s)")
    
    async with async_playwright() as p:
        scrape_timeout = max(10, 180 - (time.time() - start_time) - 10)
        browser = await p.chromium.launch(timeout=scrape_timeout * 1000)
        page = await browser.new_page()
        try:
            await page.goto(payload.url, wait_until="networkidle", timeout=scrape_timeout * 1000)
            scraped_content = await page.content()
        except Exception as e:
            await browser.close()
            raise HTTPException(status_code=500, detail=f"Failed to scrape URL: {e}")
        await browser.close()

    system_prompt = f"""You are an intelligent agent. Analyze the provided HTML content and try to decipher 
    required task, answer and submission URL.
    
    Context:
    Email: {payload.email}
    Secret: {payload.secret}
    URL: {payload.url}
    
    CRITICAL INSTRUCTION:
    If the content says something like "Scrape /path for secret" or "Go to <link>" or "The data is at...", YOU DO NOT HAVE THE ANSWER YET.
    If the content contains raw data (e.g., a CSV file, an audio file URL, encrypted text, or a list of numbers requiring computation) that must be processed to find the answer, YOU DO NOT HAVE THE ANSWER YET.
    In these cases, you MUST set "answerable": false and determine the next step: either put the URL/path in the "Required" list (for Scraping) or define the operation (for Operation).
    If any downloadable extension file links exist then it is not answerable yet without fetching that data.
    Only set "answerable": true if you can see the actual secret code/answer explicitly in the provided content AND you know the submission_url.
    If the content sayd POST the answer to a URL, or has a form with action URL, or a button link to submit the answer, and no extra links or pages given hinting an answer, then you MUST set "answerable": true.
    
    Only set "answerable": true if you can see the actual secret code/answer explicitly in the provided content AND you know the submission_url.
    
    the submission URL is typically found in a form action or a button link.
    the submission URL is different from the provided URL.
    a final submision URL will always start with http or https and will be a complete URL.
    type will be defined by the type of data you need to fetch next, either "Scraping" or "Operation".
    if a submission url is relative, convert it to absolute using the provided URL as base.
    
    If type is "Operation", provide the "operation" list with a text description of what needs to be calculated/coded (e.g. "Sum the values in column A of the CSV at URL...").
    
    Return this JSON format strictly:
    {{
      "email": "{payload.email}",
      "secret": "{payload.secret}",
      "url": "{payload.url}",
      "submission_url": <extracted_submission_url>,
      "answer": <extracted_answer>,
      "answerable": true
    }}
    
    If answerable is False (you need more data/operations):
    Return this JSON format strictly:
    {{
      "answerable": false,
      "type": "Scraping" or "Operation",
      "Required": ["<url_or_path_of_missing_data>"],
      "operation": ["<description_of_task>"]
    }}
    
    Output strictly valid JSON only."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": scraped_content[:100000]}
    ]

    async with httpx.AsyncClient() as client:
        try:
            async def get_ai_response(msgs):
                current_elapsed = time.time() - start_time
                remaining_timeout = max(10.0, 180 - current_elapsed)
                
                response = await client.post(
                    f"{AIPIPE_BASE_URL.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {AIPIPE_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-5-nano",
                        "messages": msgs,
                        "response_format": {"type": "json_object"}
                    },
                    timeout=min(60.0, remaining_timeout)
                )
                response.raise_for_status()
                ai_data = response.json()
                return json.loads(ai_data["choices"][0]["message"]["content"])

            result = await get_ai_response(messages)
            print("Initial AI Result:", result)

            if not result.get("answerable"):
                aux_content = ""
                required_items = result.get("Required", [])
                
                # Check for audio files specifically in Required list to transcribe first
                # Added .opus to the list of audio extensions
                audio_item = next((item for item in required_items if any(item.lower().split('?')[0].endswith(ext) for ext in ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.webm', '.opus'])), None)
                
                if audio_item:
                    full_next_url = audio_item if audio_item.startswith("http") else urljoin(payload.url, audio_item)
                    print(f"Detected audio file. Transcribing: {full_next_url}")
                    
                    current_elapsed = time.time() - start_time
                    aux_scrape_timeout = max(10, 180 - current_elapsed - 10)
                    
                    try:
                        audio_res = await client.get(full_next_url, timeout=aux_scrape_timeout)
                        audio_res.raise_for_status()
                        files = {
                            'file': ('audio' + os.path.splitext(full_next_url)[1], audio_res.content, mimetypes.guess_type(full_next_url)[0] or 'application/octet-stream')
                        }
                        transcribe_res = await client.post(
                            f"{AIPIPE_BASE_URL.rstrip('/')}/audio/transcriptions",
                            headers={"Authorization": f"Bearer {AIPIPE_API_KEY}"},
                            data={"model": "gpt-5-nano"},
                            files=files,
                            timeout=aux_scrape_timeout
                        )
                        transcribe_res.raise_for_status()
                        aux_content = f"Audio Transcription of {full_next_url}:\n{transcribe_res.json().get('text', '')}"
                    except Exception as e:
                        aux_content = f"Failed to process audio file: {e}"

                elif result.get("type") == "Scraping" and required_items:
                    next_path = required_items[0]
                    full_next_url = next_path if next_path.startswith("http") else urljoin(payload.url, next_path)
                    print(f"Fetching required auxiliary data from: {full_next_url}")
                    
                    current_elapsed = time.time() - start_time
                    aux_scrape_timeout = max(10, 180 - current_elapsed - 10)
                    
                    async with async_playwright() as p:
                        browser = await p.chromium.launch(timeout=aux_scrape_timeout * 1000)
                        page = await browser.new_page()
                        try:
                            await page.goto(full_next_url, wait_until="networkidle", timeout=aux_scrape_timeout * 1000)
                            aux_content = await page.content()
                        except Exception:
                            aux_content = "Failed to scrape the required URL."
                        await browser.close()
                
                elif result.get("type") == "Operation" and result.get("operation"):
                    task_desc = result["operation"][0]
                    
                    # Ensure URLs are absolute for the coding agent
                    abs_urls = []
                    for u in required_items:
                        abs_urls.append(u if u.startswith("http") else urljoin(payload.url, u))
                    
                    print(f"Executing Operation: {task_desc}")
                    aux_content = await perform_coding_task(task_desc, str(abs_urls), start_time)
                    print(f"Operation Result: {aux_content}")

                if aux_content:
                    messages.append({"role": "assistant", "content": json.dumps(result)})
                    messages.append({"role": "user", "content": f"I have performed the required step. Here is the output/content:\n\n{str(aux_content)[:100000]}"})
                    
                    print("Re-querying AI with new context...")
                    result = await get_ai_response(messages)
                    print("Updated AI Result:", result)

            if result.get("answerable") and result.get("submission_url"):
                try:
                    current_elapsed = time.time() - start_time
                    submission_timeout = max(10.0, 180 - current_elapsed)

                    submission_response = await client.post(result["submission_url"], json=result, timeout=submission_timeout)
                    submission_response.raise_for_status()
                    received = json.loads(submission_response.text)
                    print("Submission Response:", received)
                    
                    if received.get("correct"):
                        result["submission_status"] = "Answer submitted successfully."
                        if "url" in received:
                            new_payload_data = {
                                "email": payload.email,
                                "secret": payload.secret,
                                "url": received["url"]
                            }
                            return await process_data(Payload(**new_payload_data))
                    else:
                        result["submission_status"] = "Answer submission failed."
                        raise SubmissionRetryableError("Submission response indicated failure (correct: false).")
                
                except SubmissionRetryableError:
                    raise
                except httpx.HTTPStatusError as e:
                    print(f"Submission HTTP error: {e}")
                    raise SubmissionRetryableError(f"Submission failed with HTTP error: {e}")
                except Exception as e:
                    print(f"Submission error: {e}")
                    raise SubmissionRetryableError(f"Submission failed with exception: {e}")

            return result
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"AI processing or intermediate step failed: {str(e)}")


@app.post("/api")
async def process_data(payload: Payload):
    if payload.email != EXPECTED_EMAIL or payload.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret or email provided")
    
    start_time = time.time()
    max_execution_time = 180
    retry_delay = 1

    while time.time() - start_time < max_execution_time:
        try:
            return await _process_data_attempt(payload, start_time)
        
        except SubmissionRetryableError as e:
            time_spent = time.time() - start_time
            time_left = max_execution_time - time_spent
            
            if time_left <= 0:
                print("180 seconds limit reached. Stopping retries.")
                break
            
            delay = min(retry_delay, time_left)
            print(f"Retryable submission error encountered ({e}). Retrying in {delay:.2f} seconds... ({time_left:.2f}s remaining)")
            await asyncio.sleep(delay)

        except HTTPException:
            raise
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected fatal error outside submission block: {str(e)}")

    raise HTTPException(status_code=504, detail="Process execution failed: 180 seconds limit reached after retries.")
    
if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    uvicorn.run(app, host="0.0.0.0", port=8000)