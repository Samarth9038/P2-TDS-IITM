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
import shutil
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse
from urllib.parse import urljoin, urlparse




load_dotenv()




EXPECTED_EMAIL = os.getenv("EXPECTED_EMAIL")
MY_SECRET = os.getenv("MY_SECRET")
AIPIPE_API_KEY = os.getenv("AIPIPE_API_KEY")
AIPIPE_BASE_URL = os.getenv("AIPIPE_BASE_URL", "")




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




def check_ffmpeg():
    """Check if ffmpeg is available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False




def convert_to_wav(input_path, output_path):
    """Convert audio to WAV format using ffmpeg"""
    try:
        cmd = [
            'ffmpeg', '-i', input_path,
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',
            output_path
        ]
        result = subprocess.run(cmd, 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              timeout=30)
        return result.returncode == 0
    except Exception as e:
        print(f"[AUDIO CONVERT ERROR] {e}")
        return False




def make_absolute_url(base_url, relative_path):
    if relative_path.startswith("http://") or relative_path.startswith("https://"):
        return relative_path



    parsed_base = urlparse(base_url)
    base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"



    if relative_path.startswith("/"):
        return f"{base_domain}{relative_path}"
    else:
        return f"{base_domain}/{relative_path}"




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




def transcribe_audio_sync(audio_path, max_retries=3, retry_delay=1.5):
    """Transcribe audio with ffmpeg check and WAV conversion"""


    if not check_ffmpeg():
        print(f"[TRANSCRIPTION ERROR] ffmpeg not found in PATH")
        print(f"[TRANSCRIPTION] Install: choco install ffmpeg (Windows) or apt install ffmpeg (Linux)")
        return None


    file_ext = os.path.splitext(audio_path)[1].lower()
    if file_ext != '.wav':
        print(f"[TRANSCRIPTION] Converting {file_ext} to WAV format...")
        wav_path = audio_path.rsplit('.', 1)[0] + '_converted.wav'


        if convert_to_wav(audio_path, wav_path):
            print(f"[TRANSCRIPTION] Conversion successful: {wav_path}")
            transcription_path = wav_path
            cleanup_wav = True
        else:
            print(f"[TRANSCRIPTION] Conversion failed, trying original file...")
            transcription_path = audio_path
            cleanup_wav = False
    else:
        transcription_path = audio_path
        cleanup_wav = False


    for attempt in range(max_retries):
        try:
            if not os.path.exists(transcription_path):
                print(f"[TRANSCRIPTION] Attempt {attempt + 1}/{max_retries}: File not found")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return None


            file_size = os.path.getsize(transcription_path)
            print(f"[TRANSCRIPTION] Attempt {attempt + 1}/{max_retries}: File size {file_size} bytes")


            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(transcription_path)
            transcription_text = result["text"].strip()


            print(f"[TRANSCRIPTION SUCCESS] Text: {transcription_text}")


            if cleanup_wav and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except:
                    pass


            return transcription_text


        except ImportError as e:
            print(f"[TRANSCRIPTION ERROR] Whisper not installed: {e}")
            if cleanup_wav and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except:
                    pass
            return None
        except Exception as e:
            print(f"[TRANSCRIPTION] Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"[TRANSCRIPTION] Waiting {retry_delay}s before retry...")
                time.sleep(retry_delay)
            else:
                print(f"[TRANSCRIPTION ERROR] All {max_retries} attempts failed")
                if cleanup_wav and os.path.exists(wav_path):
                    try:
                        os.remove(wav_path)
                    except:
                        pass
                return None


    return None




async def perform_coding_task(task_description, audio_instructions, local_files, page_content_snippet, start_time, attempt_history=[]):
    if local_files:
        file_info = "\n".join([f"Local file path: {path}" for path in local_files])
    else:
        file_info = "No local files provided"



    context_sections = []



    if audio_instructions:
        context_sections.append(f"""
========================================
AUDIO TRANSCRIPTION (CRITICAL):
========================================
{audio_instructions}



THIS IS ESSENTIAL: Use the audio instructions above to complete the task correctly.
========================================
""")

    if attempt_history:
        history_text = "\n".join([
            f"Attempt {i+1}: Answer={attempt['answer']}, Result={attempt['result']}, Reason={attempt.get('reason', 'N/A')}"
            for i, attempt in enumerate(attempt_history)
        ])
        context_sections.append(f"""
========================================
PREVIOUS ATTEMPTS (LEARN FROM MISTAKES):
========================================
{history_text}



CRITICAL: These attempts FAILED. Analyze what went wrong and correct the logic.
Common mistakes:
- Wrong comparison operator (> vs >=, < vs <=)
- Incorrect column index
- Wrong aggregation function
- Off-by-one errors
- Missing data filtering
========================================
""")



    if page_content_snippet:
        context_sections.append(f"""
========================================
PAGE CONTENT CONTEXT:
========================================
{page_content_snippet[:2000]}
========================================
""")



    context_block = "\n".join(context_sections) if context_sections else "No additional context provided."



    messages = [
        {
            "role": "system", 
            "content": f"""You are a Python coding expert specialized in data processing, analysis, and automation.



{context_block}



Task Description: {task_description}



Available Data Files:
{file_info}



REQUIREMENTS:
1. Analyze the task description and any audio/page context to understand what needs to be done
2. If there are previous attempts, LEARN FROM THEM - identify what went wrong
3. Use the local file paths provided - do NOT download files again
4. Write complete, working Python code that accomplishes the task
5. Handle common file formats: CSV, JSON, TXT, PDF, images, audio
6. Use appropriate libraries: pandas, numpy, requests, PIL, opencv, etc.
7. For data analysis: filter, sort, aggregate, reshape, apply statistical methods as needed
8. For visualization: generate charts/plots and save to files if needed
9. Print the final result/answer to stdout (ONLY the numeric answer, no extra text)
10. Handle errors gracefully



OUTPUT FORMAT:
- Only output valid Python code inside ```python``` blocks
- No explanations, just code
            """
        }
    ]




    async with httpx.AsyncClient() as client:
        max_attempts = 3
        attempt = 0



        while attempt < max_attempts and time.time() - start_time < 165:
            attempt += 1
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
                    print(f"[CODING AGENT] Attempt {attempt}: Executing generated code")
                    success, output = await execute_generated_code(code, timeout=30)



                    if success:
                        print(f"[CODING AGENT SUCCESS] Result: {output}")
                        return output
                    else:
                        print(f"[CODING AGENT ERROR] {output}")
                        messages.append({"role": "assistant", "content": ai_content})
                        messages.append({"role": "user", "content": f"Execution failed: {output}\nFix the code and try again."})
                else:
                    messages.append({"role": "assistant", "content": ai_content})
                    messages.append({"role": "user", "content": "No python code block found. Provide code in ```python``` format."})



            except Exception as e:
                print(f"[CODING AGENT] API error: {e}")
                break



    return None




async def download_file_to_local(url, client, timeout=30):
    try:
        response = await client.get(url, timeout=timeout)
        response.raise_for_status()



        file_ext = os.path.splitext(url.split('?')[0])[1] or '.csv'



        with tempfile.NamedTemporaryFile(mode='wb', suffix=file_ext, delete=False) as tmp:
            tmp.write(response.content)
            return tmp.name
    except Exception as e:
        print(f"Failed to download file {url}: {e}")
        return None




async def _process_data_attempt(payload: Payload, start_time: float):
    elapsed_time = time.time() - start_time
    if elapsed_time >= 180:
        raise HTTPException(status_code=504, detail="Process execution time exceeded 180 seconds limit.")



    print(f"\n{'='*80}")
    print(f"Processing: {payload.url}")
    print(f"Time remaining: {180 - elapsed_time:.2f}s")
    print(f"{'='*80}\n")



    original_payload_url = payload.url
    original_email = payload.email
    original_secret = payload.secret



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




    system_prompt = f"""You are an intelligent agent that analyzes web pages to determine what actions need to be taken.



Context:
Email: {payload.email}
Secret: {payload.secret}
Current URL: {payload.url}



ANALYSIS FRAMEWORK:



1. DETERMINE IF ANSWERABLE:
   - answerable=TRUE: The final answer is explicitly visible on this page AND you have a submission URL
   - answerable=FALSE: When you dont have enough data to provide the final answer or submission URL



2. WHEN answerable=FALSE, IDENTIFY THE TYPE:



   Type="Scraping":
   - Another page/URL needs to be visited
   - Data is located at a different endpoint
   - API call required (note any headers/auth mentioned)
   - JavaScript rendering needed for hidden content



   Type="Operation":
   - Files need to be downloaded (CSV, JSON, PDF, images, audio, etc.)
   - Data needs processing (transcription, OCR, parsing)
   - Computation required (filtering, aggregating, statistical analysis)
   - Transformation needed (cleansing, reshaping, format conversion)
   - Visualization requested (charts, graphs, reports)
   - Machine learning/analysis tasks



3. POPULATE REQUIRED FIELDS:
   - Required: List of URLs/paths to fetch (for files, APIs, or other pages)
   - operation: Detailed description of what needs to be computed/processed



4. EXTRACT SUBMISSION INFO:
   - submission_url: Where to POST the final answer (form action, API endpoint, explicit URL)
   - Look for: <form action="...">, POST endpoints, submission instructions



OUTPUT FORMATS:



If answerable=TRUE (final answer visible):
{{
  "email": "{payload.email}",
  "secret": "{payload.secret}",
  "url": "{payload.url}",
  "submission_url": "<absolute_http_url>",
  "answer": "<the_actual_answer>",
  "answerable": true
}}



If answerable=FALSE (more work needed):
{{
  "answerable": false,
  "type": "Scraping" or "Operation",
  "Required": ["<list_of_urls_or_paths>"],
  "operation": ["<detailed_task_description>"]
}}



IMPORTANT:
- Be precise about file URLs, API endpoints, and paths
- Include any authentication headers or API keys mentioned on the page
- For operations, describe the task clearly (e.g., "Filter CSV where column A > 1000", "Transcribe audio and extract keywords")
- Don't guess - if you need more data, mark answerable=FALSE



Output valid JSON only."""




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
            print(f"[AI ANALYSIS] answerable={result.get('answerable')}, type={result.get('type')}, answer={result.get('answer')}")




            local_files = []
            audio_instructions = None
            audio_temp_file = None
            downloaded_urls = set()
            max_iterations = 10
            iteration_count = 0
            submission_attempts = 0
            max_submission_attempts = 5
            page_context_snippet = scraped_content[:5000]
            last_operation_result = None
            attempt_history = []

            while time.time() - start_time < 165 and iteration_count < max_iterations:
                iteration_count += 1


                current_elapsed = time.time() - start_time
                if current_elapsed >= 165:
                    print(f"\n[TIMEOUT WARNING] 165 seconds reached - forcing submission")


                    if not result.get("answerable"):
                        if not result.get("submission_url"):
                            from bs4 import BeautifulSoup
                            soup = BeautifulSoup(scraped_content, 'html.parser')
                            form = soup.find('form')
                            if form and form.get('action'):
                                submission_url = make_absolute_url(payload.url, form.get('action'))
                            else:
                                submission_url = "https://tds-llm-analysis.s-anand.net/submit"
                            result["submission_url"] = submission_url


                        result["answerable"] = True
                        result["email"] = original_email
                        result["secret"] = original_secret
                        result["url"] = original_payload_url


                        if last_operation_result:
                            result["answer"] = last_operation_result
                        elif not result.get("answer"):
                            result["answer"] = "Unable to compute within time limit"


                        print(f"[FORCED SUBMISSION] Submitting best available answer: {result.get('answer')}")
                    break

                if not result.get("answerable"):
                    print(f"\n[ITERATION {iteration_count}/{max_iterations}] Time left: {165 - current_elapsed:.1f}s")

                    aux_content = ""
                    required_items = result.get("Required", [])
                    files_added = False

                    audio_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.webm', '.opus']
                    data_extensions = ['.csv', '.txt', '.json', '.xlsx', '.xls', '.pdf', '.jpg', '.jpeg', '.png', '.gif']

                    audio_items = [item for item in required_items if any(item.lower().split('?')[0].endswith(ext) for ext in audio_extensions)]
                    data_items = [item for item in required_items if any(item.lower().split('?')[0].endswith(ext) for ext in data_extensions)]

                    if audio_items and not audio_instructions:
                        for audio_item in audio_items:
                            if time.time() - start_time >= 150:
                                print(f"[TIMEOUT] Skipping audio download - time running out")
                                break

                            full_audio_url = make_absolute_url(payload.url, audio_item)

                            if full_audio_url in downloaded_urls:
                                print(f"[AUDIO] Already processed: {full_audio_url}")
                                continue

                            downloaded_urls.add(full_audio_url)
                            files_added = True
                            print(f"[AUDIO] Downloading from: {full_audio_url}")

                            current_elapsed = time.time() - start_time
                            download_timeout = max(5, min(15, 165 - current_elapsed))

                            try:
                                audio_res = await client.get(full_audio_url, timeout=download_timeout)
                                audio_res.raise_for_status()

                                audio_ext = os.path.splitext(full_audio_url)[1] or '.opus'

                                with tempfile.NamedTemporaryFile(mode='wb', suffix=audio_ext, delete=False) as audio_tmp:
                                    audio_tmp.write(audio_res.content)
                                    audio_tmp.flush()
                                    os.fsync(audio_tmp.fileno())
                                    audio_temp_file = audio_tmp.name

                                print(f"[AUDIO] Saved to: {audio_temp_file}")

                                if os.path.exists(audio_temp_file):
                                    file_size = os.path.getsize(audio_temp_file)
                                    print(f"[AUDIO] File size: {file_size} bytes")

                                    transcription = transcribe_audio_sync(audio_temp_file)

                                    if transcription:
                                        audio_instructions = transcription
                                        print(f"[AUDIO] Transcription successful")
                                    else:
                                        print(f"[AUDIO WARNING] Transcription failed - continuing without audio")
                                else:
                                    print(f"[AUDIO ERROR] File not found after write")

                            except Exception as e:
                                print(f"[AUDIO ERROR] {e}")

                    if data_items:
                        for data_item in data_items:
                            if time.time() - start_time >= 160:
                                print(f"[TIMEOUT] Skipping data download - time running out")
                                break

                            full_data_url = make_absolute_url(payload.url, data_item)

                            if full_data_url in downloaded_urls:
                                print(f"[DATA] Already downloaded: {full_data_url}")
                                continue

                            downloaded_urls.add(full_data_url)
                            files_added = True
                            print(f"[DATA] Downloading from: {full_data_url}")

                            current_elapsed = time.time() - start_time
                            download_timeout = max(5, min(15, 165 - current_elapsed))

                            local_path = await download_file_to_local(full_data_url, client, download_timeout)
                            if local_path:
                                local_files.append(local_path)
                                print(f"[DATA] Saved to: {local_path}")

                    if result.get("operation") and (local_files or audio_instructions):
                        if time.time() - start_time >= 160:
                            print(f"[TIMEOUT] Skipping operation - time running out")
                        else:
                            task_desc = result["operation"][0]

                            print(f"[OPERATION] Starting code generation")
                            print(f"[OPERATION] Audio: {'YES' if audio_instructions else 'NO'}")
                            print(f"[OPERATION] Files: {len(local_files)}")
                            if attempt_history:
                                print(f"[OPERATION] Previous attempts: {len(attempt_history)}")


                            operation_result = await perform_coding_task(
                                task_desc, 
                                audio_instructions, 
                                local_files, 
                                page_context_snippet,
                                start_time,
                                attempt_history
                            )

                            if operation_result:
                                aux_content = operation_result
                                last_operation_result = operation_result
                            else:
                                print(f"[OPERATION] Code generation failed")

                    elif result.get("type") == "Scraping" and required_items and not audio_items and not data_items:
                        if time.time() - start_time >= 160:
                            print(f"[TIMEOUT] Skipping scraping - time running out")
                        else:
                            next_path = required_items[0]
                            full_next_url = make_absolute_url(payload.url, next_path)
                            print(f"[SCRAPING] Fetching from: {full_next_url}")

                            current_elapsed = time.time() - start_time
                            scrape_timeout = max(5, min(15, 165 - current_elapsed))

                            async with async_playwright() as p:
                                browser = await p.chromium.launch(timeout=scrape_timeout * 1000)
                                page = await browser.new_page()
                                try:
                                    await page.goto(full_next_url, wait_until="networkidle", timeout=scrape_timeout * 1000)
                                    aux_content = await page.content()
                                except Exception:
                                    aux_content = "Failed to scrape"
                                await browser.close()

                    if time.time() - start_time < 160:
                        messages.append({"role": "assistant", "content": json.dumps(result)})
                        messages.append({"role": "user", "content": f"Step completed. Output:\n{str(aux_content)[:100000]}"})

                        print(f"[AI] Re-querying with results...")
                        result = await get_ai_response(messages)
                        print(f"[AI] Updated: answerable={result.get('answerable')}")
                    else:
                        print(f"[TIMEOUT] Skipping AI re-query - forcing submission")
                        break

                if result.get("answerable") and result.get("submission_url"):
                    submission_attempts += 1
                    result["email"] = original_email
                    result["secret"] = original_secret
                    result["url"] = original_payload_url

                    print(f"\n[SUBMISSION {submission_attempts}/{max_submission_attempts}] Answer: {result.get('answer')}")

                    try:
                        current_elapsed = time.time() - start_time
                        submission_timeout = max(5.0, 180 - current_elapsed)

                        submission_response = await client.post(result["submission_url"], json=result, timeout=submission_timeout)
                        submission_response.raise_for_status()
                        received = json.loads(submission_response.text)

                        if received.get("correct"):
                            print(f"[SUBMISSION] ✓ SUCCESS")

                            for local_file in local_files:
                                try:
                                    if os.path.exists(local_file):
                                        os.remove(local_file)
                                except Exception:
                                    pass

                            if audio_temp_file and os.path.exists(audio_temp_file):
                                try:
                                    os.remove(audio_temp_file)
                                except Exception:
                                    pass

                            if "url" in received and received["url"] is not None:
                                print(f"[NEXT] {received['url']}")
                                new_payload_data = {
                                    "email": payload.email,
                                    "secret": payload.secret,
                                    "url": received["url"]
                                }
                                return await process_data(Payload(**new_payload_data))
                            else:
                                print(f"[COMPLETE] All challenges finished!")
                                result["final_completion"] = True
                                return result
                        else:
                            reason = received.get('reason', 'Unknown')
                            print(f"[SUBMISSION] ✗ FAILED: {reason}")

                            attempt_history.append({
                                "answer": result.get('answer'),
                                "result": "FAILED",
                                "reason": reason
                            })
                            print(f"[HISTORY] Recorded failed attempt: {result.get('answer')} -> {reason}")

                            if submission_attempts >= max_submission_attempts or time.time() - start_time >= 165:
                                print(f"[SUBMISSION] Max attempts or time limit reached")
                                result["submission_status"] = f"Failed: {reason}"
                                return result

                            messages.append({"role": "assistant", "content": json.dumps(result)})
                            messages.append({"role": "user", "content": f"""Submission FAILED: {reason}. The answer '{result.get('answer')}' was incorrect.

IMPORTANT: The computation has already been done. The latest result is: {last_operation_result if last_operation_result else 'not available'}

You must now:
1. Analyze why the previous answer was wrong
2. Re-execute the operation with corrected logic
3. Once you have the NEW computed result, immediately return with answerable=TRUE and the new answer

Return format:
{{
  "email": "{payload.email}",
  "secret": "{payload.secret}",
  "url": "{payload.url}",
  "submission_url": "<same_submission_url>",
  "answer": "<new_computed_answer>",
  "answerable": true
}}

DO NOT return answerable=false - you already have all the data you need."""})

                            print(f"[RETRY] Re-analyzing with explicit instruction to mark answerable=true...")
                            result = await get_ai_response(messages)
                            print(f"[RETRY] New analysis: answerable={result.get('answerable')}, answer={result.get('answer')}")

                            if not result.get("answerable"):
                                print(f"[RETRY OVERRIDE] AI marked answerable=False, forcing to True with last result")
                                result["answerable"] = True
                                result["email"] = original_email
                                result["secret"] = original_secret
                                result["url"] = original_payload_url
                                if not result.get("submission_url"):
                                    from bs4 import BeautifulSoup
                                    soup = BeautifulSoup(scraped_content, 'html.parser')
                                    form = soup.find('form')
                                    if form and form.get('action'):
                                        result["submission_url"] = make_absolute_url(payload.url, form.get('action'))
                                    else:
                                        result["submission_url"] = "https://tds-llm-analysis.s-anand.net/submit"

                                if last_operation_result and not result.get("answer"):
                                    result["answer"] = last_operation_result
                                    print(f"[RETRY OVERRIDE] Using last_operation_result: {last_operation_result}")

                            continue

                    except Exception as e:
                        print(f"[SUBMISSION ERROR] {e}")
                        result["submission_status"] = f"Error: {str(e)}"
                        return result

            for local_file in local_files:
                try:
                    if os.path.exists(local_file):
                        os.remove(local_file)
                except Exception:
                    pass

            if audio_temp_file and os.path.exists(audio_temp_file):
                try:
                    os.remove(audio_temp_file)
                    print(f"[CLEANUP] Audio temp file deleted")
                except Exception:
                    pass

            return result
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/api")
async def process_data(payload: Payload):
    if payload.email != EXPECTED_EMAIL or payload.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid credentials")

    start_time = time.time()

    try:
        return await _process_data_attempt(payload, start_time)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    uvicorn.run(app, host="0.0.0.0", port=8000)
