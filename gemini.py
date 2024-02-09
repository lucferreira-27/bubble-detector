import asyncio
import aiofiles
from aiohttp import ClientSession
import json
import os
from pathlib import Path
import google.generativeai as genai
from ratelimit import limits
import time
from google.api_core.exceptions import ResourceExhausted
import openai
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

MAX_REQUESTS = 60

generation_config = {
 "temperature": 0,
 "top_p": 1,
 "top_k": 32,
 "max_output_tokens": 4096,
}

safety_settings = [
 {
   "category": "HARM_CATEGORY_HARASSMENT",
   "threshold": "BLOCK_NONE"
 },
 {
   "category": "HARM_CATEGORY_HATE_SPEECH",
   "threshold": "BLOCK_NONE"
 },
 {
   "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
   "threshold": "BLOCK_NONE"
 },
 {
   "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
   "threshold": "BLOCK_NONE"
 },
]

model = genai.GenerativeModel(model_name="gemini-pro-vision",
                            generation_config=generation_config,
                            safety_settings=safety_settings)



async def validate_and_fix_json(json_data):
    print("[GEMINI VISION] -> Validating and fixing JSON data...")
    """
    Validates and attempts to fix JSON data using OpenAI's ChatGPT-3.5-turbo model, including examples of common errors, a double-check mechanism, and adaptive feedback based on specific past mistakes.
    
    :param json_data: The JSON data as a string that needs validation and fixing.
    :return: A tuple containing a boolean indicating success and the validated/fixed JSON data, along with specific feedback based on encountered errors.
    """
    def is_valid_json_structure(data):
        """
        Checks if the given JSON data matches the expected structure and returns validation status and error message if any.
        """
        try:
            parsed_data = json.loads(data)
            # Add more specific structure validation if necessary
            if "text_empty" in parsed_data and isinstance(parsed_data["text_empty"], bool):
                return True, ""
        except json.JSONDecodeError as e:
            return False, f"JSONDecodeError: {str(e)}"
        return False, "The JSON structure does not match the expected format."

    async def correct_json_with_ai(data, feedback=""):
        """
        Uses OpenAI's model to correct JSON data, incorporating feedback from previous attempts to highlight specific issues.
        """
        sample = '{"text_empty": Boolean, "text": []}'
        prompt_parts = [
            f"{feedback}Given the following JSON data, correct any errors to ensure it is valid JSON format. The output should be always in format {sample}:",
            f"Input: {data}\nOutput:"
        ]
        prompt = "\n".join(prompt_parts)

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()

    feedback = ""
    attempts = 0
    max_attempts = 3  # Set a maximum number of attempts to avoid infinite loops
    corrected_json = None
    while attempts < max_attempts:
        attempts += 1
        is_valid, error_message = is_valid_json_structure(json_data)
        if is_valid:
            return True, json_data  # JSON is valid

        # If not valid, prepare feedback for the AI including the specific error encountered
        feedback = f"The previous attempt was not successful due to: {error_message} Please correct the JSON data accordingly. Only return the fixed json "

        corrected_json = await correct_json_with_ai(json_data, feedback=feedback)
        json_data = corrected_json  # Use the corrected JSON for the next validation attempt

    # If the loop exits without returning, it means all attempts failed
    if not corrected_json:
        raise Exception("Failed to fix JSON data after multiple attempts.")
    return corrected_json


# Example usage remains the same as previously provided 


examples_image_directory = Path("./gemini_examples")
examples_image_files = [f for f in examples_image_directory.iterdir() if f.suffix == '.jpg']

examples_image_parts = [{
   "mime_type": "image/jpeg",
   "data": img.read_bytes()
} for img in examples_image_files if img.exists()]

async def read_image_data_async(img_file: str) -> bytes:
  async with aiofiles.open(img_file, 'rb') as img:
      return await img.read()

def check_image_exists(img_file):
  if not Path(img_file).exists():
      print(f"Could not find image: {img_file}")

def build_prompt_parts(examples_image_parts, image_data, samples):
   prompt_parts = [
       "input: Analyze the content of the attached manga page, following the manga-specific reading order: right-to-left, then top-to-bottom. Produce a continuous output without any line breaks (/n), adhering strictly to this format to ensure the response respects the manga's original layout and narrative flow.",
   ]
   for i, example in enumerate(examples_image_parts):
       output = json.dumps(samples[i], indent=0).replace("\n","")
       prompt_parts.extend([
           example,
           f"output: {output}"
       ])
   prompt_parts.extend([
       "input: Analyze the content of the attached manga page, following the manga-specific reading order: right-to-left, then top-to-bottom. Produce a continuous output without any line breaks (/n), adhering strictly to this format to ensure the response respects the manga's original layout and narrative flow.",
       {
           "mime_type": "image/jpeg",
           "data": image_data
       },
       "output: "
   ])
   return prompt_parts

async def fetch(img_file):
    check_image_exists(img_file)
    image_data = await read_image_data_async(img_file)
    
    samples = [
        {"text_empty": False, "text": ["I just assumed that the person I saw with Robin was you in human form..."]},
        {"text_empty": True},
        {"text_empty": False, "text": ["Right... It's the support that extends from stem to stern.", "That wooden beam is the most crucial part of a ship."]},
        {"text_empty": True},
        {"text_empty": False, "text": ["HA HA HA HA HA HA"]},
        {"text_empty": False, "text": ["SHAAAOOO"]}
    ]
    prompt_parts = build_prompt_parts(examples_image_parts, image_data, samples)
    loop = asyncio.get_running_loop()
    
    try:
        response = await loop.run_in_executor(None, model.generate_content, prompt_parts)
        return response.text
    except Exception as e:
        print(f"An internal server error occurred: {e}")
        # Here you can decide to retry, log the error, or handle it as needed
        return json.dumps({"gemini_block": True, "text_empty": False})

def get_image_files(image_files):
    if isinstance(image_files, str) and Path(image_files).is_dir():
        supported_formats = ['.jpg', '.jpeg', '.png']  # Add additional supported formats here
        return [str(f) for f in Path(image_files).iterdir() if f.suffix in supported_formats]
    elif isinstance(image_files, list):
        return image_files
    else:
        raise TypeError("image_files must be a directory or a list")


progress_lock = asyncio.Lock()
progress_count = 0


async def process_image(image_file, total):
    global progress_count
    response = await fetch(image_file)
    filename = os.path.basename(image_file)
        
    try:
        parsed_response = json.loads(response)
    except json.decoder.JSONDecodeError as e:
        print(f"Gemini bad answer: {response}")
        print(f"Failed to decode JSON for {filename}: {e}")
        _,response_fixed = await validate_and_fix_json(response)
        parsed_response = json.loads(response_fixed)
        if parsed_response:
            return filename, parsed_response
        raise e
    
    async with progress_lock:
        progress_count += 1
        progress = f"[GEMINI VISION] -> ({progress_count}/{total}) tasks"
    
    if parsed_response.get('text_empty', True):
        print(f"{progress} | Response: No text detected for {filename}")
    else:
        print(f"{progress} | Response: Text successfully detected for {filename}")
    return filename, parsed_response



async def process_images(image_files):
 print("[GEMINI VISION] -> Starting image processing")
 image_files = get_image_files(image_files)
 if len(image_files) == 0:
     return {"text_empty": True}

 # Split image_files into chunks of 60
 chunks = [image_files[i:i + 60] for i in range(0, len(image_files), 60)]
 print(f"[GEMINI VISION] -> Created {len(chunks)} chunks from a total of {len(image_files)} images")

 total_tasks = 0
 total_responses = 0
 total_time = 0

 for i, chunk in enumerate(chunks):
   print(f"[GEMINI VISION] -> Processing chunk {i+1} of {len(chunks)}")
   # Create tasks for processing images with progress tracking
   if total_tasks == 0:
       total_tasks = len(chunk)
   else:
       total_tasks += len(chunk)
       
   tasks = [process_image(img_file, total_tasks) for img_file in chunk]
   print(f"[GEMINI VISION] -> Fetching image data for {len(chunk)} images")
   global progress_count

   retries = 0
   max_retries = 5
   backoff_factor = 1.5 # Exponential backoff factor
   wait_time = 15 # Initial wait time in seconds

   while retries < max_retries:
       try:
           # Measure the time taken to process images
           start_time = time.time()
           responses = await asyncio.gather(*tasks)
           elapsed_time = time.time() - start_time
           break # If the call was successful, break out of the loop
       except ResourceExhausted:
           print(f"[GEMINI VISION] -> Request limit exceeded. Retrying in {wait_time} seconds...")
           time.sleep(wait_time)
           async with progress_lock:
              progress_count = abs(progress_count - len(tasks))
           retries += 1
           wait_time = 60
           tasks = [process_image(img_file, total_tasks) for img_file in chunk] # Recreate tasks

   if retries == max_retries:
       print("[GEMINI VISION] -> Resource quota exceeded. Please try again later.")
       continue # Skip to the next iteration of the loop

   # Build a dictionary of responses with text detected
   progress_count = 0
   responses_dict = {img_file: response for img_file, response in responses if not response['text_empty']}
   print("[GEMINI VISION] -> Parsing responses")

   total_responses += len(responses_dict)
   total_time += elapsed_time

   # Calculate the remaining time to reach 80 seconds
   remaining_time = 30
   if remaining_time > 0 and chunks.index(chunk) != len(chunks) - 1:
       print(f"[GEMINI VISION] -> Waiting for {remaining_time:.2f} seconds before proceeding to next chunk.")
       time.sleep(remaining_time)

 # Print summary of the processing
 print(f"[GEMINI VISION] -> Total tasks: {total_tasks}, Text found in: {total_responses} tasks, Total time taken: {total_time:.2f} seconds")
 return responses_dict
def transcript(image_files):
  return asyncio.run(process_images(image_files))


