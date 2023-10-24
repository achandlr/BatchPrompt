import openai
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

def read_api_token(token_path):
    # Read API token from a dedicated file
    with open(token_path, "r") as f:
        API_TOKEN = f.read().strip()
    return API_TOKEN
    
def query_model(model, prompt, model_temperature=.2, timeout=10):
    message = [{"role": "user", "content": prompt}]
    # Estimate the number of tokens used
    estimated_tokens = len(prompt.split()) * 3
    # Set the rate limits for different models
    rate_limit = 10_000 if "gpt-4" in model else 90_000  
    try_cnt = 0
    max_try_cnt = 10  
    while try_cnt < max_try_cnt:
        with ThreadPoolExecutor() as executor:
            future = executor.submit(openai.ChatCompletion.create,
                                     model=model,
                                     messages=message,
                                     temperature=model_temperature,
                                     frequency_penalty=0.0)
            try:
                response = future.result(timeout=timeout)
                text_response = response["choices"][0]["message"]["content"]
                return text_response
            except (TimeoutError, Exception) as e:
                wait_time = (estimated_tokens / rate_limit) * 60 * (1 + try_cnt / 4)
                print(f"Error {str(e)} occurred. Waiting for {wait_time} seconds...")
                time.sleep(wait_time)
                try_cnt += 1
    
    raise Exception(f"Errors occurred too many times. Aborting...")

def set_api_key(api_token_path):
    openai.api_key = read_api_token(api_token_path)