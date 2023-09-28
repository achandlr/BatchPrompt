import openai
import time

def read_api_token(token_path):
    # Read API token from a dedicated file
    with open(token_path, "r") as f:
        API_TOKEN = f.read().strip()
    return API_TOKEN
    

def query_model(model, prompt):
    message = [{"role": "user", "content": prompt}]
    # Estimate the number of tokens used
    estimated_tokens = len(prompt.split()) * 3
    # Set the rate limits for different models
    rate_limit = 10_000 if "gpt-4" in model else 90_000
    try_cnt = 0
    max_try_cnt = 10
    while try_cnt < max_try_cnt:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=message,
                temperature=0.2, # TODO: experiment with temperature
                # max_tokens=4000,
                frequency_penalty=0.0
            )
            break
        except Exception as e:
         # except openai.error.RateLimitError as e:
            # Calculate the wait time based on the estimated token usage and rate limit
            wait_time = (estimated_tokens / rate_limit) * 60 * (1+try_cnt/4)
            print(f"{str(e)} occured. Waiting for {wait_time} seconds...")
            time.sleep(wait_time)
            try_cnt +=1
            continue
    if try_cnt == max_try_cnt:
        raise Exception("Errors occurred too many times. Aborting...")
    text_response = response["choices"][0]["message"]["content"]
    return text_response

def set_api_key(api_token_path):
    openai.api_key = read_api_token(api_token_path)