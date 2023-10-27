import openai
import time
import together
import backoff
from typing import List, Dict, Any, Tuple, TypedDict, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError

def read_api_token(token_path):
    # Read API token from a dedicated file
    with open(token_path, "r") as f:
        API_TOKEN = f.read().strip()
    return API_TOKEN

def set_api_key(api_token_path):
    openai.api_key = read_api_token(api_token_path)

class LanguageModel:
    def __init__(self):
        raise NotImplementedError

    def query(self, prompt : str) -> str:
        raise NotImplementedError

class TogetherAIModel(LanguageModel):
    """ 
    A wrapper class for the Together AI API.
    
    NOTE: do NOT specify any default values for model/generation parameters.
    I'd like to keep the configurations separate from the code as much as possible
    so we have good documentation/reproducibility of the experiments without having
    to dig into the code/commit history.

    Attributes:
        api_token (str): The API token for the Together AI API.
        model_name (str): The name of the model to use.
            Llama2-70B is togethercomputer/llama-2-70b, for example.
        generation_params (dict): The parameters to use for generation.
            - max_tokens (int): The maximum number of tokens to generate.
            - temperature (float): The temperature to use for generation.
            - top_p (float): The top_p to use for generation.
            - top_k (int): The top_k to use for generation.
            - repetition_penalty (float): The repetition penalty to use for generation.
            - logprobs (int): The number of logprobs to return.
    """
    def __init__(
        self,
        api_token : str,
        model_name : str,
        generation_params : dict,
    ):
        together.api_key = api_token
        self.api_token = api_token
        self.model_name = model_name
        self.generation_params = generation_params

    def __repr__(self):
        return f"TogetherAIModel(model_name={self.model_name}, generation_params={self.generation_params})"

    @backoff.on_exception(backoff.expo, Exception, max_tries=10)
    def query(self, prompt : str) -> str:
        response = together.Complete.create(
            prompt = prompt,
            model = self.model_name,
            **self.generation_params,
        )
        return response["output"]["choices"][0]["text"]
    

class OpenAIModel(LanguageModel):
    def __init__(
        self,
        api_token,
        model_name,
        generation_params,
    ):
        openai.api_key = api_token
        self.api_token = api_token
        self.model_name = model_name
        self.generation_params = generation_params

    def __repr__(self):
        return f"OpenAIModel(model_name={self.model_name}, generation_params={self.generation_params})"
    
    @backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_tries=10)
    def query(self, prompt : str, timeout = 10) -> str:
        message = [{"role": "user", "content": prompt}]
        estimated_tokens = len(prompt.split()) * 3
        # Set the rate limits for different models
        rate_limit = 10_000 if "gpt-4" in self.model_name else 90_000  
        try_cnt = 0
        max_try_cnt = 10  
        while try_cnt < max_try_cnt:
            with ThreadPoolExecutor() as executor:
                future = executor.submit(openai.ChatCompletion.create,
                                        model=self.model_name,
                                        messages=message,
                                        **self.generation_params)
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
if __name__ == "__main__":
    model = OpenAIModel(
        api_token=read_api_token("data//imported//datasets//api_token.txt"),
        model_name="gpt-3.5-turbo",
        generation_params={"temperature" : .2} # TODO: Not sure if temperature will work as input as string
    )
    output = model.query("What is the meaning of life?")
    print("DONE")
    # model = TogetherAIModel(
    #     api_token=read_api_token("RJHA_TOGETHER_AI_TOKEN.txt"),
    #     model_name="togethercomputer/llama-2-7b",
    #     generation_params={}
    # )
