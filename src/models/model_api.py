import together
import openai
import backoff
from typing import List, Dict, Any, Tuple, TypedDict, Optional


def read_api_token(token_path : str) -> str:
    # Read API token from a dedicated file
    with open(token_path, "r") as f:
        API_TOKEN = f.read().strip()
    return API_TOKEN

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
    def query(self, prompt : str) -> str:
        message = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=self.model_name,
            message=message,
            **self.generation_params,
        )
        text_response = response["choices"][0]["message"]["content"]
        return text_response
    

if __name__ == "__main__":
    model = TogetherAIModel(
        api_token=read_api_token("RJHA_TOGETHER_AI_TOKEN.txt"),
        model_name="togethercomputer/llama-2-7b",
        generation_params={}
    )