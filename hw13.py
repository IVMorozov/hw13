from api import API_KEY
from typing import Any
import base64
from mistralai import Mistral
from abc import ABC, abstractmethod


class RequestStrategy(ABC):
    """
    Абстрактный метод для выполнения запроса
    """
    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> dict:
        pass

# – для отправки текстовых запросов;
class  TextRequest(RequestStrategy):
    """
    Реализует отправку текстового запроса к API Mistral с накоплением истории
    """
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.client = Mistral(api_key=API_KEY)
        pass
    def execute(self, text: str, model: str, history: list = None) -> dict:
        self.text = text
        self.model = model     
        self.client = Mistral(api_key=API_KEY)

        messages = []
        if history:
            messages.extend([{'role': msg['role'], 'content': msg['content']} for msg in history])
        messages.append({'role':'user', 'content': text})

        chat_response = self.client.chat.complete(model = model, messages = messages)
        result = {'role': 'assistant', 'content': chat_response.choices[0].message.content}

        return result




# для отправки запросов с изображением;
class  ImageRequest(RequestStrategy):
    """
    Реализует отправку запроса с изображением (текст + изображение в Base64) с накоплением истории
    """
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.client = Mistral(api_key=API_KEY)
        pass

    def __encode_image(self, image_path: str) -> str:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            print(f"Ошибка: файл {image_path} не найден.")
            return ""
        except Exception as e:  # Added general exception handling
            print(f"Error: {e}")
            return ""
    
    
    def execute(self, text: str, image_path: str, model: str, history: list = None) -> dict:
        base64_image = self.__encode_image(image_path)

        messages = []
        if history:
            messages.extend([{'role': msg['role'], 'content': msg['content']} for msg in history])
        # messages.append({'role':'user', 'content': text})
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text
                },          
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                }
            ]
        } )        

        # Get the chat response
        chat_response = self.client.chat.complete(model=model, messages=messages)

        # Print the content of the response
        result = {'role': 'assistant', 'content': chat_response.choices[0].message.content}
        return result