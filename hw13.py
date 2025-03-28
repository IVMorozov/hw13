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
    
# фасад для объединения функционала и удобного взаимодействия пользователя с системой
class  ChatFacade:  
    """
    Создание экземпляров стратегий и инициализация модели 
    """
    def __init__(self, api_key: str) -> None: 
        self.api_key = api_key
        self.models: dict[str, list[str]] = {'text':["mistral-large-latest"], 'image':["pixtral-12b-2409"]}
        self.request: TextRequest|ImageRequest = self.change_strategy()
        self.model: str = self.__set_model()
        self.history:list[Any]= []
        
    #  Инициализация с API-ключом. Создаются экземпляры `TextRequest` и `ImageRequest`, а также инициализируется список доступных моделей.
    def change_strategy(self) -> TextRequest|ImageRequest:
        mode:str = input('Укажите тип запроса: 1 – текстовый, 2 – с изображением: ')
        if mode == '1':
            self.mode = '1'
            return TextRequest(api_key=self.api_key)
        elif mode == '2':
            self.mode = '2'
            return ImageRequest(api_key=self.api_key)
        else:
            raise ValueError("Неверныый режим запроса")
        
    def __set_model(self) -> str: 
        if self.mode == '1':
            model:str = "mistral-large-latest"
        
        if self.mode == '2':
            model:str = "pixtral-12b-2409"

        return model    

    #  Метод для загрузки изображения (если выбран режим с изображением). Отвечает за валидацию пути и преобразование изображения в Base64 (может делегировать эту задачу классу `ImageRequest`).
    def ask_question(self, text: str, image_path: str = None) -> dict:
        user_message = {'role': 'user', 'content': text}
        current_history = [msg for _, msg in self.history]

        if image_path:
            response:dict[Any, Any] = self.request.execute(text=text, image_path=image_path, history=current_history, model=self.model)
        else: 
            response:dict[Any, Any] = self.request.execute(text=text, history=current_history, model=self.model)
        self.history.append((text, user_message))
        self.history.append((text, response))
        return response
    
    def __call__(self):
        # text:str = input('\n Введите текст запроса')
        text:str = 'расскажи шутку про французов'
        image_path = None
        if isinstance (self.request, ImageRequest):
            # image_path:str = input('ВВедите путь к изображению') 
            text:str = 'опиши картинку'
            image_path:str = 'lemon.jpg'
        response:dict[Any, Any] = self.ask_question(text=text, image_path=image_path if image_path else None)
        print(response)

chat_facade = ChatFacade(api_key=API_KEY)
chat_facade()