from .dialogue_client import DialogueClient
from time import sleep
import torch
from models.transform_based_model import get_model, beam_predict


class AIDialogueClient(DialogueClient):
    """Uses AI model to generate messages
    """
    def __init__(self, dialogue, model=None, wait_time: float = 2):
        """Initializes AIClientDialogueClient
        :param model: AI model to predict new message
        :param wait_time: float - time to wait before sending message
        """
        super().__init__(dialogue)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not model:
            self.model = get_model(self.device)
        self.wait_time = wait_time

    def get_message(self, **kwargs) -> str:
        """Not implemented yet
        """
        sleep(self.wait_time)
        last_message = self.dialogue.get_messages()[-1].message
        reply_length = 50
        reply = beam_predict(self.model, last_message, reply_length, device=self.device)
        return reply
