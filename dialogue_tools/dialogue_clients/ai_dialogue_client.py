from .dialogue_client import DialogueClient
from time import sleep


class AIDialogueClient(DialogueClient):
    """Uses AI model to generate messages
    """
    def __init__(self, dialogue, model, wait_time: float = 1.5):
        """Initializes AIClientDialogueClient
        :param model: AI model to predict new message
        :param wait_time: float - time to wait before sending message
        """
        super().__init__(dialogue)
        self.model = model
        self.wait_time = wait_time

    def get_message(self, **kwargs) -> str:
        """Not implemented yet
        """
        sleep(self.wait_time)
        return 'Not implemented yet :/'
