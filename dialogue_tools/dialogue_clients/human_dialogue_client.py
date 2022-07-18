from .dialogue_client import DialogueClient


class HumanDialogueClient(DialogueClient):
    """Dialogue client for manual message writing (mostly used for testing)
    """
    def __init__(self, dialogue):
        """Initializes HumanDialogueClient
        """
        super().__init__(dialogue)
        pass

    def get_message(self, **kwargs) -> str:
        """Reads user message from standard input
        """
        return input('Enter a message to send (or quit message to end the dialogue): ')
