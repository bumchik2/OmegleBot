class DialogueClient:
    """Base dialogue client class
    """
    def __init__(self, dialogue):
        """Initializes DialogueClient
        """
        self.dialogue = dialogue

    def get_message(self, **kwargs) -> str:
        """Gets message to send. If `quit()` is received, the dialogue should end
        """
        return 'Hi, I guess :/'
