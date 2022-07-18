from time import sleep
from dialogue_tools.dialogue_clients import DialogueClient
from dialogue_tools import Dialogue
from dialogue_tools import DialogueProxy


class DialogueManager:
    """Manages dialogue
    """

    def __init__(self, dialogue: Dialogue, proxy: DialogueProxy, client: DialogueClient, wait_time: float = 0.1,
                 quit_message: str = 'quit()'):
        """Initializes DialogueManager
        :param dialogue: Dialogue object
        :param proxy: DialogueProxy object
        :param client: DialogueClient object
        :param wait_time: float - time to wait between dialogue updates
        """
        self.dialogue = dialogue
        self.proxy = proxy
        self.client = client
        self.wait_time = wait_time
        self.quit_message = quit_message

    def restart_dialogue(self):
        self.dialogue.__init__()
        self.proxy.restart_dialogue()

    def update_dialogue(self):
        self.dialogue.update_messages(self.proxy.get_all_messages(),
                                      self.proxy.get_you_messages(),
                                      self.proxy.get_stranger_messages())

    def prepare(self):
        self.dialogue.__init__()
        self.proxy.prepare()

    def run(self):
        iteration = 0
        while self.proxy.dialogue_is_active():
            sleep(self.wait_time)
            self.update_dialogue()

            if self.dialogue.pointer_stranger > iteration:  # get message after every stranger message
                new_message = self.client.get_message()
                if new_message == self.quit_message:
                    self.proxy.end_dialogue()
                else:
                    self.proxy.send_message(new_message)
                iteration += 1
