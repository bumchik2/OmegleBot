from dataclasses import dataclass
from typing import List


def matches(message1: str, message2: str):
    """Checks if these two messages are the same
    """
    return message1.lower() == message2.lower()


@dataclass
class Message:
    sender: str
    message: str


class Dialogue:
    def __init__(self):
        """Initializes Dialogue
        """
        self.messages = []  # list of Message objects
        self.pointer_you = 0
        self.pointer_stranger = 0

    def update_messages(self, all_messages: List[str], you_messages: List[str], stranger_messages: List[str]):
        """Adds new messages into account
        :param all_messages: list of strings ['message1', 'message2', ...]
        :param you_messages: list of strings ['You: message1: ', ...]
        :param stranger_messages: list of strings ['Stranger: message1: ', ...]
        """
        all_messages = [message for message in all_messages if message != '']

        if len(all_messages) != len(you_messages) + len(stranger_messages):
            print('Failed to update messages - inconsistent data')
            return

        all_messages = [message.lower() for message in all_messages]
        you_messages = [message.lower() for message in you_messages]
        stranger_messages = [message.lower() for message in stranger_messages]

        for message in all_messages[self.pointer_you + self.pointer_stranger:]:
            if self.pointer_you != len(you_messages):
                you_message = you_messages[self.pointer_you][len('you: '):].lower()
                if matches(you_message, message):
                    self.messages.append(Message('you', message))
                    self.pointer_you += 1
                    continue

            if self.pointer_stranger != len(stranger_messages):
                stranger_message = stranger_messages[self.pointer_stranger][len('stranger: '):].lower()
                if matches(stranger_message, message):
                    self.messages.append(Message('stranger', message))
                    self.pointer_stranger += 1
                    continue

        # all messages should have matched
        assert len(all_messages) == len(self.messages), \
            f'{all_messages} length should be same as {self.messages} length'

    def get_messages(self) -> List[Message]:
        return self.messages
