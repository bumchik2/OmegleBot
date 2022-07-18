from dialogue_tools import Dialogue
from dialogue_tools import DialogueProxy
from dialogue_tools import AIDialogueClient
from dialogue_tools import HumanDialogueClient
from dialogue_tools import DialogueManager
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_type', type=str, choices=('human', 'ai'), required=True)
    args = parser.parse_args()
    return args


def main():
    """ Starts an Oglemeo dialogue in a browser,
    allows you to chat depending on the selected client
    """
    args = parse_args()

    dialogue = Dialogue()
    proxy = DialogueProxy()
    client = AIDialogueClient(dialogue, model=None) \
        if args.client_type == 'ai' else HumanDialogueClient(dialogue)
    manager = DialogueManager(dialogue, proxy, client)

    manager.prepare()

    while True:
        try:
            manager.run()
            manager.restart_dialogue()
        except KeyboardInterrupt:
            print('Stopping chatting...')
            break


if __name__ == '__main__':
    main()


# example: python main.py --client_type=ai
