from time import sleep
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException


class DialogueProxy:
    """Omegle ("https://omegle.com") proxy. Uses chrome as a browser.
    When dialogue page is opened (after creating OmegleClient object and passing captcha),
    call `self.prepare`
    """

    def __init__(self):
        """Initializes DialogueProxy
        """
        options = Options()
        options.add_argument("--start-maximized")
        options.headless = True

        self.browser = webdriver.Chrome(ChromeDriverManager().install())
        self.browser.get("https://omegle.com")

    def _test_internet_connection(self):
        """Not implemented yet
        """
        pass

    def _get_messages(self, class_name):
        messages = self.browser.find_elements(By.CLASS_NAME, class_name)
        messages = [message.text.lower() for message in messages]
        return messages

    def get_you_messages(self):
        """Find only messages written by you
        """
        return self._get_messages('youmsg')

    def get_stranger_messages(self):
        """Find only messages written by stranger
        """
        return self._get_messages('strangermsg')

    def get_all_messages(self):
        """Find all the messages by any user
        """
        return self._get_messages('notranslate')

    def prepare(self):
        """Method should be called after dialogue page is opened.
        Prepares to send messages in a new dialogue (finds required page elements etc.).
        """
        input('Press Enter after captcha is passed and dialogue is found in browser...')
        self.chat_box = self.browser.find_element(By.CSS_SELECTOR, "textarea.chatmsg")
        self.body = self.browser.find_element(By.TAG_NAME, 'body')

        self.restart_button = None
        try:
            self.find_restart_button()
        except NoSuchElementException as e:
            pass

    def find_restart_button(self):
        """
        Searches for restart button on a web page
        """
        sleep(0.1)
        self.restart_button = self.browser.find_element(By.CSS_SELECTOR, '[alt="New chat"]')

    def dialogue_is_active(self):
        """
        :return: True if dialogue is active; False otherwise
        """
        try:
            self.find_restart_button()
        except NoSuchElementException as e:
            return True
        return not self.restart_button.is_enabled()

    def restart_dialogue(self):
        """Restarts the dialogue if it ended. Does nothing otherwise
        """
        sleep(0.5)
        if self.dialogue_is_active():
            raise ValueError('can not restart dialogue which is still alive')
        self.restart_button.click()
        self.prepare()

    def send_message(self, message: str):
        """Sends message a message to omegle chat
        :param message: str - message to send
        """
        if not self.dialogue_is_active():
            return
        self.chat_box.send_keys(message)
        self.chat_box.send_keys(Keys.ENTER)

    def end_dialogue(self):
        """Ends the current dialogue with double escape
        """
        if not self.dialogue_is_active():
            return
        self.body.send_keys(Keys.ESCAPE)
        self.body.send_keys(Keys.ESCAPE)
