import sys
import logging
import json

import requests
from bs4 import BeautifulSoup
import openai
import tiktoken

from client import openai_client

handler = logging.StreamHandler(stream=sys.stdout)
handler.setLevel(logging.DEBUG)


# TODO add Page
class WebsiteParser:

    def __init__(self, url, parse_hyperlinks=False):
        self.url = url.rstrip('/')
        self._raw_contents = ''
        self._text_contents = ''

        self._parse_hyperlinks = parse_hyperlinks

        self._hyperlinks = []
        self._hyperlink_raw_contents = []
        self._hyperlink_text_contents = []

        self._model_max_token_limit = 16000

        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging.DEBUG)
        self._logger.addHandler(handler)

        self._logger.info(f'WebsiteParser instance has been created for {self.url}')

        self._preprocess_website(url, in_homepage=True)
        if parse_hyperlinks:
            for hyperlink in self._hyperlinks:
                self._preprocess_website(hyperlink)
        else:
            self._logger.info('Not parsing hyperlinks')

    def _preprocess_website(self, url, in_homepage=False):

        self._logger.info('Processing ' + url)

        # define headers with a user-agent string that mimics a typical web browser
        # to possibly avoid receiving 403 Forbidden
        headers = {
            "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
        }

        try:
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                raw_contents = BeautifulSoup(response.content, 'html.parser')

                # clear empty lines
                def _is_empty(line):
                    return len(line) == 0 or line.isspace()

                text_contents = '\n'.join([line for line in raw_contents.get_text().split('\n') if not _is_empty(line)])

                # if the url specifies homepage collect useful hyperlinks
                if in_homepage:
                    self.title = raw_contents.find_all('title')[0].text
                    self._raw_contents = raw_contents
                    self._text_contents = text_contents
                    if self._parse_hyperlinks:
                        anchors = self._raw_contents.find_all('a', href=True)
                        for anchor in anchors:
                            # collect only hyperlinks with sheer text
                            if not anchor.find_all(recursive=False):
                                anchor_url = anchor['href']
                                if anchor_url[:4] != 'http':
                                    anchor_url = self.url + anchor_url
                                self._hyperlinks.append(anchor_url)
                        self._trim_hyperlinks()
                        self._logger.info(f'Found {len(self._hyperlinks)} hyperlinks')
                else:
                    self._hyperlink_raw_contents.append(raw_contents)
                    self._hyperlink_text_contents.append(text_contents)

            else:
                if not in_homepage:
                    self._hyperlink_raw_contents.append(f'Error {response.status_code}')
                    self._hyperlink_text_contents.append(f'Error {response.status_code}')
                self._logger.info(f'Failed to retrieve page. Status code: {response.status_code}')

        except requests.exceptions.RequestException as e:
            self._logger.error("Error: " + str(e))

    def _trim_hyperlinks(self):
        for i in range(len(self._hyperlinks)):
            link = self._hyperlinks[i]
            link_split = link.split('/')
            if link_split[-1] == '':
                self._hyperlinks[i] = '/'.join(link_split[:-1])
            else:
                hash_i = link_split[-1].find('#')
                if hash_i != -1:
                    link_split[-1] = link_split[-1][:hash_i]
                self._hyperlinks[i] = '/'.join(link_split)
        self._hyperlinks = list(set(self._hyperlinks))

    def get_text_repr(self, keep_coeff: float = 1):
        def _cut(text):
            return text[:int(len(text) * keep_coeff)]

        if not self._parse_hyperlinks:
            return _cut(self._text_contents)

        urls = self._hyperlinks
        text_contents = self._hyperlink_text_contents
        if self.url not in self._hyperlinks:
            urls = [self.url, *self._hyperlinks]
            text_contents = [self._text_contents, *self._hyperlink_text_contents]
        text_repr = ''
        for i in range(len(urls)):
            text_repr += f'Contents of {urls[i]}\n'
            text_repr += _cut(text_contents[i]) + '\n\n'
        return text_repr

    def get_html_repr(self):
        raw_contents = BeautifulSoup(str(self._raw_contents))
        for tag in ['img', 'style', 'script', 'noscript', 'iframe']:
            for item in raw_contents.find_all(tag):
                item.extract()
        return raw_contents

    def calculate_keep_coeff(self, max_token_limit=None):
        if not max_token_limit:
            max_token_limit = self._model_max_token_limit

        def _calculate_token_count(text):
            enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
            return len(enc.encode(text))

        text_repr = self.get_text_repr()
        token_count = _calculate_token_count(text_repr)
        keep_coeff = min(max_token_limit / token_count, 1)

        self._logger.debug(f'Token count: {token_count}')
        self._logger.debug(f'Keep coefficient: {keep_coeff}')

        return keep_coeff

    def get_website_diagnosis(self):

        self._logger.info('Waiting for openai to extract website info...')

        keep_coeff = self.calculate_keep_coeff()
        text_repr_cut = self.get_text_repr(keep_coeff=keep_coeff)

        response_successful = False
        json_conversion_successful = False
        while not response_successful or not json_conversion_successful:
            try:
                stream = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user",
                               "content": f'''This is the text part of a web page. Tell me the following after parsing it:
                                               1. "VC name",
                                               2. "Contacts",
                                               3. "Industries they invest in",
                                               4. "Investment rounds they participate/lead".
                                               Provide your answer in a json format.
                                               {text_repr_cut}'''}]
                )
                response_successful = True
            except openai.BadRequestError as e:
                error_message = e.message if hasattr(e, 'message') else str(e)
                error_info = error_message.split(" - ")[1]
                error_details = eval(error_info)['error']
                error_type = error_details['type']
                error_message = error_details['message']
                self._logger.error(f'Exception while interacting with openai: {error_type} - {error_message}')
                keep_coeff -= min(0.05, keep_coeff / 2)
                text_repr_cut = self.get_text_repr(keep_coeff=keep_coeff)
                self._logger.info(f'Retrying with keep_coeff={keep_coeff}...')
                continue

            try:
                message_json = json.loads(stream.choices[0].message.content)
                message_dict = {key: str(message_json[key]) for key in message_json}
                json_conversion_successful = True
            except json.decoder.JSONDecodeError as e:
                self._logger.exception(f'Exception while converting response to json: {e}')
                self._logger.info('Retrying...')

        self._logger.info('')
        return message_dict
