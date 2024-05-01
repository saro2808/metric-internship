import sys
import logging
import json

import requests
from bs4 import BeautifulSoup
import openai
import tiktoken

from models import gpt_model, embed_model, models

handler = logging.StreamHandler(stream=sys.stdout)
handler.setLevel(logging.DEBUG)


def trim_url(url):
    url_split = url.split('/')
    if url_split[-1] == '':
        return '/'.join(url_split[:-1])
    hash_i = url_split[-1].find('#')
    if hash_i != -1:
        url_split[-1] = url_split[-1][:hash_i]
    return '/'.join(url_split)


def parse_error(err):
    error_message = err.message if hasattr(err, 'message') else str(err)
    error_info = error_message.split(" - ")[1]
    error_details = eval(error_info)['error']
    error_type = error_details['type']
    error_message = error_details['message']
    return error_type, error_message


class WebPage:

    def __init__(self, url, logger):
        self.url = trim_url(url)
        self.raw_contents = None
        self._text_contents = None

        self._logger = logger

        self._preprocess_webpage()

    def _preprocess_webpage(self):

        self._logger.info('Processing ' + self.url)

        # define headers with a user-agent string that mimics a typical web browser
        # to possibly avoid receiving 403 Forbidden
        headers = {
            "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/96.0.4664.110 Safari/537.36'
        }

        try:
            response = requests.get(self.url, headers=headers)

            if response.status_code == 200:
                self.raw_contents = BeautifulSoup(response.content, 'html.parser')

                # clear empty lines
                def _is_empty(line):
                    return len(line) == 0 or line.isspace()

                lines = self.raw_contents.get_text().split('\n')
                self._text_contents = '\n'.join([line for line in lines if not _is_empty(line)])

            else:
                self._logger.info(f'Failed to retrieve page. Status code: {response.status_code}')

        except requests.exceptions.RequestException as e:
            self._logger.error(f'Error while requesting {self.url}: {e}')

    def get_text_repr(self, keep_coeff=1):
        def _cut(text):
            return text[:int(len(text) * keep_coeff)]

        text_repr = 'Contents of ' + self.url + '\n'
        text_repr += str(self._text_contents) + '\n\n'
        return _cut(text_repr)


class WebsiteParser:

    def __init__(self, url, parse_hyperlinks=False):

        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging.DEBUG)
        self._logger.addHandler(handler)

        self._parse_hyperlinks = parse_hyperlinks
        self.homepage = WebPage(url, self._logger)
        self.subpages = []  # will hold webpages of hyperlinks

        self._logger.info(f'WebsiteParser instance has been created for {url}')

        if parse_hyperlinks:
            self._preprocess_hyperlinks()
        else:
            self._logger.info('Not parsing hyperlinks')

    def _preprocess_hyperlinks(self):
        hyperlinks = []
        anchors = self.homepage.raw_contents.find_all('a', href=True)
        for anchor in anchors:
            # collect only hyperlinks with sheer text
            if not anchor.find_all(recursive=False):
                anchor_url = anchor['href']
                if anchor_url[:4] != 'http':
                    anchor_url = self.homepage.url + anchor_url
                hyperlinks.append(trim_url(anchor_url))

        # remove duplicates
        hyperlink_set = set(hyperlinks)
        hyperlink_set.discard(self.homepage.url)
        hyperlinks = list(hyperlink_set)

        self._logger.info(f'Found {len(hyperlinks)} hyperlinks')

        for hyperlink in hyperlinks:
            subpage = WebPage(hyperlink, self._logger)
            self.subpages.append(subpage)

    def get_text_repr(self, keep_coeff=1, model=None):

        if model:
            enc = tiktoken.encoding_for_model(model)
            _, text_repr = self.get_text_repr()
            token_count = len(enc.encode(text_repr))
            keep_coeff = min(models[model].max_token_limit / token_count, 1)

            self._logger.debug(f'Token count: {token_count}')
            self._logger.debug(f'Keep coefficient: {keep_coeff}')

            return self.get_text_repr(keep_coeff=keep_coeff)

        if not self._parse_hyperlinks:
            return keep_coeff, self.homepage.get_text_repr(keep_coeff)

        text_repr = ''
        for page in [self.homepage, *self.subpages]:
            text_repr += page.get_text_repr(keep_coeff)

        return keep_coeff, text_repr

    def _evoke_model(self, model):
        keep_coeff, text_repr_cut = self.get_text_repr(model=model)

        while True:
            if model == gpt_model:
                args = f'''Below is the text part of a website.
                    Tell me the following after parsing it:
                    1. "VC name",
                    2. "Contacts",
                    3. "Industries they invest in",
                    4. "Investment rounds they participate/lead".
                    Provide your answer in a json format.
                    {text_repr_cut}'''
            else:  # if model == embed_model
                args = text_repr_cut

            try:
                return models[model].func(args)

            except openai.BadRequestError as e:
                error_type, error_message = parse_error(e)
                self._logger.error(f'Exception while interacting with openai: {error_type} - {error_message}')

                keep_coeff -= min(0.05, keep_coeff / 2)
                _, text_repr_cut = self.get_text_repr(keep_coeff=keep_coeff)
                self._logger.info(f'Retrying with keep_coeff={keep_coeff}...')

    def get_diagnosis(self):

        self._logger.info('Waiting for openai to extract website info...')

        while True:
            stream = self._evoke_model(gpt_model)
            try:
                message_json = json.loads(stream.choices[0].message.content)
                message_dict = {key: str(message_json[key]) for key in message_json}
                self._logger.info('')
                return message_dict
            except json.decoder.JSONDecodeError as e:
                self._logger.error(f'Exception while converting response to json: {e}')
                self._logger.info('Retrying...')

    def get_embeddings(self):
        self._logger.info('Embedding website...')
        return self._evoke_model(embed_model)
