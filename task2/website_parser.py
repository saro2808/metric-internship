import sys
import logging
import json
from typing import List, Tuple, Union, Optional

import requests
from bs4 import BeautifulSoup
import openai
import tiktoken
from json2html import json2html

from models import gpt_model, embed_model, models

handler = logging.StreamHandler(stream=sys.stdout)
handler.setLevel(logging.DEBUG)


def trim_url(url: str) -> str:
    
    url_split = url.split('/')
    if url_split[-1] == '':
        return '/'.join(url_split[:-1])
    
    hash_i = url_split[-1].find('#')
    if hash_i != -1:
        url_split[-1] = url_split[-1][:hash_i]
    
    quest_i = url_split[-1].find('?')
    if quest_i != -1:
        url_split[-1] = url_split[1][:quest_i]

    return '/'.join(url_split)


def parse_error(err) -> Tuple[str, str]:
    error_message = err.message if hasattr(err, 'message') else str(err)
    error_info = error_message.split(" - ")[1]
    error_details = eval(error_info)['error']
    error_type = error_details['type']
    error_message = error_details['message']
    return error_type, error_message


class WebPage:
    """A single webpage."""

    def __init__(self, url: str, logger: logging.Logger):
        self.url = trim_url(url)
        self._raw_contents = None
        self._text_contents = None

        self._logger = logger

        self._fetch_webpage()

    def _fetch_webpage(self) -> None:

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
                self._raw_contents = BeautifulSoup(response.content, 'html.parser')

                # clear empty lines
                def _is_empty(line):
                    return len(line) == 0 or line.isspace()

                lines = self._raw_contents.get_text().split('\n')
                self._text_contents = '\n'.join([line for line in lines if not _is_empty(line)])

            else:
                self._logger.info(f'Failed to retrieve page. Status code: {response.status_code}')

        except requests.exceptions.RequestException as e:
            self._logger.error(f'Error while requesting {self.url}: {e}')

    def get_hyperlinks(self) -> List[str]:
        """
        Returns the relevant hyperlinks in the webpage.
        :returns: list of hyperlinks
        """
        hyperlinks = []
        anchors = self._raw_contents.find_all('a', href=True)
        for anchor in anchors:
            # collect only hyperlinks with sheer text
            if not anchor.find_all(recursive=False):
                anchor_url = anchor['href']
                if anchor_url[:4] != 'http':
                    anchor_url = self.url + anchor_url
                hyperlinks.append(trim_url(anchor_url))

        # remove duplicates
        hyperlink_set = set(hyperlinks)
        hyperlink_set.discard(self.url)
        hyperlinks = list(hyperlink_set)

        self._logger.info(f'Found {len(hyperlinks)} hyperlinks')

        return hyperlinks

    def get_text_repr(self, keep_coeff: float = 1) -> str:
        """
        Represents the webpage as text by keeping only the specified part.

        :param keep_coeff: Number in the interval (0, 1]. Specifies the part of the text to keep.
        :returns: The text representation.
        """
        def _cut(text):
            return text[:int(len(text) * keep_coeff)]

        text_repr = 'Contents of ' + self.url + '\n'
        text_repr += str(self._text_contents) + '\n\n'
        return _cut(text_repr)


class WebsiteParser:
    """
    Website parser.

    :ivar homepage: Represents the website's homepage.
    :type homepage: WebPage
    :ivar subpages: Webpages reachable from homepage via a hyperlink.
    :type subpages: List[WebPage]
    """

    def __init__(self, url: str, parse_hyperlinks: bool = False):

        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging.DEBUG)
        self._logger.addHandler(handler)

        self.homepage = WebPage(url, self._logger)
        self.subpages: List[WebPage] = []  # will hold webpages of hyperlinks
        self._parse_hyperlinks = parse_hyperlinks

        self._logger.info(f'WebsiteParser instance created for {url}')

        if parse_hyperlinks:
            self._parse_subpages()
        else:
            self._logger.info('Not parsing hyperlinks')

    def _parse_subpages(self) -> None:
        """Fetches subpages one hyperlink apart from the homepage."""
        hyperlinks = self.homepage.get_hyperlinks()
        for hyperlink in hyperlinks:
            subpage = WebPage(hyperlink, self._logger)
            self.subpages.append(subpage)

    def get_text_repr(self, keep_coeff: float = 1, model: Optional[str] = None) -> Tuple[float, str]:
        """
        Represents the website as text by concatenating the text representations of its pages.

        keep_coeff is calculated depending on the model's max token limit.
        :param keep_coeff: Number in the interval (0, 1]. Specifies the part of the text to keep.
        :param model: The model name.
        :return:
        """
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

    def _evoke_model(self, model: str) -> Union[Optional[str], List[float]]:
        """Loops until the model responds without errors."""

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

    def get_diagnosis(self) -> str:
        """
        Summarizes website info.
        :return: The diagnosis of the website in JSON format
        """
        self._logger.info('Waiting for openai to extract website info...')

        while True:
            diagnosis = self._evoke_model(gpt_model)
            try:
                diagnosis_json = json.loads(diagnosis)
                return json2html.convert(json=diagnosis_json)

            except json.decoder.JSONDecodeError as e:
                self._logger.error(f'Exception while converting response to json: {e}')
                self._logger.info('Retrying...')

    def get_embedding(self) -> List[float]:
        """
        Embeds the website.
        :return: The embedding.
        """
        self._logger.info('Embedding website...')
        return self._evoke_model(embed_model)
