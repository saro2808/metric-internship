import sys
import logging
import json

import requests
from bs4 import BeautifulSoup
from client import openai_client

handler = logging.StreamHandler(stream=sys.stdout)
handler.setLevel(logging.DEBUG)


class WebsiteParser:

    def __init__(self, url, parse_hyperlinks=False):
        self.url = url.rstrip('/')
        self.raw_contents = ''
        self.text_contents = ''
        self.hyperlinks = []
        self.hyperlink_raw_contents = []
        self.hyperlink_text_contents = []

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(handler)

        self.logger.info(f'WebsiteParser instance has been created for {self.url}')

        self._preprocess_website(url, in_homepage=True)
        if parse_hyperlinks:
            for hyperlink in self.hyperlinks:
                self._preprocess_website(hyperlink)

        self.logger.info('')

    def _preprocess_website(self, url, in_homepage=False):

        self.logger.info('processing ' + url)

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
                    self.raw_contents = raw_contents
                    self.text_contents = text_contents
                    anchors = self.raw_contents.find_all('a', href=True)
                    for anchor in anchors:
                        # collect only hyperlinks with sheer text
                        if not anchor.find_all(recursive=False):
                            anchor_url = anchor['href']
                            if anchor_url[:4] != 'http':
                                anchor_url = self.url + anchor_url
                            self.hyperlinks.append(anchor_url)
                    self._trim_hyperlinks()
                    self.logger.info(f'found {len(self.hyperlinks)} hyperlinks')
                else:
                    self.hyperlink_raw_contents.append(raw_contents)
                    self.hyperlink_text_contents.append(text_contents)

            else:
                if not in_homepage:
                    self.hyperlink_raw_contents.append(f'Error {response.status_code}')
                    self.hyperlink_text_contents.append(f'Error {response.status_code}')
                self.logger.info(f'Failed to retrieve page. Status code: {response.status_code}')

        except requests.exceptions.RequestException as e:
            self.logger.error("Error: " + str(e))

    def _trim_hyperlinks(self):
        for i in range(len(self.hyperlinks)):
            link = self.hyperlinks[i]
            link_split = link.split('/')
            if link_split[-1] == '':
                self.hyperlinks[i] = '/'.join(link_split[:-1])
            else:
                hash_i = link_split[-1].find('#')
                if hash_i != -1:
                    link_split[-1] = link_split[-1][:hash_i]
                self.hyperlinks[i] = '/'.join(link_split)
        self.hyperlinks = list(set(self.hyperlinks))

    def get_text_repr(self):
        urls = self.hyperlinks
        text_contents = self.hyperlink_text_contents
        if self.url not in self.hyperlinks:
            urls = [self.url, *self.hyperlinks]
            text_contents = [self.text_contents, *self.hyperlink_text_contents]
        text_repr = ''
        for i in range(len(urls)):
            text_repr += f'Contents of {urls[i]}\n'
            text_repr += text_contents[i] + '\n\n'
        return text_repr

    def get_html_repr(self):
        raw_contents = BeautifulSoup(str(self.raw_contents))
        for tag in ['img', 'style', 'script', 'noscript', 'iframe']:
            for item in raw_contents.find_all(tag):
                item.extract()
        return raw_contents

    def get_website_diagnosis(self):
        text_repr = self.get_text_repr()
        stream = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user",
                       "content": f'''This is the text part of a web page. Tell me the following after parsing it:
                                       1. VC name,
                                       2. contacts,
                                       3. industries they invest in,
                                       4. investment rounds they participate/lead.
                                       Provide your answer in a json format.
                                       {text_repr}'''}]
        )
        return json.loads(stream.choices[0].message.content)
