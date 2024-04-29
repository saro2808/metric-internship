from flask import Flask, render_template, request

from client import openai_client, chroma_client
from website_parser import WebsiteParser


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        url = request.form['url']
        result = process_url(url)
    return render_template('index.html', result=result)


def process_url(url):
    return "Result for URL: " + url


if __name__ == '__main__':
    app.run(debug=True)

# if __name__ == '__main__':
#
#     collection = chroma_client.get_collection(name="VC-homepages")
#     print(f'col count: {collection.count()}')
#
#     urls = [
#         'http://www.benchmark.com/'
#     ]
#     embeddings = []
#     text_contents = []
#     for url in urls:
#         parser = WebsiteParser(url)
#         embeddings.append(
#             openai_client.embeddings.create(
#                 input=[str(parser.text_contents)],
#                 model="text-embedding-ada-002"
#             ).data[0].embedding
#         )
#         text_contents.append(
#             parser.text_contents
#         )
#
#     res = collection.query(
#         query_embeddings=embeddings,
#         n_results=1
#     )
#
#     print('res', res['ids'], res['distances'])

    # collection.add(
    #     ids=urls,
    #     embeddings=embeddings,
    #     documents=text_contents
    # )
