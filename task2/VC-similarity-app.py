from flask import Flask, render_template, request

from client import openai_client, chroma_client
from website_parser import WebsiteParser


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    closest_urls = None
    if request.method == 'POST':
        url = request.form['url']
        closest_urls = process_url(url)
    return render_template('index.html',
                           closest_urls=closest_urls)


def process_url(url):
    parser = WebsiteParser(url)
    embedding = openai_client.embeddings.create(
                    input=[str(parser.text_contents)],
                    model="text-embedding-ada-002"
                ).data[0].embedding

    collection = chroma_client.get_collection(name="VC-homepages")

    closest_urls = collection.query(
        query_embeddings=[embedding],
        n_results=2
    )
    return closest_urls


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
