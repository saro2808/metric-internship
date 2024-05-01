from flask import Flask, render_template, request

from client import chroma_client
from website_parser import WebsiteParser

embed_model_token_limit = 8000

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    url = None
    diagnosis = None
    closest_urls = None
    diagnoses = None
    if request.method == 'POST':
        url = request.form['url']
        diagnosis, closest_urls, diagnoses = process_url(url)
    return render_template('index.html',
                           url=url,
                           diagnosis=diagnosis,
                           closest_urls=closest_urls,
                           diagnoses=diagnoses)


def process_url(url):
    parser = WebsiteParser(url, parse_hyperlinks=True)
    diagnosis = parser.get_diagnosis()

    embedding = parser.get_embeddings()

    collection = chroma_client.get_collection(name="VC-homepages")
    closest_urls = collection.query(
        query_embeddings=[embedding],
        n_results=3
    )
    diagnoses = closest_urls['metadatas'][0]
    return diagnosis, closest_urls, diagnoses


if __name__ == '__main__':
    app.run(debug=True)
