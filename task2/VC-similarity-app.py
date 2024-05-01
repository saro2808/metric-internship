from flask import Flask, render_template, request

from client import chroma_client
from website_parser import WebsiteParser

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    url = None
    diagnosis = None
    closest_urls = None
    diagnoses_of_results = None
    if request.method == 'POST':
        url = request.form['url']
        diagnosis, closest_urls, diagnoses_of_results = process_url(url)
    return render_template('index.html',
                           url=url,
                           diagnosis=diagnosis,
                           closest_urls=closest_urls,
                           diagnoses_of_results=diagnoses_of_results)


def process_url(url):
    parser = WebsiteParser(url, parse_hyperlinks=True)
    diagnosis = parser.get_diagnosis()
    embedding = parser.get_embedding()

    collection = chroma_client.get_collection(name="VC-homepages")
    closest_urls = collection.query(
        query_embeddings=[embedding],
        n_results=3
    )
    diagnoses_of_results = [item['diagnosis'] for item in closest_urls['metadatas'][0]]
    return diagnosis, closest_urls, diagnoses_of_results


if __name__ == '__main__':
    app.run(debug=True)
