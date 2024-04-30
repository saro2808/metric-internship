from flask import Flask, render_template, request
import openai

from client import openai_client, chroma_client
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
    diagnosis = parser.get_website_diagnosis()

    keep_coeff = parser.calculate_keep_coeff(embed_model_token_limit)

    print('Embedding text...')
    while True:
        text_repr_cut = parser.get_text_repr(keep_coeff=keep_coeff)
        try:
            embedding = openai_client.embeddings.create(
                            input=[text_repr_cut],
                            model="text-embedding-ada-002"
                        ).data[0].embedding
            break
        except openai.BadRequestError as e:
            error_message = e.message if hasattr(e, 'message') else str(e)
            error_info = error_message.split(" - ")[1]
            error_details = eval(error_info)['error']
            error_type = error_details['type']
            error_message = error_details['message']
            print(f'Exception while interacting with openai: {error_type} - {error_message}')
            keep_coeff -= min(0.05, keep_coeff / 2)
            print(f'Retrying with keep_coeff={keep_coeff}...')

    collection = chroma_client.get_collection(name="VC-homepages")
    closest_urls = collection.query(
        query_embeddings=[embedding],
        n_results=3
    )
    diagnoses = closest_urls['metadatas'][0]
    return diagnosis, closest_urls, diagnoses


if __name__ == '__main__':
    app.run(debug=True)
