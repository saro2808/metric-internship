import openai

from client import openai_client, chroma_client
from website_parser import WebsiteParser

embed_model_token_limit = 8000

urls = [
    'https://a16z.com/',
    "https://www.accel.com/",
    'https://greylock.com/',
    'http://www.benchmark.com/'
]

metadatas = []
embeddings = []
text_contents = []
for url in urls:
    parser = WebsiteParser(url, parse_hyperlinks=True)

    keep_coeff = parser.calculate_keep_coeff(embed_model_token_limit)

    print('Embedding text...')
    while True:
        text_repr_cut = parser.get_text_repr(keep_coeff=keep_coeff)
        try:
            embeddings.append(
                openai_client.embeddings.create(
                    input=[text_repr_cut],
                    model="text-embedding-ada-002"
                ).data[0].embedding
            )
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

    text_contents.append(
        parser.get_text_repr()
    )
    metadatas.append(
        parser.get_website_diagnosis()
    )

collection = chroma_client.create_collection(name="VC-homepages")

collection.add(
    ids=urls,
    embeddings=embeddings,
    documents=text_contents,
    metadatas=metadatas
)

print(f'Populated {collection.count()} items into VC-homepages db')
