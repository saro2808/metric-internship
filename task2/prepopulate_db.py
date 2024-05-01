from client import chroma_client
from website_parser import WebsiteParser

urls = [
    "https://www.accel.com/",
    'http://www.500.co/',
    'https://a16z.com/',
    'https://greylock.com/',
    'http://www.benchmark.com/'
]

metadatas = []
embeddings = []
text_contents = []
for url in urls:
    parser = WebsiteParser(url, parse_hyperlinks=True)

    embeddings.append(
        parser.get_embeddings()
    )
    text_contents.append(
        parser.get_text_repr()
    )
    metadatas.append(
        parser.get_diagnosis()
    )

collection = chroma_client.create_collection(name="VC-homepages")

collection.add(
    ids=urls,
    embeddings=embeddings,
    documents=text_contents,
    metadatas=metadatas
)

print(f'Populated {collection.count()} items into VC-homepages db')
