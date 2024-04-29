from client import openai_client, chroma_client
from website_parser import WebsiteParser

urls = [
    "https://www.accel.com/",
    'https://a16z.com/',
    'https://greylock.com/',
    'http://www.benchmark.com/'
]

embeddings = []
text_contents = []
for url in urls:
    parser = WebsiteParser(url)
    embeddings.append(
        openai_client.embeddings.create(
            input=[str(parser.text_contents)],
            model="text-embedding-ada-002"
        ).data[0].embedding
    )
    text_contents.append(
        parser.text_contents
    )

collection = chroma_client.create_collection(name="VC-homepages")

collection.add(
    ids=urls,
    embeddings=embeddings,
    documents=text_contents
)

print(f'populated {collection.count()}')
