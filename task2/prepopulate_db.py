from client import chroma_client
from website_parser import WebsiteParser

urls = [
    'http://www.benchmark.com/',
    # 'http://www.lsvp.com/',
    # 'https://www.accel.com/',
    # 'https://a16z.com/',
    # 'https://greylock.com/',
    # 'http://www.sequoiacap.com/',
    # 'http://www.indexventures.com/',
    # 'http://www.kpcb.com/',
    # 'http://www.matrixpartners.com/',
    'http://www.500.co/',
    # 'http://www.sparkcapital.com/',
    # 'http://www.insightpartners.com/'
]

metadatas = []
embeddings = []
text_contents = []

parsed_urls_count = 0

for url in urls:
    parser = WebsiteParser(url, parse_hyperlinks=True)

    embeddings.append(
        parser.get_embedding()
    )
    text_contents.append(
        parser.get_text_repr()
    )
    metadatas.append(
        {'diagnosis': parser.get_diagnosis()}
    )

    parsed_urls_count += 1
    print(f'Parsed {parsed_urls_count} out of {len(urls)} urls to populate the database\n')

collection = chroma_client.create_collection(name="VC-homepages")

collection.add(
    ids=urls,
    embeddings=embeddings,
    documents=text_contents,
    metadatas=metadatas
)

print(f'Populated {collection.count()} items into VC-homepages db')
