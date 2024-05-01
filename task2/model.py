from client import openai_client


class Model:
    def __init__(self, name, max_token_limit, func):
        self.name = name
        self.max_token_limit = max_token_limit
        self.func = func


gpt_model = 'gpt-3.5-turbo'
embed_model = 'text-embedding-ada-002'


def gpt_func(prompt):
    return openai_client.chat.completions.create(
        model=gpt_model,
        messages=[{"role": "user", "content": prompt}]
    )


def embed_func(text):
    return openai_client.embeddings.create(
        input=[text],
        model=embed_model
    ).data[0].embedding


# though these limits are higher we cut them a bit for our insurance
models = {
    gpt_model: Model(gpt_model, 16000, gpt_func),
    embed_model: Model(embed_model, 8000, embed_func)
}
