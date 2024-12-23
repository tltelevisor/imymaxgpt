# from app import app, db
# from app.models import Batch, catgr_batches
# import pickle, tiktoken
# from pgpt_python.client import PrivateGPTApi
# #https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
# from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
# from mistral_common.protocol.instruct.messages import UserMessage
# from sqlalchemy import insert, delete
# from mistral_common.protocol.instruct.request import ChatCompletionRequest
# tokenizer = MistralTokenizer.v1()
#
#
#
#
# if __name__ == '__main__':
#     with open('../utilits/text.txt') as f:
#     #f'{app.config["UPLOAD_FOLDER"]}/{prdctid}/{filename}'
#     #with open('.'+app.config['UPLOAD_FOLDER']+'/2/MG_recognized.txt') as f:
#         text = f.read()
#     #max_tokens = 1000
#     #delimiters = ["\n\n", "\n", ". ", " "]
#     str_list = split_strings_from_text(text, max_tokens, delimiters)
#     print(f'Файл состоит из {num_tokens(text)} токенов. Разбит на {len(str_list)} блоков')
#
#     for i,ee in enumerate(str_list):
#         print(f'Блок {i}, {num_tokens(ee)}, {ee[:50]}')