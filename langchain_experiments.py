import getpass
import os
import streamlit as st
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
# from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
from langchain.chains.query_constructor.schema import AttributeInfo
# from langchain_core.documents import Document
from elasticsearch import Elasticsearch
from langchain_elasticsearch import ElasticsearchStore

st.title("🦜🔗 First Attempt")

# huggingface_api_key = st.sidebar.text_input("Huggingface API Key", type="password")


# from langchain_openai import OpenAIEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.retrievers.self_query.base import SelfQueryRetriever
# if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
#   os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass()

# model_name = "deepseek-ai/deepseek-llm-7b-chat"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
# model.generation_config = GenerationConfig.from_pretrained(model_name)
# model.generation_config.pad_token_id = model.generation_config.eos_token_id

# messages = [
#     {"role": "user", "content": "Who are you?"}
# ]
# input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
# outputs = model.generate(input_tensor.to(model.device), max_new_tokens=100)

# result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
# print(result)

# llm = HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-beta",
#     task="text-generation",
#     max_new_tokens=512,
#     do_sample=False,
#     repetition_penalty=1.03,
# )

# from transformers import BitsAndBytesConfig

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype="float16",
#     bnb_4bit_use_double_quant=True,
# )

# llm = HuggingFacePipeline.from_model_id(
#     model_id="HuggingFaceH4/zephyr-7b-beta",
#     task="text-generation",
#     pipeline_kwargs=dict(
#         max_new_tokens=512,
#         do_sample=False,
#         repetition_penalty=1.03,
#         return_full_text=False,
#     ),
#     model_kwargs={"quantization_config": quantization_config},
# )

# hf = HuggingFaceEmbeddings(
#     model_name="HuggingFaceH4/zephyr-7b-beta"
#     # model_kwargs={"quantization_config": quantization_config}
# )

# chat_model = ChatHuggingFace(llm=llm)

# from langchain_core.messages import (
#     HumanMessage,
#     SystemMessage,
# )

# messages = [
#     SystemMessage(content="You're a helpful assistant"),
#     HumanMessage(
#         content="What happens when an unstoppable force meets an immovable object?"
#     ),
# ]

# ai_msg = chat_model.invoke(messages)
# print(ai_msg.content)

# with st.form("my_form"):
#     text = st.text_area(
#         "Enter text:",
#         "What are the three key pieces of advice for learning how to code?",
#     )
#     submitted = st.form_submit_button("Submit")
#     if not huggingface_api_key.startswith("hf_"):
#         st.warning("Please enter your Huggingface API key!", icon="⚠")
#     if submitted and huggingface_api_key.startswith("hf_"):
es_client = Elasticsearch(
    "https://unified-dev-monitorai.digit.org/",
    verify_certs=False,
    ssl_show_warn=False
)

vectorstore = ElasticsearchStore(
    index_name="property-application",
    es_connection=es_client,
    es_user = st.secrets["ES_USER"],
    es_password = st.secrets["ES_PASS"]
)

st.info(vectorstore.search("*.*"))

# print("STOP")

# metadata_field_info = [
#     AttributeInfo(
#         name="state",
#         description="The state where the school is located",
#         type="string",
#     ),
#     AttributeInfo(
#         name="psuedocode",
#         description="The unique identifier for the school across all tables, the primaray key",
#         type="integer",
#     ),
#     AttributeInfo(
#         name="district",
#         description="The district where the school is located",
#         type="string",
#     ),
#     AttributeInfo(
#         name="managment",
#         description="The management code for the school, that indicates whether it is public or private or other",
#         type="integer",
#     ),
# ]

# document_content_description = "Profile of a school"
# retriever = SelfQueryRetriever.from_llm(
#     llm, vectorstore, document_content_description, metadata_field_info, verbose=True
# )

# print(retriever.invoke("What are the districts for which we have school information?"))

# from langchain_openai import OpenAIEmbeddings

# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# from langchain_elasticsearch import ElasticsearchStore
# import ssl
# ssl_context = create_ssl_context(<use `cafile`, or `cadata` or `capath` to set your CA or CAs)
# context.check_hostname = False
# context.verify_mode = ssl.CERT_NONE

# elastic_vector_search = ElasticsearchStore(
#     es_url='^"https://unified-dev.digit.org/kibana/api/console/proxy?path=^%^2Fproperty-application^%^2F_mapping^&method=GET^" ^ -X ^"POST^" ^ -H ^"accept: */*^" ^ -H ^"accept-language: en-US,en;q=0.8^" ^ -H ^"content-length: 0^" ^ -H ^"content-type: application/json^" ^ -b ^"sid=Fe26.2**c261c3f5b9a9da8c84a30c4225fe44b97ae6298f6285c1e0c72c8769b5fa982f*23JXU1bz_QP6C05d3uU2iA*8BEBaMfUfv0lDO5e6Knjjvpp-FEblTyVOIB2fbbekEADimfSenAETMlRJYpcWfsDKgQUMNR5ybddedom2gIiQeOlu0KNdQgQgeOZ8Qyy3YrKDQxVq7u9uGUGKN6FsO7QC74utfKsx5qc_djQmObEL4TMlfpdouTu11PC86ki6mDCbgWeMbIJchkbUlgDfyGLcQdznwph5y6ZxTLihNpICcyOHkKwolAdKIBVA8756MAxD8D0VBJk8bka7cvax42XZhXbb8psFAtsW4t_0OklWA**a86dc1dc2a489fd67a8f5a84885e44f0baf59817b48f5b552811f50ef035bee3*Z8koWavnmE2B_uvNBWx1qLeuih_xPs0Us-dNRhYihZQ^" ^ -H ^"kbn-build-number: 68312^" ^ -H ^"kbn-version: 8.11.3^" ^ -H ^"origin: https://unified-dev.digit.org^" ^ -H ^"priority: u=1, i^" ^ -H ^"referer: https://unified-dev.digit.org/kibana/app/dev_tools^" ^ -H ^"sec-ch-ua: ^\^"Brave^\^";v=^\^"135^\^", ^\^"Not-A.Brand^\^";v=^\^"8^\^", ^\^"Chromium^\^";v=^\^"135^\^"^" ^ -H ^"sec-ch-ua-mobile: ?0^" ^ -H ^"sec-ch-ua-platform: ^\^"Windows^\^"^" ^ -H ^"sec-fetch-dest: empty^" ^ -H ^"sec-fetch-mode: cors^" ^ -H ^"sec-fetch-site: same-origin^" ^ -H ^"sec-gpc: 1^" ^ -H ^"user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36^" ^ -H ^"x-elastic-internal-origin: Kibana^" ^-H ^"x-kbn-context: ^%^7B^%^22type^%^22^%^3A^%^22application^%^22^%^2C^%^22name^%^22^%^3A^%^22dev_tools^%^22^%^2C^%^22url^%^22^%^3A^%^22^%^2Fkibana^%^2Fapp^%^2Fdev_tools^%^22^%^7D^"',
#     index_name="property-application",
#     # embedding=embeddings,
#     es_user="read_user",
#     es_password="egov@4321",
# )
# from datetime import datetime
# from elasticsearch import Elasticsearch

# # Connect to 'http://localhost:9200'
# client = Elasticsearch("https://unified-dev.digit.org/kibana/api/console/proxy")

# print(client.get(index="property-application", id=1))