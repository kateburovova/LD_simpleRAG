import os
import streamlit as st

from langchain import hub
from elasticsearch import Elasticsearch
from elasticsearch import BadRequestError
from elasticsearch.exceptions import NotFoundError
from angle_emb import AnglE, Prompts
from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from authentificate import check_password

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"Tracing Streamlit RAG ES app"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = st.secrets['ld_rag']['LANGCHAIN_API_KEY']
os.environ["LANGSMITH_ACC"] = st.secrets['ld_rag']['LANGSMITH_ACC']

url = f'{os.environ["LANGSMITH_ACC"]}/simple-rag'
prompt_template = hub.pull(url)

if not check_password():
    st.stop()

OPENAI_API_KEY = st.secrets['ld_rag']['OPENAI_KEY_ORG']
llm_chat = ChatOpenAI(temperature=0.0, openai_api_key=OPENAI_API_KEY,
             model_name='gpt-4-turbo-preview')

elastic_host = st.secrets['ld_rag']['ELASTIC_HOST']
elastic_port = st.secrets['ld_rag']['ELASTIC_PORT']
api_key = st.secrets['ld_rag']['ELASTIC_API']

# create prompt vector
input_question = None
input_question = st.text_input("Ask your question")

if input_question:
    @st.cache(allow_output_mutation=True,
              hash_funcs={"_thread.RLock": lambda _: None, "builtins.weakref": lambda _: None})
    def load_model():
        angle_model = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1',
                                            pooling_strategy='cls')
        angle_model.set_prompt(Prompts.C)
        return angle_model

    angle = load_model()
    # angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
    # angle.set_prompt(prompt=Prompts.C)
    vec = angle.encode({'text': input_question}, to_numpy=True)
    question_vector = vec.tolist()[0]

    formatted_start_date, formatted_end_date = None, None

    # get input date
    selected_start_date = st.date_input("Select start date:")
    formatted_start_date = selected_start_date.strftime("%Y-%m-%d")

    st.write("You selected start date:", selected_start_date)

    selected_end_date = st.date_input("Select end date:")
    formatted_end_date = selected_end_date.strftime("%Y-%m-%d")
    st.write("You selected end date:", selected_end_date)


    if formatted_start_date and formatted_end_date:

        # get input index
        index_options = ['ua_by_facebook', 'ua_by_telegram', 'ua_by_web', 'ua_by_youtube', 'dm_8_countries_twitter', 'dm_8_countries_telegram']
        selected_index = st.selectbox('Please choose index', index_options, key='index')
        st.write(f"We'll search the answer in index: {selected_index}")

        # run search
        if st.button('RUN SEARCH'):
            try:
                texts_list = []
                st.write(f'Running search for question: {input_question}')
                # es = Elasticsearch(f'https://{elastic_host}:{elastic_port}', api_key=api_key)
                try:
                    es = Elasticsearch(f'https://{elastic_host}:{elastic_port}', api_key=api_key)
                except Exception as e:
                    st.error(f'Failed to connect to Elasticsearch: {str(e)}')

                response = es.search(index=selected_index,
                                     knn={"field": "embeddings.WhereIsAI/UAE-Large-V1",
                                          "query_vector":  question_vector,
                                          "k": 20,
                                          "num_candidates": 1000,
                                          "filter": {"range": {"date": {"gte": formatted_start_date,  "lte": formatted_end_date}}}})
                for doc in response['hits']['hits']:
                    texts_list.append(doc['_source']['translated_text'])

                st.write('Searching for documents, please wait...')

                customer_messages = prompt_template.format_messages(
                    question=input_question,
                    texts=texts_list)
                resp = llm_chat.invoke(customer_messages)

                st.markdown('### This is the GPT summary for the question:')

                st.write(resp.content)

                st.write('******************')

                st.markdown('### These are the texts retrieved by search:')

                for doc in response['hits']['hits']:
                    st.write(doc['_source']['translated_text'])
                    st.write()
                    st.write(doc['_score'])
                    st.write('******************')

            except BadRequestError as e:
                st.error(f'Failed to execute search (embeddings might be missing for this index): {e.info}')
            except NotFoundError as e:
                st.error(f'Index not found: {e.info}')
            except Exception as e:
                st.error(f'An unknown error occurred: {str(e)}')