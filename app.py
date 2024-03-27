import os
import streamlit as st

from langchain import hub
from elasticsearch import Elasticsearch
from elasticsearch import BadRequestError
from elasticsearch.exceptions import NotFoundError
from angle_emb import AnglE, Prompts
from langchain_community.chat_models import ChatOpenAI
from authentificate import check_password

# Init Langchain and Langsmith services
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"rag_app : summarization : production"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = st.secrets['ld_rag']['LANGCHAIN_API_KEY']
os.environ["LANGSMITH_ACC"] = st.secrets['ld_rag']['LANGSMITH_ACC']
url = f'{os.environ["LANGSMITH_ACC"]}/simple-rag'
prompt_template = hub.pull(url)

# Init openai model
OPENAI_API_KEY = st.secrets['ld_rag']['OPENAI_KEY_ORG']
llm_chat = ChatOpenAI(temperature=0.0, openai_api_key=OPENAI_API_KEY,
             model_name='gpt-4-turbo-preview')

# Load Elastic creds
elastic_host = st.secrets['ld_rag']['ELASTIC_HOST']
elastic_port = st.secrets['ld_rag']['ELASTIC_PORT']
api_key = st.secrets['ld_rag']['ELASTIC_API']

# create prompt vector
input_question = None
st.markdown('### Please enter your question:')
input_question = st.text_input("Enter your question here (phrased as if you ask a human)")


with st.popover("Tap to define filters"):
    st.markdown("Hello World 👋")
    options = st.multiselect('options available', ['option1', 'option1'], default=None, format_func=None, key=None, help=None,
                   on_change=None, args=None, kwargs=None, placeholder="Choose an option",
                   disabled=False, label_visibility="visible")
    name = st.text_input("What's your name?")

st.write("Your options:", options)

if input_question:

    formatted_start_date, formatted_end_date = None, None
    @st.cache(allow_output_mutation=True,
              hash_funcs={"_thread.RLock": lambda _: None, "builtins.weakref": lambda _: None})
    def load_model():
        angle_model = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1',
                                            pooling_strategy='cls')
        angle_model.set_prompt(Prompts.C)
        return angle_model

    # Create question embedding
    angle = load_model()
    vec = angle.encode({'text': input_question}, to_numpy=True)
    question_vector = vec.tolist()[0]

    # Get input dates
    selected_start_date = st.date_input("Select start date:")
    formatted_start_date = selected_start_date.strftime("%Y-%m-%d")
    st.write("You selected start date:", selected_start_date)
    selected_end_date = st.date_input("Select end date:")
    formatted_end_date = selected_end_date.strftime("%Y-%m-%d")
    st.write("You selected end date:", selected_end_date)


    if formatted_start_date and formatted_end_date:

        # Get input index
        index_options = [
        # 'detector-media-tiktok',
        'ua-by-facebook',
        'ua-by-telegram',
        'ua-by-web',
        'ua-by-youtube',
        'dm-8-countries-twitter',
        'dm-8-countries-telegram',
        'ndi-lithuania-instagram',
        'ndi-lithuania-web',
        'ndi-lithuania-youtube',
        'ndi-lithuania-telegram',
        'ndi-lithuania-initial-kivu-twitter',
        'recovery-win-facebook',
        'recovery-win-telegram',
        'recovery-win-web',
        'recovery-win-twitter',
        'recovery-win-comments-telegram']
        selected_index = st.selectbox('Please choose index', index_options, key='index')
        st.write(f"We'll search the answer in index: {selected_index}")

        # Authorise user
        if not check_password():
            st.stop()

        # Run search
        if st.button('RUN SEARCH'):
            try:
                texts_list = []
                st.write(f'Running search for question: {input_question}')
                try:
                    es = Elasticsearch(f'https://{elastic_host}:{elastic_port}', api_key=api_key, request_timeout=300)
                except Exception as e:
                    st.error(f'Failed to connect to Elasticsearch: {str(e)}')

                response = es.search(index=selected_index,
                                     size=50,
                                     knn={"field": "embeddings.WhereIsAI/UAE-Large-V1",
                                          "query_vector":  question_vector,
                                          "k": 20,
                                          "num_candidates": 10000,
                                          "filter": {"range": {"date": {"gte": formatted_start_date,  "lte": formatted_end_date}}}})
                for doc in response['hits']['hits']:
                    # texts_list.append(doc['_source']['translated_text'])
                    texts_list.append((doc['_source']['translated_text'], doc['_source']['url']))

                st.write('Searching for documents, please wait...')

                # formatting urls so they work properly within streamlit
                corrected_texts_list = [(text, 'https://' + url if not url.startswith('http://') and not url.startswith(
                    'https://') else url) for text, url in texts_list]

                # Get summary for the retrieved data
                customer_messages = prompt_template.format_messages(
                    question=input_question,
                    texts=corrected_texts_list)
                resp = llm_chat.invoke(customer_messages)

                # Print GPT summary
                st.markdown('### This is the GPT summary for the question:')
                st.markdown(resp.content)
                st.write('******************')
                st.markdown('### These are the texts retrieved by search:')

                # Print source texts
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