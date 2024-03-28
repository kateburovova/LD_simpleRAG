import os
import streamlit as st
import plotly.express as px
import streamlit.components.v1 as components

from langchain import hub
from elasticsearch import Elasticsearch
from elasticsearch import BadRequestError
from elasticsearch.exceptions import NotFoundError
from angle_emb import AnglE, Prompts
from langchain_openai import ChatOpenAI
from authentificate import check_password
from utils import (display_distribution_charts,populate_default_values, project_indexes,
                   populate_terms,create_must_term, create_dataframe_from_response,flat_index_list)


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

es_config = {
    'host': st.secrets['ld_rag']['ELASTIC_HOST'],
    'port': st.secrets['ld_rag']['ELASTIC_PORT'],
    'api_key': st.secrets['ld_rag']['ELASTIC_API']
}

########## APP start ###########
st.set_page_config(layout="wide")

# Get input index
selected_index = None
search_option = st.radio("Choose your search option", ['Specific Indexes', 'All Project Indexes'])

if search_option == 'Specific Indexes':
    selected_indexes = st.multiselect('Please choose one or more indexes', flat_index_list, default=None, placeholder="Select one or more indexes")
    if selected_indexes:
        selected_index = ",".join(selected_indexes)
        st.write(f"We'll search the answer in indexes: {', '.join(selected_indexes)}")
    else:
        selected_index = None
else:
    project_choice = st.selectbox('Please choose a project', list(project_indexes.keys()), index=None, placeholder="Select project")
    if project_choice:
        selected_indexes = project_indexes[project_choice]
        selected_index = ",".join(selected_indexes)
        st.write(f"We'll search the answer across all indexes for project: {project_choice}")

if selected_index:
    category_values, language_values, country_values = populate_default_values(selected_index, es_config)

    with st.popover("Tap to define filters"):
        st.markdown("Hihi 👋")
        st.markdown("If Any remains selected or no values at all, filtering will not be applied to this field.")
        st.markdown("Start typing to find the option faster.")
        categories_selected = st.multiselect('Select "Any" or choose one or more categories', category_values, default=['Any'])
        languages_selected = st.multiselect('Select "Any" or choose one or more languages', language_values, default=['Any'])
        countries_selected = st.multiselect('Select "Any" or choose one or more countries', country_values, default=['Any'])

    category_terms = populate_terms(categories_selected, 'category.keyword')
    language_terms = populate_terms(languages_selected, 'language.keyword')
    country_terms = populate_terms(countries_selected, 'country.keyword')

# create prompt vector
input_question = None
st.markdown('### Please enter your question:')
input_question = st.text_input("Enter your question here (phrased as if you ask a human)")


if input_question:

    formatted_start_date, formatted_end_date = None, None
    @st.cache_resource(hash_funcs={"_thread.RLock": lambda _: None, "builtins.weakref": lambda _: None})
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
    must_term = create_must_term(category_terms,
                                 language_terms,
                                 country_terms,
                                 formatted_start_date=formatted_start_date,
                                 formatted_end_date=formatted_end_date)


    if formatted_start_date and formatted_end_date:

        # Authorise user
        if not check_password():
            st.stop()

        # Run search
        if st.button('RUN SEARCH'):
            try:
                texts_list = []
                st.write(f'Running search for question: {input_question}')
                try:
                    es = Elasticsearch(f'https://{es_config["host"]}:{es_config["port"]}', api_key=es_config["api_key"], request_timeout=600)
                except Exception as e:
                    st.error(f'Failed to connect to Elasticsearch: {str(e)}')

                response = es.search(index=selected_index,
                                     size=30,
                                     knn={"field": "embeddings.WhereIsAI/UAE-Large-V1",
                                          "query_vector":  question_vector,
                                          "k": 20,
                                          "num_candidates": 10000,
                                          "filter": {
                                              "bool": {
                                                  "must": must_term
                                              }
                                          }
                                          }
                                     )

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
                df = create_dataframe_from_response(response)
                st.dataframe(df)

                display_distribution_charts(df)

                # tally_form_url = 'https://tally.so/embed/wzq1Aa?alignLeft=1&hideTitle=1&transparentBackground=1&dynamicHeight=1'
                # components.iframe(tally_form_url, width=700, height=500, scrolling=True)
                prefilled_url = "https://smith.langchain.com/o/2e725250-e617-4fde-9e57-409055af96f8/projects/p/4d4a23f4-d0e4-4c73-ae2a-76dfb006b670?&peek=1918e5a1-4b5c-4f33-8985-04f9c75b4584"
                prefilled_time = 10
                tally_form_url = f'https://tally.so/embed/wzq1Aa?alignLeft=1&hideTitle=1&transparentBackground=1&dynamicHeight=1&run_id={prefilled_url}&time={prefilled_time}'
                components.iframe(tally_form_url, width=700, height=500, scrolling=True)

            except BadRequestError as e:
                st.error(f'Failed to execute search (embeddings might be missing for this index): {e.info}')
            except NotFoundError as e:
                st.error(f'Index not found: {e.info}')
            except Exception as e:
                st.error(f'An unknown error occurred: {str(e)}')