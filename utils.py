from elasticsearch import Elasticsearch
import logging

logging.basicConfig(level=logging.INFO)

def get_unique_category_values(index_name, field, es_config):
    """
    Retrieve unique values from the field in a specified Elasticsearch index.
    Returns:
    list: A list of unique values from the 'category.keyword' field.
    """
    try:
        es = Elasticsearch(f'https://{es_config["host"]}:{es_config["port"]}', api_key=es_config["api_key"], request_timeout=300)

        agg_query = {
            "size": 0,
            "aggs": {
                "unique_categories": {
                    "terms": {"field": field, "size": 10000}
                }
            }
        }

        response = es.search(index=index_name, body=agg_query)
        unique_values = [bucket['key'] for bucket in response['aggregations']['unique_categories']['buckets']]

        return unique_values
    except Exception as e:
        logging.error(f"Error retrieving unique values from {field}: {e}")
        return []

def populate_default_values(index_name, es_config):
    category_values = get_unique_category_values(index_name, 'category.keyword', es_config)
    language_values = get_unique_category_values(index_name, 'language.keyword', es_config)
    country_values = get_unique_category_values(index_name, 'country.keyword', es_config)
    category_values.append("Any")
    language_values.append("Any")
    country_values.append("Any")

    return sorted(category_values), sorted(language_values), sorted(country_values)

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

# def compile_terms(term):
#     if "Any" in term:
#         result = None
#     else:
#         result = term
#     return result


def populate_terms(selected_items, field):
    """
    Creates a list of 'term' queries for Elasticsearch based on selected items.
    Returns:
        list: A list of 'term' queries for inclusion in an Elasticsearch 'should' clause.
    """
    if (selected_items is None) or ("Any" in selected_items):
        return []
    else:
        return [{"term": {field: item}} for item in selected_items]


def add_terms_condition(must_list, terms):
    if terms:
        must_list.append({
            "bool": {
                "should": terms,
                "minimum_should_match": 1
            }
        })


def create_must_term(category_terms, language_terms, country_terms, formatted_start_date, formatted_end_date):
    must_term = [
        {"range": {"date": {"gte": formatted_start_date, "lte": formatted_end_date}}}
    ]

    add_terms_condition(must_term, category_terms)
    add_terms_condition(must_term, language_terms)
    add_terms_condition(must_term, country_terms)

    return must_term


import pandas as pd


def create_dataframe_from_response(response):
    """
    Creates a pandas DataFrame from Elasticsearch response data.
    Returns:
        pd.DataFrame: A DataFrame containing the selected fields from the response.
    """
    try:
        selected_documents = []

        if 'hits' not in response or 'hits' not in response['hits']:
            print("No data found in the response.")
            return pd.DataFrame()  # Return an empty DataFrame

        for doc in response['hits']['hits']:
            selected_doc = {
                'date': doc['_source'].get('date', ''),
                'text': doc['_source'].get('text', ''),
                'translated_text': doc['_source'].get('translated_text', ''),
                'url': doc['_source'].get('url', ''),
                'country': doc['_source'].get('country', ''),
                'language': doc['_source'].get('language', ''),
                'category': doc['_source'].get('category', ''),
                'id': doc.get('_id', '')
            }
            selected_documents.append(selected_doc)

        df_selected_fields = pd.DataFrame(selected_documents)

        if 'date' in df_selected_fields.columns:
            df_selected_fields['date'] = pd.to_datetime(df_selected_fields['date']).dt.date

        return df_selected_fields

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()





