import os
import streamlit as st

from dotenv import load_dotenv
from langchain import hub
from langchain import hub
from elasticsearch import Elasticsearch
from elasticsearch import BadRequestError
from elasticsearch.exceptions import NotFoundError
from angle_emb import AnglE, Prompts
from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from authentificate import check_password