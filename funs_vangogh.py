import os
from pinecone import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_pinecone import Pinecone
from langchain.storage import LocalFileStore
from langchain import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import format_document
from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_community.embeddings.sentence_transformer import (
SentenceTransformerEmbeddings,)
import getpass
import pandas as pd


def embed_file(file_path, index_name="vangogh"):
    with open(file_path, "rb") as file:  # Ensure the file is opened properly
        file_content = file.read()
        file_path = f"./.cache/files/{file_path}"  # Adjusted to use file_path for naming
    with open(file_path, "wb") as f:
        f.write(file_content)

    index_name = index_name
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file_path}")
    loader = UnstructuredFileLoader(file_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=200, chunk_overlap=100, separator="\n")
    docs = loader.load_and_split(text_splitter=splitter)
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstores = Pinecone.from_documents(docs, embedding_function, index_name=index_name)
    retriever = vectorstores.as_retriever()
    return retriever

retriever = embed_file("./vangogh_collection.xlsx")

DEFAULT_DOCUMENT_PROMPT= PromptTemplate.from_template(template="{page_content}")

# Arching docs to one doc
def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

# Config for LLM
os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter Your Pinecone API KEY : ")
pinecone_api_key = os.environ["PINECONE_API_KEY"] 
os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter Your OPENAI API KEY : ")
openai_api_key = os.environ["OPENAI_API_KEY"] 


# Select LLM 
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model_name='gpt-4o')


def invoke(formatted_prompt):
    parser = StrOutputParser()
    result = llm.invoke(formatted_prompt)
    result=parser.invoke(result)
    return result


memories = []

def save(question, answer):
    chat_memory = {
        "User": question,
        "AI": answer
    }
    memories.append(chat_memory)


def reset_memory():
    return memories.clear()


def final_prompt(
    authors="Vincent van Gogh, Johann Wolfgang von Goethe, Charles Dickens, Leo Tolstoy",
    authors_tone_description="The tone qualities of the examples above can be described as follows:\n\n1. **Pace**: Moderate - The narrative progresses at a steady, unhurried pace, allowing for detailed descriptions and reflections.\n2. **Mood**: Reflective and warm - The emotional atmosphere is thoughtful and affectionate, with a sense of longing and appreciation for the recipient.\n3. **Tone**: Affectionate and sincere - The author's attitude is caring and genuine, showing a deep concern for the recipient's well-being and experiences.\n4. **Voice**: Personal and intimate - The writing style is conversational and familiar, reflecting a close relationship between the author and the recipient.\n5. **Tension**: Low - There is little sense of suspense or conflict; the narrative is calm and composed.\n6. **Imagery**: Vivid and descriptive - The use of detailed descriptions creates clear mental images of the author's surroundings and experiences.\n7. **Formality**: Semi-formal - The writing adheres to traditional conventions but maintains a personal and approachable tone.\n8. **Perspective**: First-person - The story is told from the author's point of view, providing a direct and personal account of their experiences.\n9. **Rhythm**: Smooth and flowing - The writing has a natural cadence, with well-structured sentences and paragraphs.\n10. **Emotion**: Affectionate and nostalgic - The writing evokes feelings of warmth, longing, and appreciation.\n11. **Clarity**: High - The writing is clear and easy to understand, with well-organized thoughts and descriptions.\n12. **Conciseness**: Moderate - The writing is detailed and elaborative, but not overly verbose.\n13. **Descriptiveness**: High - The level of detail and elaboration is significant, providing a rich and immersive reading experience.\n14. **Humor**: Minimal - There is little to no presence of comedic elements; the tone remains earnest and sincere.\n15. **Seriousness**: Moderate - The writing is earnest and thoughtful, but not overly grave or solemn.\n16. **Form**: Letter format - The structure is organized as a personal letter, with a clear beginning, body, and closing.\n17. **Dialogue**: Minimal - The writing primarily consists of the author's reflections and descriptions, with little to no direct conversation.\n18. **Symbolism**: Minimal - The writing focuses more on direct descriptions and personal reflections rather than symbolic representations.\n19. **Irony**: Minimal - The use of language is straightforward and sincere, with little to no ironic undertones.\n20. **Theme**: Connection and reflection - The central topics revolve around maintaining personal connections, sharing experiences, and reflecting on one's surroundings and circumstances.",
    users_sentence='''My dear Theo,
You’re probably longing to hear from me,1 so I don’t want to keep you waiting for a letter any longer.
I heard from home that you’re now staying with Mr Schmidt, and that Pa has been to see you. I sincerely hope that this will be more to your liking than your previous boarding-house, and don’t doubt that it will be.2 Write to me soon, I’m longing to hear from you, and tell me how you’re  1v:2 spending your days at present, &c. Write to me especially about the paintings you’ve seen recently, and also whether anything new has been published in the way of etchings or lithographs. You must keep me well informed about this, because here I don’t see much in that genre, as the firm here is just a stockroom.3
I’m very well, considering the circumstances.
I’ve come by a boarding-house that suits me very well for the present.4 There are also three Germans in the house who really love music and play piano and sing themselves, which makes the evenings  1v:3 very pleasant indeed. I’m not as busy here as I was in The Hague, as I only have to be in the office from 9 in the morning until 6 in the evening, and on Saturdays I’m finished by 4 o’clock. I live in one of the suburbs of London, where it’s comparatively quiet. It’s a bit like Tilburg5 or some such place.
I spent some very pleasant days in Paris and, as you can imagine, very much enjoyed all the beautiful things I saw at the exhibition6 and in the Louvre and the Luxembourg.7 The Paris branch is splendid, and much larger than I’d imagined. Especially the Place de l’Opéra.8
Life here is very expensive. I pay  18 shillings a week for my lodgings, not including the washing, and then I still have to eat in town.9
Last Sunday I went on an outing with Mr Obach, my superior,10 to Box Hill, which is a high hill (some 6 hours from L.),11 partly of chalk and covered with box trees, and on one side a wood of tall oak trees. The countryside here is magnificent, completely different from Holland or Belgium. Everywhere one sees splendid parks with tall trees and shrubs, where one is allowed to walk. During the Whitsun holiday12 I also took a nice trip with those Germans, but those gentlemen spend a great deal of money and I shan’t go out with them any more.
I was glad to hear from Pa that Uncle H. is reasonably well. Would you give my warm regards to him and Aunt13 and give them news of me? Bid good-day to Mr Schmidt and Eduard from me,14 and write to me soon. Adieu, I wish you well.
''',
    retriever=retriever,
    memories=memories,
    question="",
    ):

    template = """
    `% INSTRUCTIONS
    - You are an AI Bot that is very good at mimicking an author writing style.
    - Your goal is to answer the following question and context with the tone that is described below.
    - Do not go outside the tone instructions below
    - Respond in ONLY KOREAN 
    - Check chat history first and answer 
    - You must say you are "반 고흐" IF you are told 'who you are?'
    - Never use emoji and Special characters 
    - Speak ONLY informally

    % Mimic These Authors:
    {authors}

    % Description of the authors tone:
    {tone}

    % Authors writing samples
    {example_text}
    % End of authors writing samples

    % Context
    {context}

    % Question
    {question}

    % YOUR TASK
    1st - Write out topics that this author may talk about
    2nd - Answer with a concise passage (under 300 characters) as if you were the author described above 
    """

    method_4_prompt_template = PromptTemplate(
        input_variables=["authors", "tone", "example_text", "question", "history", "context", "example_answer"],
        template=template,
    )                   
    formatted_prompt = method_4_prompt_template.format(authors=authors,
                                               tone=authors_tone_description,
                                               example_text=users_sentence,
                                               question=question,
                                               context=_combine_documents(retriever.get_relevant_documents(question)),
                                                )
    return formatted_prompt

# TODO : Preprocessing Code
def extract_answer(data):
    # 데이터를 줄바꿈 기준으로 분할하여 리스트로 저장
    sentences = data.split("\n")
    
    # 마지막 문장을 반환
    if sentences:
        return sentences[-1].strip()
    else:
        return "텍스트를 찾을 수 없습니다."
    

def run(question):
    result = invoke(final_prompt(question=question))
    save(question, extract_answer(result))
    return memories[-1]['AI']


