import streamlit as st
from config import QDRANT_DB_API_KEY, QDRANT_URL, OPEN_AI_API_KEY
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from openai import OpenAI
import ast


# Connection for semantic queries
qdrant_client = QdrantClient(
    url=QDRANT_URL, 
    api_key=QDRANT_DB_API_KEY,
)


# Connection to OpenAI API
client = OpenAI(
  organization='org-Cl2hWJbfjhnqQWMc1MTRzyqy',
  api_key=OPEN_AI_API_KEY
)

# Load encoder
@st.cache_data
def load_encoder():
    encoder = SentenceTransformer('all-MiniLm-L6-v2')
    return encoder



def make_query(prompt):
    if prompt != None:
        hits = qdrant_client.search(
        collection_name="my_hotels",
        query_vector=encoder.encode(prompt).tolist(),
        limit=4
        )
        return hits
    else:
        return None

@st.cache_data
def generate_response(prompt, _db_response):
    conversation = []
    new_prompt = f'Give a JSON with fields -  "hotel_name" and "reason" (which states in 2 sentences why the hotel matches prompt) for each unique hotel  that satisfies the user prompt- "{prompt}" using information below .\n"""'
    for hotel in _db_response:
        new_prompt = new_prompt + '\n#' + str(hotel.payload) + " hotel id: "
    new_prompt += '"""'
    conversation.append({"role":"system", "content":"You are a hotel advisor."})
    conversation.append({"role": "user", "content":new_prompt})
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=conversation
    )
    st.session_state.messages = []
    return response

if "messages" not in st.session_state:
    st.session_state.messages = []

encoder = load_encoder()
st.title("Hotel Recommendation Chatbot")
prompt = st.text_input('Enter prompt: ')
try:
    if prompt != None and len(prompt) > 1:
        db_response = make_query(prompt)
        response = generate_response(prompt, db_response)
        response_hotels = ast.literal_eval(response.choices[0].message.content)
        db_hotels = {}

        for hotel in db_response:
            name = hotel.payload['hotel_name']
            if name not in db_hotels:
                db_hotels[name] = hotel.payload

        completed = set()
        #print(response_hotels)
        usr_input = ""
        for hotel in response_hotels['hotels']:
            element = db_hotels[hotel['hotel_name']]
            if element['hotel_name'] in completed:
                continue
            else:
                usr_input += str(element) + '\n'
                completed.add(element['hotel_name'])
            st.subheader(hotel['hotel_name'])
            st.write(f"Location: {element['locality']}")
            st.image(element['hotel_image'])
            st.write(hotel['reason'])
            st.link_button("Visit hotel", element["hotel_url"])
            st.session_state.messages.append({"role": "user", "content": usr_input})
except:
    st.write("Error: Please try again")

#for message in st.session_state.messages:
    #with st.chat_message(message["role"]):
        #st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    print(st.session_state.messages)
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature= 0.5,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
