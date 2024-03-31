## Import packages
import streamlit as st
import asyncio
import os

# import models - using gpt-3.5-turbo
from langchain.chat_models import ChatOpenAI

# prompt template
from langchain.prompts import PromptTemplate
from langchain.callbacks import StreamlitCallbackHandler

## load Chains to pipe
from langchain.chains import (
  LLMChain, 
  SimpleSequentialChain, 
  ConversationalRetrievalChain, 
  ConstitutionalChain, 
  ConversationChain
)

# load methods/functions related to memory/history
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.schema import SystemMessage
from langchain.prompts import MessagesPlaceholder

# Load agents
from langchain.agents import (
  load_tools, 
  initialize_agent, 
  AgentType, Agent
)

## Initialize start VARs
stop_session = False
site_title = "Zilliar"
ZILLIAR = "images/zilliar.jpg"

st.set_page_config(page_title=site_title)
st.title(f"{site_title}")
# -

## Async function which implements await
async def generate_response(user_input):
    response = await agent_chain.arun(user_input)
    # Handle results as they are completed
    return response

# +
## check for OpenAI key
prompt_placeholder = "Please enter your OPENAI Access Token"
promt_success = "Thank you for providing OPENAI Access Key"
input_msg = "Hello! How can I be of assistance ::writing_hand::"

with st.sidebar:
  st.image(ZILLIAR)
  st.title("Welcome :balloon:")
  if "OPENAI_API_KEY" in st.secrets:
    st.success(f"{promt_success}")
    openai_api_key = st.secrets["OPENAI_API_KEY"]
  else:
    openai_api_key = st.text_input(label=prompt_placeholder,             # must enter a string - cannot be empty
                                   label_visibility="hidden",            # to hide label
                                   placeholder=prompt_placeholder, 
                                   type="password")
    if not openai_api_key:
      st.warning(f"{prompt_placeholder}")
      stop_session = True
    else:
      st.success(f"{promt_success}")
      
  st.markdown("""
  APP uses: **gpt-3.5-turbo**.
  \nPlease use subject keywords to narrow summary search - get good feedback from OpenAI GPT model.
  \nIf you do not have a [OPENAI ACCESS TOKEN](https://platform.openai.com/account/api-keys) please sign-up\n
  
  \nIn your question include prompts - for example:
  - give me a summary on ....
  - list your reasoning in bullet points....
  """)

##check if key is correct - else system throws an error
if not openai_api_key:
  pass
else:
  ## post welcome greeting
  print(input_msg)

  ## initialize langchain setup
  st_msgs = StreamlitChatMessageHistory(key="langchain_messages")
  memory = ConversationBufferMemory(chat_memory=st_msgs, return_messages=True)

  os.environ["OPENAI_API_KEY"] = openai_api_key
  llm = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo", api_key=openai_api_key, streaming=True)

  agent_msg = SystemMessage(content="""
  Guidelines: Gather inormation on topic in language aked by User.
  If language == german, completion should be in German language
  if french then completion should be in French language.
  if language requested by User is english then completions should be English language
  Translate [topic completion] to [target language] in layman's terms before response to User.
  if you cannot find any information about the question asked or if the information 
  gathered is completed, apologize to User and ask if there is another question the User needs help with.
  - Response must adhere to HHH: Helpful, Harmless, honest
  - Response should be concise, do not repeat words or similarities in summary
  - Make every word count
  """)

  topic = st.text_input('Input your question ...')
  language = 'English'
  prompts = PromptTemplate(
    input_variables = [topic],
    template = "generate a brief summary on {topic} and respond back in {language} language"
  )
  
  # 'wolfram-alpha', - this needs a license
  # tools = load_tools(['wikipedia', 'ddg-search', 'arxiv'], llm = llm)
  tools = load_tools(['wikipedia'], llm = llm)

  agent_kwargs = {
      "system_message": agent_msg,
      "extra_prompt_messages": [MessagesPlaceholder(variable_name="history")]
  }

  # agent chain setup
  agent_chain = initialize_agent(tools = tools, 
                                 llm=llm,
                                 agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                 agent_kwargs=agent_kwargs,
                                 verbose=True,
                                 memory=memory
                          )

  ## write to memory
  for msg in st_msgs.messages:
    st.chat_message(msg.type).write(msg.content)
    
  if topic:
    with st.spinner('waiting on ChatGPT response'):
      response = asyncio.run(generate_response(topic))
      st.chat_message("ai").write(response)

##
# if __name__ == "__main__":
#     main()
