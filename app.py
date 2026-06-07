import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

## Streamlit Page Config
st.set_page_config(
    page_title="Text To Math Problem Solver And Data Search Assistant",
    page_icon="🧮"
)

st.title("🧮 Text To Math Problem Solver Using Llama 3")

## Sidebar API Key
groq_api_key = st.sidebar.text_input(
    label="Enter Groq API Key",
    type="password"
)

if not groq_api_key:
    st.info("Please add your Groq API key to continue.")
    st.stop()

## Load LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key
)

## Wikipedia Tool
wikipedia_wrapper = WikipediaAPIWrapper()

wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="Useful for searching general information from Wikipedia."
)

## Prompt Template
prompt = """
You are a helpful AI assistant.

Solve mathematical and reasoning questions step-by-step.

Give clean and detailed explanations.

Question:
{question}

Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

## Reasoning Chain
chain = LLMChain(
    llm=llm,
    prompt=prompt_template
)

## Reasoning Tool
reasoning_tool = Tool(
    name="Reasoning Tool",
    func=chain.run,
    description="Useful for solving math, logic, and reasoning problems."
)

## Initialize Agent
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, reasoning_tool],   # FIX: removed broken calculator tool
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

## Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi 👋 I'm your Math & Reasoning ChatBot."
        }
    ]

## Display Chat Messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

## User Input
question = st.text_area(
    "Enter your question:",
    "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?"
)

## Generate Response
if st.button("Find My Answer"):

    if question.strip():

        ## Show User Message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        st.chat_message("user").write(question)

        with st.spinner("Generating response..."):

            try:

                st_cb = StreamlitCallbackHandler(
                    st.container(),
                    expand_new_thoughts=False
                )

                ## Generate Response
                response = assistant_agent.run(
                    question,
                    callbacks=[st_cb]
                )

                response = str(response)

                ## Save Assistant Message
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response
                    }
                )

                ## Display Response
                st.write("### Response")
                st.success(response)

            except Exception as e:

                st.error(f"Error: {str(e)}")

    else:
        st.warning("Please enter a question.")


## groq key = "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"