import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Predefined API keys for both LLMs
GOOGLE_API_KEYS = ["AIzaSyBWEZ3366hOs5E0gdWtgQs6fb-QfVg37sc", "AIzaSyDeTi3OoiHrVTLURRengKDAtbU9fDAtc98"]
GROQ_API_KEYS = ["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", "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"]

# Streamlit app title
st.title("SmartPrompt: Effective Prompt Engineering")

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Input prompt from the user
user_prompt = st.text_area("Enter your prompt here:")

prompt_template = """
Act as a professional prompt engineer.
Rephrase the user's input prompt according to the following guidelines. Identify missing information that might be critical for providing an accurate and comprehensive response and ask the user to specify them and additional details if needed, such as the goal they want to achieve, or if there are implicit assumptions in the input that need clarification. Ensure that the rephrased prompt is clearer, more actionable, and achieves the desired task effectively based on the following criteria:

Clarity & Specificity: Ensure the task is clear, unambiguous, and uses precise language.

Context Enhancement: Add relevant background or context if it's missing or unclear in the original input.

Expected Output: Specify the desired output format (e.g., paragraph, list, table), tone, style, or length as necessary and make sure the prompt indicates if the response should be formal, casual, persuasive, technical, or another tone.

Task Breakdown: For complex tasks, break the instructions into manageable steps for ease of understanding and execution, and provide logical sequencing if multiple parts need to be completed.

Examples for Guidance: Include relevant examples, if necessary, to guide the LLM in interpreting the prompt.

Tone & Style Alignment: If the tone or style isn’t specified, set a suitable one (e.g., formal, casual, persuasive) based on the content and intent of the task.

Constraints & Limitations: Apply any relevant constraints such as word count, time limits, vocabulary choices, or format requirements.

Use delimiters to clearly indicate distinct parts of the input.

Option to remove hallucination or reduce by mentioning, if you do not know, say “I don’t Know”.

Add where to focus – key areas to focus on should be specified by the user.

Mention the role for the LLM: Consider yourself as a professional [specific role] (e.g., lawyer, engineer, consultant) while generating the response.
User Prompt: ```{user_prompt}```

"""

# Function to call the Gemini model using langchain_google_genai
def call_gemini_model(api_keys, user_prompt):
    for api_key in api_keys:
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key)
            prompt = PromptTemplate(input_variables=["user_prompt"], template=prompt_template)
            chain = prompt | llm | StrOutputParser()
            response = chain.invoke(user_prompt)
            return response if isinstance(response, str) else str(response)
        except Exception as e:
            continue  # Try the next API key if one fails
    return "An error occurred with all Google API keys."

# Function to call the Groq model using langchain_groq
def call_groq_model(api_keys, user_prompt):
    for api_key in api_keys:
        try:
            llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key)
            prompt = PromptTemplate(input_variables=["user_prompt"], template=prompt_template)
            chain = prompt | llm | StrOutputParser()
            response = chain.invoke(user_prompt)
            return response if isinstance(response, str) else str(response)
        except Exception as e:
            continue  # Try the next API key if one fails
    return "An error occurred with all Groq API keys."

# Function to sequentially choose between Gemini and Groq models
def choose_llm_model(user_prompt):
    refined_prompt = call_gemini_model(GOOGLE_API_KEYS, user_prompt)
    if "An error occurred" in refined_prompt:
        refined_prompt = call_groq_model(GROQ_API_KEYS, user_prompt)
    return refined_prompt

# Display conversation history
if st.session_state.conversation_history:
    st.write("**Conversation History:**")
    for i, (user_input, bot_response) in enumerate(st.session_state.conversation_history):
        st.write(f"**User {i+1}:** {user_input}")
        st.write(f"**Bot {i+1}:** {bot_response}")

# Button to trigger the prompt engineering
if st.button("Generate Refined Prompt"):
    if not user_prompt:
        st.error("Please enter a prompt.")
    else:
        with st.spinner("Refining your prompt..."):
            refined_prompt = choose_llm_model(user_prompt)
            st.session_state.conversation_history.append((user_prompt, refined_prompt))
            st.write(f"**Refined Prompt:** {refined_prompt}")

            # Option to download the refined prompt as a text file
            st.download_button(
                label="Download Refined Prompt",
                data=refined_prompt,
                file_name="refined_prompt.txt",
                mime="text/plain"
            )
