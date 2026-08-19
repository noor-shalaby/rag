import streamlit as st
import requests

# Set page configuration
st.set_page_config(
    page_title="Clinical RAG Assistant",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Clinical RAG Assistant")
st.markdown("Ask clinical questions and get evidence-based answers backed by your medical database.")

# Define your backend URL
# (Use 'http://localhost:8000/ask' for local testing, or your deployed FastAPI Cloud URL in production)
BACKEND_URL = "http://localhost:8000/ask"

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat bubbles
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input from chat box
if query := st.chat_input("Type your medical question here..."):
    # Add user message to state and display it
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Call your FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Analyzing medical literature..."):
            try:
                # Send POST request matching your FastAPI Pydantic schema
                response = requests.post(BACKEND_URL, json={"query": query}, timeout=60)

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer returned.")
                    st.markdown(answer)
                    # Save assistant response to state
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    error_detail = response.json().get("detail", "Unknown server error")
                    err_msg = f"⚠️ **Server Error:** {error_detail}"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})

            except requests.exceptions.ConnectionError:
                err_msg = "⚠️ **Connection Error:** Could not connect to the FastAPI backend. Make sure your backend server is running!"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
            except Exception as e:
                err_msg = f"⚠️ **Error:** {str(e)}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
