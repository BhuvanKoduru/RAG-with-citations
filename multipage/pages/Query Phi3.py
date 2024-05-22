import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma

from streamlit_pdf_viewer import pdf_viewer
def view_pdf(file,page):
    st.header(file.split("\\")[-1]+ " Page number "+str(page+1))
    pdf_viewer(file,pages_to_render=[page+1])

st.title("phi3 Bot")
if "db_created" not in st.session_state:
        st.session_state["db_created"] = False
if not st.session_state['db_created']:
    st.warning("Please ensure the database has been created and loaded before querying")
else:
    if "messages_phi3" not in st.session_state:
        st.session_state.messages_phi3 = []

# Display chat messages from history on app rerun
    for message in st.session_state.messages_phi3:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if user_query := st.chat_input("Enter your queries here"):
        # Display user message in chat message container
        st.chat_message("user").markdown(user_query)
        # Add user message to chat history
        st.session_state.messages_phi3.append({"role": "user", "content": user_query})
        with st.spinner('Generating answer...'):
            embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
           #FAISS
            db=FAISS.load_local(f"D:\\ML\\superkalam\\database\\{st.session_state['subject']}\\{st.session_state['grade']}", embedding_function,allow_dangerous_deserialization=True)
            #Chroma
            # db = Chroma(persist_directory=f"D:\\ML\\superkalam\\database_chroma\\{st.session_state['subject']}\\{st.session_state['grade']}", embedding_function=embedding_function)
            
            docs_faiss = db.similarity_search(user_query)
            llm = Ollama(model='phi3')
            chain = load_qa_chain(llm, chain_type="stuff")
            answer = chain.run(input_documents=docs_faiss, question=user_query)

        # if(st.button("View Sources")):
            
                # st.write(doc.metadata)
        with st.chat_message("assistant"):
            st.markdown(answer + " The sources have been shown below:")
        for doc in docs_faiss:
            view_pdf(doc.metadata['source'], doc.metadata['page'])
        # Add assistant response to chat history
        st.session_state.messages_phi3.append({"role": "assistant", "content": answer})
