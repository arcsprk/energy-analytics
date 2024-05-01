#!/bin/env python3
import os
import tempfile
import streamlit as st
from streamlit_chat import message
from rag import ChatPDF, ChatCSV

st.set_page_config(page_title="SPark Assistant")


def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def read_and_save_file():
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""



    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            # st.session_state["assistant"].ingest(file_path)
            # if file_format == 'pdf':
            #     st.session_state["assistant"].ingest_pdf(file_path)                 
            # elif file_format == 'csv':
            #     st.session_state["assistant"].ingest_csv(file_path)
            # else:
            #     raise ValueError("Unsupported file format: {}".format(file_format))

            print("file_path:", file_path)
            uploaded_file = st.session_state.file_uploader

            print("uploaded_file:", uploaded_file)

            if uploaded_file is not None:
                # 파일 타입에 따라 session_state["assistant"]를 변경
                if uploaded_file[-1].type == "application/pdf":
                    st.session_state["assistant"] = ChatPDF()
                    st.session_state["assistant"].ingest_pdf(file_path)  
                    # st.session_state["assistant"] = "PDF 처리 Assistant"
                    # process_pdf(uploaded_file)
                elif uploaded_file[-1].type == "text/csv":
                    # st.session_state["assistant"] = "CSV 처리 Assistant"
                    # process_csv(uploaded_file)
                    st.session_state["assistant"] = ChatCSV()
                    st.session_state["assistant"].ingest_csv(file_path)  

 
            # st.session_state["assistant"].ingest_pdf(file_path)                 


        os.remove(file_path)


def page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatPDF()

    st.header("SPark Assistant")

    st.subheader("Upload a document")
    st.file_uploader(
        "Upload document",
        type=["pdf", "csv"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )


    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)


if __name__ == "__main__":
    page()
