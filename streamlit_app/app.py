#!/bin/env python3
import os
import tempfile
import pandas as pd
import streamlit as st
from streamlit_chat import message
# from chat_model import ChatBase, ChatPDF, ChatCSV
from chat_model import ChatWithCustomRetriver, ChatTable
# from classes import get_primer,format_question,run_request

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


# def read_and_save_file():
#     st.session_state["assistant"].clear()
#     st.session_state["messages"] = []
#     st.session_state["user_input"] = ""


#     for file in st.session_state["file_uploader"]:
#         with tempfile.NamedTemporaryFile(delete=False) as tf:
#             tf.write(file.getbuffer())
#             file_path = tf.name

#         with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
#             # st.session_state["assistant"].ingest(file_path)
#             # if file_format == 'pdf':
#             #     st.session_state["assistant"].ingest_pdf(file_path)                 
#             # elif file_format == 'csv':
#             #     st.session_state["assistant"].ingest_csv(file_path)
#             # else:
#             #     raise ValueError("Unsupported file format: {}".format(file_format))

#             print("file_path:", file_path)
#             uploaded_file = st.session_state.file_uploader

#             print("uploaded_file:", uploaded_file)

#             if uploaded_file is not None:
#                 # 파일 타입에 따라 session_state["assistant"]를 변경
#                 if uploaded_file[-1].type == "application/pdf":
#                     st.session_state["assistant"] = ChatPDF()
#                     st.session_state["assistant"].ingest_pdf(file_path)  
#                     # st.session_state["assistant"] = "PDF 처리 Assistant"
#                     # process_pdf(uploaded_file)
#                 elif uploaded_file[-1].type == "text/csv":
#                     # st.session_state["assistant"] = "CSV 처리 Assistant"
#                     # process_csv(uploaded_file)
#                     st.session_state["assistant"] = ChatCSV()
#                     st.session_state["assistant"].ingest_csv(file_path)
#             else:
#                 st.session_state["assistant"] = ChatBase()

 
#             # st.session_state["assistant"].ingest_pdf(file_path)                 


#         os.remove(file_path)

def format_question(primer_desc,primer_code , question, model_type):
    # Fill in the model_specific_instructions variable
    instructions = ""
    if model_type == "Code Llama":
        # Code llama tends to misuse the "c" argument when creating scatter plots
        instructions = "\nDo not use the 'c' argument in the plot function, use 'color' instead and only pass color names like 'green', 'red', 'blue'."
    primer_desc = primer_desc.format(instructions)  
    # Put the question at the end of the description primer within quotes, then add on the code primer.
    return  '"""\n' + primer_desc + question + '\n"""\n' + primer_code

def get_primer(df_dataset,df_name):
    # Primer function to take a dataframe and its name
    # and the name of the columns
    # and any columns with less than 20 unique values it adds the values to the primer
    # and horizontal grid lines and labeling
    primer_desc = "Use a dataframe called df from data_file.csv with columns '" \
        + "','".join(str(x) for x in df_dataset.columns) + "'. "
    for i in df_dataset.columns:
        if len(df_dataset[i].drop_duplicates()) < 20 and df_dataset.dtypes[i]=="O":
            primer_desc = primer_desc + "\nThe column '" + i + "' has categorical values '" + \
                "','".join(str(x) for x in df_dataset[i].drop_duplicates()) + "'. "
        elif df_dataset.dtypes[i]=="int64" or df_dataset.dtypes[i]=="float64":
            primer_desc = primer_desc + "\nThe column '" + i + "' is type " + str(df_dataset.dtypes[i]) + " and contains numeric values. "   
    primer_desc = primer_desc + "\nLabel the x and y axes appropriately."
    primer_desc = primer_desc + "\nAdd a title. Set the fig suptitle as empty."
    primer_desc = primer_desc + "{}" # Space for additional instructions if needed
    primer_desc = primer_desc + "\nUsing Python version 3.9.12, create a script using the dataframe df to graph the following: "
    pimer_code = "import pandas as pd\nimport matplotlib.pyplot as plt\n"
    pimer_code = pimer_code + "fig,ax = plt.subplots(1,1,figsize=(10,4))\n"
    pimer_code = pimer_code + "ax.spines['top'].set_visible(False)\nax.spines['right'].set_visible(False) \n"
    pimer_code = pimer_code + "df=" + df_name + ".copy()\n"
    return primer_desc,pimer_code


def page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatWithCustomRetriver()
        st.session_state["assistant"] = ChatTable()
        # st.session_state["assistant"] = ChatPDF()

    st.header("SPark Assistant")

    st.subheader("Upload a document")
    # st.file_uploader(
    #     "Upload document",
    #     type=["pdf", "csv"],
    #     key="file_uploader",
    #     on_change=read_and_save_file,
    #     label_visibility="collapsed",
    #     accept_multiple_files=True,
    # )

    st.session_state["ingestion_spinner"] = st.empty()

    # Radio button selection for assistant type
    assistant_type = st.radio("Assistant Type", ("ChatWithCustomRetriver", "ChatTable"))
    st.session_state["assistant_type"] = assistant_type  # Update session state based on selection


    display_messages()


    # # List to hold datasets
    # if "datasets" not in st.session_state:
    #     datasets = {}
    #     # Preload datasets
    #     # datasets["Movies"] = pd.read_csv("movies.csv")
    #     # datasets["Housing"] =pd.read_csv("housing.csv")
    #     # datasets["Cars"] =pd.read_csv("cars.csv")
    #     # datasets["Colleges"] =pd.read_csv("colleges.csv")
    #     # datasets["Customers & Products"] =pd.read_csv("customers_and_products_contacts.csv")
    #     # datasets["Department Store"] =pd.read_csv("department_store.csv")
    #     datasets["Energy Production"] =pd.read_csv("./data/energy_production.csv")
    #     st.session_state["datasets"] = datasets
    # else:
    #     # use the list already loaded
    #     datasets = st.session_state["datasets"]
    # with st.sidebar:
    #     # First we want to choose the dataset, but we will fill it with choices once we've loaded one
    #     dataset_container = st.empty()

    #     # Radio buttons for dataset choice
    #     chosen_dataset = dataset_container.radio(":bar_chart: Choose your data:",datasets.keys(),index=index_no)#,horizontal=True,)

    # selected_models = [model_name for model_name, choose_model in use_model.items() if choose_model]
    # model_count = 1
    # plots = st.columns(model_count)
    # # Get the primer for this dataset
    # primer1,primer2 = get_primer(datasets[chosen_dataset],'datasets["'+ chosen_dataset + '"]') 
    # # Create model, run the request and print the results
    # for plot_num, model_type in enumerate(selected_models):
    #     with plots[plot_num]:
    #         st.subheader(model_type)
    #         try:
    #             # Format the question 
    #             question_to_ask = format_question(primer1, primer2, question, model_type)   
    #             # Run the question
    #             answer=""
    #             answer = run_request(question_to_ask, available_models[model_type], key=openai_key,alt_key=hf_key)
    #             # the answer is the completed Python script so add to the beginning of the script to it.
    #             answer = primer2 + answer
    #             print("Model: " + model_type)
    #             print(answer)
    #             plot_area = st.empty()
    #             plot_area.pyplot(exec(answer))           
    #         except Exception as e:
    #             if type(e) == openai.error.APIError:
    #                 st.error("OpenAI API Error. Please try again a short time later. (" + str(e) + ")")
    #             elif type(e) == openai.error.Timeout:
    #                 st.error("OpenAI API Error. Your request timed out. Please try again a short time later. (" + str(e) + ")")
    #             elif type(e) == openai.error.RateLimitError:
    #                 st.error("OpenAI API Error. You have exceeded your assigned rate limit. (" + str(e) + ")")
    #             elif type(e) == openai.error.APIConnectionError:
    #                 st.error("OpenAI API Error. Error connecting to services. Please check your network/proxy/firewall settings. (" + str(e) + ")")
    #             elif type(e) == openai.error.InvalidRequestError:
    #                 st.error("OpenAI API Error. Your request was malformed or missing required parameters. (" + str(e) + ")")
    #             elif type(e) == openai.error.AuthenticationError:
    #                 st.error("Please enter a valid OpenAI API Key. (" + str(e) + ")")
    #             elif type(e) == openai.error.ServiceUnavailableError:
    #                 st.error("OpenAI Service is currently unavailable. Please try again a short time later. (" + str(e) + ")")               
    #             else:
    #                 st.error("Unfortunately the code generated from the model contained errors and was unable to execute.")

    # # Display the datasets in a list of tabs
    # # Create the tabs
    # tab_list = st.tabs(datasets.keys())

    # # Load up each tab with a dataset
    # for dataset_num, tab in enumerate(tab_list):
    #     with tab:
    #         # Can't get the name of the tab! Can't index key list. So convert to list and index
    #         dataset_name = list(datasets.keys())[dataset_num]
    #         st.subheader(dataset_name)
    #         st.dataframe(datasets[dataset_name],hide_index=True)

    # # Insert footer to reference dataset origin  
    # footer="""<style>.footer {position: fixed;left: 0;bottom: 0;width: 100%;text-align: center;}</style><div class="footer">
    # <p> <a style='display: block; text-align: center;'> Datasets courtesy of NL4DV, nvBench and ADVISor </a></p></div>"""
    # st.caption("Datasets courtesy of NL4DV, nvBench and ADVISor")

    # # Hide menu and footer
    # hide_streamlit_style = """
    #             <style>
    #             #MainMenu {visibility: hidden;}
    #             footer {visibility: hidden;}
    #             </style>
    #             """
    # st.markdown(hide_streamlit_style, unsafe_allow_html=True)


    st.text_input("Message", key="user_input", on_change=process_input)


if __name__ == "__main__":
    page()
