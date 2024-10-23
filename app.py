import streamlit as st
from PIL import Image
import uuid
import csv
import os
from langchain_openai import AzureChatOpenAI


def save_unique_id_to_csv(unique_id, uploaded_file, config_file_path, csv_file_path='unique_ids.csv'):
    # # Ensure the directory for the CSV file exists
    os.makedirs("transactions", exist_ok=True)

    # Get the location of the uploaded file
    uploaded_file_path = os.path.abspath(uploaded_file.name)

    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([unique_id, uploaded_file_path, config_file_path])

    else:
        # Write the unique_id, uploaded_file_path, and config_file_path to the CSV file
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([unique_id, uploaded_file_path, config_file_path])



def save_config_to_csv(configurations, unique_id, csv_file_path='transactions/configurations.csv'):
    # Ensure the directory for the CSV file exists
    os.makedirs("transactions", exist_ok=True)

    # Create the CSV file if it does not exist
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['unique_id'] + list(configurations.keys()))

    # Write the configurations to the CSV file
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([unique_id] + list(configurations.values()))


def save_file_to_db(file_name, file_content, unique_id):
    pass

def upload_to_rabbitmq(file_content):
    return True

# Loading Image using PIL
im = Image.open('content/App_Icon.png')

# Adding Image to web app
st.set_page_config(page_title="Prediction App", page_icon=im, layout="wide")


        
def handle_question(question: str) -> str:
    """
    Function to handle the question and return a response.
    
    Args:
        question (str): The question asked by the user.
    
    Returns:
        str: The response to the question.
    """
    from langchain_community.utilities import SQLDatabase

    db = SQLDatabase.from_cnosdb(url="127.0.0.1:8902",
                              user="root",
                              password="",
                              tenant="cnosdb",
                              database="public")

    from langchain.agents import create_sql_agent
    from langchain_community.agent_toolkits import SQLDatabaseToolkit

    os.environ["AZURE_OPENAI_API_KEY"] = ""
    os.environ["AZURE_OPENAI_ENDPOINT"] = ""
    os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "Standard"
    os.environ["AZURE_OPENAI_MODEL_NAME"] = "gpt-4"

    os.environ["OPENAI_API_VERSION"] = "2024-02-15-preview"

    llm_azure_openai = AzureChatOpenAI(
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        # azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        model_name=os.environ["AZURE_OPENAI_MODEL_NAME"],
        # openai_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        temperature=0.7
    )


    toolkit = SQLDatabaseToolkit(db=db, llm=llm_azure_openai)
    agent = create_sql_agent(llm=llm_azure_openai, toolkit=toolkit, verbose=True)
    response = agent.run( str )

    return response



# Custom CSS to set background colors and center content
st.markdown(
    """
    <style>
    .left-panel {
        # background-color: #003366;  /* Dark blue color for left sidebar */
        padding: 10px;
        height: 100vh;  /* Full height */
    }
    .central-content {
        # background-color: #ffffff;
        padding: 10px;
    }
    .right-sidebar {
        background-color: #e0e0e0;
        padding: 10px;
    }
    .center-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define the content for each page
def home():
    st.markdown('<div class="center-content">', unsafe_allow_html=True)
    st.title("Welcome, I am your personal Prediction App!")
    st.image(r'content/PredictApp_Icon2.jpg', caption="Prediction App")
    st.markdown('</div>', unsafe_allow_html=True)
    st.write("Discover the power of prediction with our Streamlit Python app! It excels in time series forecasting and YouTube video sentiment analysis. Currently in its learning phase, it features a telemetry system to monitor model data usage. Unlock insights and make informed decisions effortlessly!")


def page1():
    st.title("Time Series Data Analysis")
    st.header("Time Series Data Analysis")

    # State management to handle form submission and unique ID display
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False
    if 'unique_id' not in st.session_state:
        st.session_state.unique_id = None

    if not st.session_state.submitted:
        # Streamlit widgets to get user input
        # Data source selection
        data_source = st.selectbox('Data Source', [' ', 'File Upload'])

        uploaded_file = st.file_uploader("Upload your file")

        # Input field for data format
        data_format = st.selectbox('Format', [' ', 'CSV', 'JSON'])

        # Data processing options
        st.write('Data Processing Options')
        missing_data_handling = st.selectbox('Missing Data Handling', [' ', 'Drop', 'Fill'])
        normalization = st.selectbox('Normalization', [' ', 'Min-Max', 'Z-Score'])

        # Algorithm selection
        algorithm = st.selectbox('Algorithm', [' ', 'ARIMA', 'LSTM', 'Prophet'])

        # Task selection
        task = st.selectbox('Task', [' ', 'Prediction', 'Forecasting', 'Anomaly Detection'])

        # Target variable input field
        target_variable = st.text_input('Target Variable')

        configurations = {
            'data_source': data_source,
            'data_format': data_format,
            'missing_data_handling': missing_data_handling,
            'normalization': normalization,
            'algorithm': algorithm,
            'task': task,
            'target_variable': target_variable
        }

        if st.button('Generate Unique ID and Upload'):
            # Check if any configuration value is blank
            if any(value == '' or value == ' ' for value in configurations.values()):
                st.error("All fields must be filled out. Please ensure no field is left blank.")
                st.rerun()  # Refresh the page to reset fields

            if uploaded_file is None:
                st.error("File upload is required. Please ensure no field is left blank.")
                st.rerun()  # Refresh the page to reset fields

            unique_id = str(uuid.uuid4())
            st.session_state.unique_id = unique_id
            st.session_state.submitted = True

            # Save configurations to csv
            save_config_to_csv(configurations, unique_id, 'transactions/configurations_'+unique_id+'.csv')

            # Save unique_id to csv
            save_unique_id_to_csv(unique_id, uploaded_file, 'transactions/configurations_'+unique_id+'.csv')

            if uploaded_file is not None:
                file_content = uploaded_file.read()
                file_name = uploaded_file.name

                # Save file to transactions folder
                file_extension = 'csv' if data_format == 'CSV' else 'json'
                file_path = f'transactions/tsdata_{unique_id}.{file_extension}'
                with open(file_path, 'wb') as f:
                    f.write(file_content)

                # Save file to database
                save_file_to_db(file_name, file_content, unique_id)

                if upload_to_rabbitmq(file_content):
                    st.success('File uploaded successfully to RabbitMQ')

            st.experimental_set_query_params(unique_id=st.session_state.unique_id)
            st.rerun()  # Refresh the page to reset fields

    else:
        st.write(f"Generated Unique ID: {st.session_state.unique_id}")

        # Search functionality
        search_id = st.text_input('Enter Unique ID to search')
        if st.button('Search'):
            # Implement search logic here
            forecast_file_path = f'transactions/forecast_{search_id}.txt'
            if os.path.exists(forecast_file_path):
                with open(forecast_file_path, 'r') as file:
                    forecast_content = file.read()
                st.write(f"Forecast for ID {search_id}:")
                st.write(forecast_content)
            else:
                st.write("We are still calculating your predictions or you provided incorrect data to predict.")


    # Button to refresh and go back to the main page
    if st.button('Refresh'):
        st.session_state.submitted = False
        st.session_state.unique_id = None
        st.markdown('<meta http-equiv="refresh" content="0">', unsafe_allow_html=True)

    st.write('Model creation and execution based on user input will begin...')

def page2():
    st.title("Talk to my TS Data")
    st.header("Talk to my TS Data")

    # Sample data for tenants, assets, and variables
    tenants = ["betauser", "tpusr12"]
    assets = {
        
        "betauser": ["DataAsset_XceleratorIOTLab"],
        "tpusr12": ["e2ePuneADMCSA05"]
    }
    variables = {
        "DataAsset_XceleratorIOTLab": ["RoomTemperature"],
        "e2ePuneADMCSA05": ["IntTriangle1000ms"]
    }

    # Dropdown for selecting tenant
    selected_tenant = st.selectbox("Select Tenant", [" "] + tenants)

    if selected_tenant != " ":
        # Dropdown for selecting asset
        selected_asset = st.selectbox("Select Asset", [" "] + assets[selected_tenant])

        if selected_asset != " ":
            # Dropdown for selecting variable
            selected_variable = st.selectbox("Select Variable", [" "] + variables[selected_asset])

            if selected_variable != " ":
                # Text field for asking questions
                question = st.text_input("Ask a question about the selected variable")

                # Option to show sample questions
                if st.checkbox("Show sample questions"):
                    sample_questions = [
                        f"What is the average value of {selected_variable} in table tpusr12?",
                        f"What is the range of data present on table tpusr12?",
                        f"What is the trend of {selected_variable} over the last month?",
                        f"How does {selected_variable} correlate with other variables?",
                        f"What are the anomalies in {selected_variable} data?",
                        f"What is the average value of {selected_variable} at table {selected_tenant}?"
                    ]
                    st.write("Sample Questions:")
                    for q in sample_questions:
                        st.write(f"- {q}")

                # Button to submit the question
                if st.button("Submit Question"):
                    st.write(f"Your question: {question}")
                    #  logic to handle the question 
                    p = handle_question(question)
                    st.write(f"Response: {p}")
    if st.button('Refresh'):
        st.session_state.submitted = False
        st.session_state.unique_id = None
        st.markdown('<meta http-equiv="refresh" content="0">', unsafe_allow_html=True)

def page3():
    st.title("Page 3")
    st.write("Content for Page 3 goes here.")

# Sidebar navigation
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to", ["Home", "Time Series Data Analysis", "Talk to my TS Data", "Page 3"], key="left_sidebar_radio")

# Main content and right sidebar layout
col1, col2, col3 = st.columns([1, 3, 1])

with col1:
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="central-content">', unsafe_allow_html=True)
    # Display the selected page
    if page == "Home":
        home()
    elif page == "Time Series Data Analysis":
        page1()
    elif page == "Talk to my TS Data":
        page2()
    elif page == "Page 3":
        page3()
    st.markdown('</div>', unsafe_allow_html=True)

