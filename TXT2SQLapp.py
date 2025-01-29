import gradio as gr
import sqlite3
import sqlparse
import os
import re
import random
from tqdm import tqdm
from time import sleep
import dspy
from dspy.datasets import DataLoader
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, LabeledFewShot


TOKEN="TOKEN"
ENDPOINT = 'https://models.inference.ai.azure.com'
HF_TOKEN = 'API_TOKEN'

os.environ['HUGGINGFACE_TOKEN']= HF_TOKEN
os.environ['GITHUB_TOKEN']= TOKEN
os.environ["OPENAI_API_KEY"] = TOKEN
os.environ['OPENAI_API_BASE'] = ENDPOINT


generation_args = {
    "temperature":0,
    "max_tokens":500,
    "stop":"\n\n",
    "model_type":"chat",
    "n": 1
}

model_info = {
    "gpt-4o": {"model": "openai/gpt-4o", "api_base": ENDPOINT, "api_key":TOKEN}
}



# Database setup
DATABASE_PATH = "employees.db"

table_schemas = """
-- Employee Data Table
CREATE TABLE employee_data (
    EmpID INTEGER,
    FirstName TEXT,
    LastName TEXT,
    StartDate TEXT,
    ExitDate TEXT,
    Title TEXT,
    Supervisor TEXT,
    ADEmail TEXT,
    BusinessUnit TEXT,
    EmployeeStatus TEXT,
    EmployeeType TEXT,
    PayZone TEXT,
    EmployeeClassificationType TEXT,
    TerminationType TEXT,
    TerminationDescription TEXT,
    DepartmentType TEXT,
    Division TEXT,
    DOB TEXT,
    State TEXT,
    JobFunctionDescription TEXT,
    GenderCode TEXT,
    LocationCode INTEGER,
    RaceDesc TEXT,
    MaritalDesc TEXT,
    PerformanceScore TEXT,
    CurrentEmployeeRating INTEGER
);

-- Recruitment Data Table
CREATE TABLE recruitment_data (
    ApplicantID INTEGER,
    ApplicationDate TEXT,
    FirstName TEXT,
    LastName TEXT,
    Gender TEXT,
    DOB TEXT,
    PhoneNumber TEXT,
    Email TEXT,
    Address TEXT,
    City TEXT,
    State TEXT,
    ZipCode INTEGER,
    Country TEXT,
    EducationLevel TEXT,
    YearsofExperience INTEGER,
    DesiredSalary REAL,
    Title TEXT,
    Status TEXT
);

-- Training and Development Data Table
CREATE TABLE training_and_development_data (
    EmpID INTEGER,
    TrainingDate TEXT,
    TrainingProgramName TEXT,
    TrainingType TEXT,
    TrainingOutcome TEXT,
    Location TEXT,
    Trainer TEXT,
    TrainingDurationInDays INTEGER,
    TrainingCost INTEGER
);

-- Employee Engagement Survey Data Table
CREATE TABLE employee_engagement_survey_data (
    EmpID INTEGER,
    SurveyDate TEXT,
    EngagementScore INTEGER,
    SatisfactionScore INTEGER,
    WorklifeBalanceScore INTEGER
);
"""

class Text2SQLSignature(dspy.Signature):
	"""Transform a natural language query into a SQL query.
	You will be given a question which will tell what you need to do
	and a sql_context which will give some additional context to generate the right SQL.
	```Only generate the SQL query nothing else. You should give one correct answer.
	starting and ending with ```
	"""
	question = dspy.InputField(desc="Natural language query")
	sql_context = dspy.InputField(desc="Context for the query")
	sql = dspy.OutputField(desc="SQL Query")

class Text2SQLProgram(dspy.Module):
	def __init__(self):
		super().__init__()
		self.program = dspy.ChainOfThought(signature=Text2SQLSignature)

	def forward(self, question, sql_context):
		return self.program(
			question=question,
			sql_context=sql_context
		)

def setup_sql_chain():
    lm = dspy.LM(**model_info["gpt-4o"], **generation_args)
    dspy.configure(lm=lm)
    
    text2sql = Text2SQLProgram()

    text2sql.load("optimized_text2sql_model.json")
  
    return text2sql

text2sql = setup_sql_chain()

def execute_query(history, question):
    """Executes the SQL query on the SQLite database and returns the result"""
    
    db_connection = sqlite3.connect(DATABASE_PATH)
    history.append(gr.ChatMessage(role="user", content = question))
    # interaction = [{"role": "user", "content": question}]
    # interaction.append(("user", question))
    
    try:
        sql_query = text2sql(question = question, sql_context = table_schemas)
        formatted_sql = sqlparse.format(sql_output, reindent=True)  
        cursor = db_connection.cursor()
        cursor.execute(formatted_sql)
        results = cursor.fetchall()
        
        if results:
            message = f"**Generated SQL Query:**\n```\n{formatted_sql}\n```\n\n**Query Results:**\n{results}"
        else:
            message = f"**Generated SQL Query:**\n```\n{formatted_sql}\n```\n\n**No Results Found.**"

    except Exception as e:
        message = f"**Error:** {str(e)}"

    finally:
        db_connection.close()

    # history.append(gr.ChatMessage(role="assistant", content = ''))
    # for character in message:
    #     history[-1].content += character
    #     sleep(0.05)
    #     yield history
    
    history.append(gr.ChatMessage(role="assistant", content = message))
    # interaction.append({"role": "assistant", "content": message})
    # interaction.append(("assistant", message))
    # history.append(interaction)

    return history, ''




# Gradio Interface
start = [gr.ChatMessage(role="assistant", content = "Please ask me any question about the Employees database")]


with gr.Blocks() as app:
    # def user(user_message, history: list):
    #     return history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": ""}]

    # def bot(history: list):
    #     bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
    #     history.append({"role": "assistant", "content": ""})
    #     for character in bot_message:
    #         history[-1]['content'] += character
    #         time.sleep(0.05)
    #         yield history

    # msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
    #     bot, chatbot, chatbot
    # )

    gr.Markdown("# Text-to-SQL Chatbot")
    gr.Markdown("Ask questions about the database in natural language. The chatbot will translate your query into SQL, execute it, and return the results.")
    
    chatbot = gr.Chatbot(start, label="SQL Chatbot", value = [], type = "messages")
    query_input = gr.Textbox(label="Your Query", placeholder="Type your question about the database...")
    # send_button = gr.Button("Send")

    # Bind actions
    query_input.submit(execute_query, [chatbot, query_input], [chatbot, query_input])

# Run the app
app.launch()
