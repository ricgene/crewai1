from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
from langchain_community.utilities.sql_database import SQLDatabase
from crewai_tools.tools import tool 
#from crewai.tools import tool
from langchain_community.llms import Ollama

# Load the database
db = SQLDatabase.from_uri("sqlite:///salaries.db")

# Define the tools
@tool("list_tables")
def list_tables() -> str:
    return ListSQLDatabaseTool(db=db).invoke("")

@tool("tables_schema")
def tables_schema(tables: str) -> str:
    tool = InfoSQLDatabaseTool(db=db)
    return tool.invoke(tables)

@tool("execute_sql")
def execute_sql(sql_query: str) -> str:
    return QuerySQLDataBaseTool(db=db).invoke(sql_query)

@tool("check_sql")
def check_sql(sql_query: str) -> str:
    return QuerySQLCheckerTool(db=db, llm=llm).invoke({"query": sql_query})

###########

from crewai import Agent

llm = Ollama(model="phi3")

sql_dev = Agent(
    role="Senior Database Developer",
    goal="Construct and execute SQL queries based on a request",
    backstory="""
        You are an experienced database engineer who is a master at creating efficient and complex SQL queries.
        You have a deep understanding of how different databases work and how to optimize queries.
        Use the `list_tables` to find available tables.
        Use the `tables_schema` to understand the metadata for the tables.
        Use the `execute_sql` to check your queries for correctness.
        Use the `check_sql` to execute queries against the database.
    """,
    llm=llm,
    tools=[list_tables, tables_schema, execute_sql, check_sql],
    allow_delegation=False,
)

data_analyst = Agent(
    role="Senior Data Analyst",
    goal="Analyze data retrieved from the database.",
    backstory="""
        You have deep experience with analyzing datasets using Python.
        Your work is always based on the provided data and is clear,
        easy-to-understand, and to the point. You have attention
        to detail and always produce very detailed work.
    """,
    llm=llm,
    allow_delegation=False,
)

report_writer = Agent(
    role="Senior Report Editor",
    goal="Write an executive summary based on the analyst's work.",
    backstory="""
        Your writing style is well-known for clear and effective communication.
        You always summarize long texts into bullet points that contain the most
        important details.
    """,
    llm=llm,
    allow_delegation=False,
)

#############

from crewai import Task, Crew, Process

extract_data = Task(
    description="Extract data that is required for the query {query}.",
    expected_output="Database result for the query",
    agent=sql_dev,
)

analyze_data = Task(
    description="Analyze the data from the database and write an analysis for {query}.",
    expected_output="Detailed analysis text",
    agent=data_analyst,
    context=[extract_data],
)

write_report = Task(
    description="Write an executive summary of the report from the analysis. The report must be less than 100 words.",
    expected_output="Markdown report",
    agent=report_writer,
    context=[analyze_data],
)

crew = Crew(
    agents=[sql_dev, data_analyst, report_writer],
    tasks=[extract_data, analyze_data, write_report],
    process=Process.sequential,
    verbose=2,
    memory=False,
    output_log_file="crew.log",
)

###############

inputs = {
    "query": "How is the salaray in USD based on employment type and experience level? "
}

result = crew.kickoff(inputs=inputs)

