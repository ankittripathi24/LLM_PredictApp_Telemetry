import os
import json
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback
# from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from langchain_community.agent_toolkits.load_tools import load_tools
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
from time import time

# Initialize OpenTelemetry SDK for tracing
trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

# Initialize OpenTelemetry SDK for metrics
metric_exporter = ConsoleMetricExporter()
metric_reader = PeriodicExportingMetricReader(metric_exporter)
metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))

# Instrument LangChain
# LangchainInstrumentor().instrument()

# Set environment variables
os.environ["AZURE_OPENAI_API_KEY"] = "93b312962a264b469d28bb4025344dfd"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://openai-aiattack-msa-000898-australiaeast-genaiih-00.openai.azure.com/"
os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "Standard"
os.environ["AZURE_OPENAI_MODEL_NAME"] = "gpt-4"
os.environ["OPENAI_API_VERSION"] = "2024-02-15-preview"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"

# Initialize AzureChatOpenAI model
llm_azure_openai = AzureChatOpenAI(
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    model_name=os.environ["AZURE_OPENAI_MODEL_NAME"],
    temperature=0.7
)

# Define the LangChain pipeline for time series prediction
@traceable
def time_series_prediction(question, context):
    tracer = trace.get_tracer(__name__)
    meter = metrics.get_meter(__name__)
    
    # Create custom metrics
    response_time_histogram = meter.create_histogram("llm.response_time", unit="s", description="Response time of LLM calls")
    token_usage_counter = meter.create_counter("llm.token_usage", unit="tokens", description="Token usage of LLM calls")
    
    with tracer.start_as_current_span("time_series_prediction") as span:
        start_time = time()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Please respond to the user's request only based on the given context."),
            ("user", "Question: {question}\nContext: {context}")
        ])

        output_parser = StrOutputParser()
        chain = prompt | llm_azure_openai | output_parser

        result = chain.invoke({"question": question, "context": context})
        
        end_time = time()
        response_time = end_time - start_time
        
        # Parse the result if it's a JSON string
        try:
            result_dict = json.loads(result)
        except json.JSONDecodeError:
            result_dict = {}

        # Log custom metrics
        span.set_attribute("llm.response_time", response_time)
        span.set_attribute("llm.token_usage", result_dict.get("usage", {}).get("total_tokens", "unknown"))
        span.set_attribute("llm.model_name", os.environ["AZURE_OPENAI_MODEL_NAME"])
        
        # Record custom metrics
        response_time_histogram.record(response_time)
        token_usage_counter.add(result_dict.get("usage", {}).get("total_tokens", 0))
        
        return result

# Example usage
if __name__ == "__main__":
    question = "Can you summarize this morning's meetings?"
    context = "During this morning's meeting, we solved all world conflict."

    result = time_series_prediction(question, context)
    print(result)