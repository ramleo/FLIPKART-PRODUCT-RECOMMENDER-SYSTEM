from flask import render_template, Flask, request, Response
from prometheus_client import Counter, generate_latest
from flipkart.data_ingestion import DataIngestor
# Ensure this matches your filename (rag_chain.py or rag_chain_copy.py)
from flipkart.rag_chain import RAGChainBuilder 
from rich.console import Console
from rich.markdown import Markdown
import markdown

from dotenv import load_dotenv
load_dotenv()

REQUEST_COUNT = Counter("http_requests_total", "Total HTTP Request")
PREDICTION_COUNT = Counter("responses_total", "Total Responses")
# console = Console()
# console.print(Markdown(response_text))


def create_app():
    app = Flask(__name__)
    vector_store = DataIngestor().ingest(load_existing=True)
    rag_chain = RAGChainBuilder(vector_store).build_chain()

    @app.route("/")
    def index():
        REQUEST_COUNT.inc()
        return render_template("index.html")
    
    @app.route("/get", methods=["POST"])
    def get_response():
        PREDICTION_COUNT.inc()
        user_input = request.form["msg"]

        # FIX: The chain returns a STRING directly. 
        # Accessing ["answer"] on a string causes the 'indices must be integers' error.
        response_text = rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "user-session"}}
        )

        # console = Console()
        # console.print(Markdown(response_text))

        html_output = markdown.markdown(response_text, extensions=['extra'])
        
        return html_output # This is now the clean string answer
    
    @app.route("/metrics")
    def metrics():
        return Response(generate_latest(), mimetype="text/plain")
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
