# importing the necessary libraries
from flask import Flask, render_template

# creating a Flask application instance
app = Flask(__name__)


# creating an API endpoint
@app.route("/")
def welcome():
    """
    A simple welcome endpoint that returns a greeting message.
    """
    return "<html><h1>Welcome to the Flask application!</h1></html>"

@app.route("/index")
def index():
    """
    An index endpoint that returns a simple HTML page.
    """
    return render_template("index.html")


if __name__ == '__main__':    # running the Flask application
    app.run(debug=True)
