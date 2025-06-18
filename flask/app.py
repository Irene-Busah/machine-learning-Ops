# importing needed libraries
from flask import Flask

# creating a Flask application instance
app = Flask(__name__)

# creating an API endpoint
@app.route("/")
def welcome():
    """
    A simple welcome endpoint that returns a greeting message.
    """
    return "Welcome to the Flask application!"



if __name__ == '__main__':
    # running the Flask application
    app.run(debug=True)
