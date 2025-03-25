# # # from flask import Flask, request, render_template, url_for
# # # import pickle
# # # import numpy as np
# # # import json
# # # import requests

# # # app = Flask(__name__)
# # # with open('Phishing_model.pkl', 'rb') as file:
# # #     model = pickle.load(file)

# # # @app.route("/")
# # # def f():
# # #     return render_template("index.html")

# # # @app.route("/inspect")
# # # def inspect():
# # #     return render_template("inspect.html")


# # # @app.route("/output", methods=["GET", "POST"])
# # # def output():
# # #     if request.method == 'POST':
# # #         var1 = request.form["UsingIP"]
# # #         var2 = request.form["PrefixSuffix-"]
# # #         var3 = request.form["SubDomains"]
# # #         var4 = request.form["HTTPS"]
# # #         var5 = request.form["NonStdPort"]
# # #         var6 = request.form["HTTPSDomainURL"]
# # #         var7 = request.form["RequestURL"]
# # #         var8 = request.form["AnchorURL"]
# # #         var9 = request.form["LinksInScriptTags"]
# # #         var10 = request.form["ServerFormHandler"]
# # #         var11 = request.form["InfoEmail"]
# # #         var12 = request.form["AbnormalURL"]
# # #         var13 = request.form["WebsiteForwarding"]
# # #         var14 = request.form["StatusBarCust"]
# # #         var15 = request.form["DisableRightClick"]
# # #         var16 = request.form["AgeofDomain"]
# # #         var17 = request.form["DNSRecording"]
# # #         var18 = request.form["WebsiteTraffic"]
# # #         var19 = request.form["PageRank"]
# # #         var20 = request.form["GoogleIndex"]
# # #         var21 = request.form["StatsReport"]

# # #         # Convert the input data into a numpy array
# # #         predict_data = np.array([var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, 
# # #                                  var11, var12,var13,var14,var15,var16,var17,var18,var19,var20,var21]).reshape(1, -1)
# # #         # Use the loaded model to make predictions
# # #         predict = model.predict(predict_data)

# # #         if (predict == 1):
# # #             return render_template('output.html', predict="safe")
# # #         elif (predict == -1):
# # #             return render_template('output.html', predict= "Not safe")
# # #         else:
# # #             return render_template('output.html', predict="Suspicious")
# # #     return render_template("output.html")

# # # if __name__ == "__main__":
# # #     app.run(debug=False)
# # # from flask import Flask, request, render_template, jsonify
# # # from flask_cors import CORS
# # # import pandas as pd

# # # app = Flask(__name__)
# # # CORS(app)  # Enable CORS for all routes

# # # # Load dataset and ensure proper URL formatting
# # # df = pd.read_csv("dataset.csv")
# # # df['url'] = df['url'].astype(str).str.strip().str.lower()  # Normalize dataset URLs
# # # known_phishing_urls = set(df['url'])  # Store in a set for fast lookup

# # # @app.route("/")
# # # def home():
# # #     return render_template("index.html")

# # # @app.route("/predict", methods=["POST"])
# # # def predict():
# # #     try:
# # #         # Parse incoming JSON data
# # #         data = request.get_json()
# # #         url = data.get("url")

# # #         if not url:
# # #             return jsonify({"error": "No URL provided"}), 400

# # #         # Normalize input URL for proper matching
# # #         normalized_url = url.strip().lower()

# # #         # Check if the URL exists in the dataset
# # #         if normalized_url in known_phishing_urls:
# # #             result = "Phishing"
# # #         else:
# # #             result = "Legitimate"

# # #         response = {"url": url, "prediction": result}
# # #         print(response)  # Log the response for debugging
# # #         return jsonify(response)

# # #     except Exception as e:
# # #         print("Error occurred:", e)  # Log error for debugging
# # #         return jsonify({"error": "Internal Server Error"}), 500

# # # if __name__ == "__main__":
# # #     app.run(debug=True)








# # # from flask import Flask, request, render_template, jsonify
# # # from flask_cors import CORS
# # # import pandas as pd

# # # app = Flask(__name__)
# # # CORS(app)  # Enable CORS for all routes

# # # # Load dataset and ensure proper URL formatting
# # # df = pd.read_csv("balanced_dataset.csv")
# # # df['url'] = df['url'].astype(str).str.strip().str.lower()  # Normalize dataset URLs
# # # known_phishing_urls = set(df['url'])  # Store in a set for fast lookup

# # # @app.route("/")
# # # def home():
# # #     return render_template("index.html")

# # # @app.route("/predict", methods=["POST"])
# # # def predict():
# # #     try:
# # #         # Parse incoming JSON data
# # #         data = request.get_json()
# # #         url = data.get("url")

# # #         if not url:
# # #             return jsonify({"error": "No URL provided"}), 400

# # #         # Normalize input URL for proper matching
# # #         normalized_url = url.strip().lower()

# # #         # Check if the URL exists in the dataset
# # #         result = "Phishing" if normalized_url in known_phishing_urls else "Legitimate"

# # #         response = {"url": url, "prediction": result}
# # #         print(response)  # Log the response for debugging
# # #         return jsonify(response)

# # #     except Exception as e:
# # #         print("Error occurred:", e)  # Log error for debugging
# # #         return jsonify({"error": "Internal Server Error"}), 500

# # # if __name__ == "__main__":
# # #     app.run(debug=True)


# # # from flask import Flask, request, render_template, jsonify
# # # from flask_cors import CORS
# # # import pandas as pd
# # # import joblib
# # # import re

# # # app = Flask(__name__)
# # # CORS(app)  # Enable CORS for all routes

# # # # Load dataset and ensure proper URL formatting
# # # df = pd.read_csv("balanced_dataset.csv")
# # # df['url'] = df['url'].astype(str).str.strip().str.lower()  # Normalize dataset URLs
# # # known_phishing_urls = set(df['url'])  # Store in a set for fast lookup

# # # # Load the trained ML model and scaler
# # # model = joblib.load("Phishing_model.pkl")
# # # scaler = joblib.load("scaler.pkl")

# # # # Feature extraction function
# # # def extract_features(url):
# # #     features = [
# # #         len(url),  # URL length
# # #         url.count('.'),  # Number of dots
# # #         1 if url.startswith("https") else 0,  # HTTPS usage
# # #         len(re.findall(r'[?&=%@]', url)),  # Special characters
# # #         len(re.findall(r'\\d', url)),  # Number of digits
# # #         url.count('.') - 1,  # Subdomain count
# # #         1 if re.search(r'\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b', url) else 0  # IP presence
# # #     ]
# # #     return features

# # # @app.route("/")
# # # def home():
# # #     return render_template("index.html")

# # # @app.route("/predict", methods=["POST"])
# # # def predict():
# # #     try:
# # #         # Parse incoming JSON data
# # #         data = request.get_json()
# # #         url = data.get("url")

# # #         if not url:
# # #             return jsonify({"error": "No URL provided"}), 400

# # #         # Normalize input URL for proper matching
# # #         normalized_url = url.strip().lower()

# # #         # Check if the URL exists in the dataset
# # #         if normalized_url in known_phishing_urls:
# # #             result = "Phishing"
# # #         else:
# # #             # Extract features and scale input for ML model
# # #             features = extract_features(normalized_url)
# # #             features_scaled = scaler.transform([features])
# # #             prediction = model.predict(features_scaled)[0]
# # #             result = "Phishing" if prediction == 1 else "Legitimate"

# # #         response = {"url": url, "prediction": result}
# # #         print(response)  # Log the response for debugging
# # #         return jsonify(response)

# # #     except Exception as e:
# # #         print("Error occurred:", e)  # Log error for debugging
# # #         return jsonify({"error": "Internal Server Error"}), 500

# # # if __name__ == "__main__":
# # #     app.run(debug=True)



# # from flask import Flask, request, render_template, jsonify
# # from flask_cors import CORS
# # import pandas as pd

# # app = Flask(__name__)
# # CORS(app)  # Enable CORS for all routes

# # # Load dataset
# # df = pd.read_csv("dataset_updated (1).csv")
# # known_phishing_urls = set(df['url'])  # Store in a set for fast lookup

# # @app.route("/")
# # def home():
# #     return render_template("index.html")

# # @app.route("/predict", methods=["POST"])
# # def predict():
# #     try:
# #         # Parse incoming JSON data
# #         data = request.get_json()
# #         url = data.get("url")

# #         if not url:
# #             return jsonify({"error": "No URL provided"}), 400

# #         # Check if the URL exists in the dataset
# #         result = "Phishing" if url in known_phishing_urls else "Legitimate"

# #         response = {"url": url, "prediction": result}
# #         print(response)  # Log the response for debugging
# #         return jsonify(response)

# #     except Exception as e:
# #         print("Error occurred:", e)  # Log error for debugging
# #         return jsonify({"error": "Internal Server Error"}), 500

# # if __name__ == "__main__":
# #     app.run(debug=True)


# from flask import Flask, request, render_template, jsonify
# from flask_cors import CORS
# import pandas as pd

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # Load dataset
# df = pd.read_csv("balanced_dataset.csv")
# known_phishing_urls = set(df['url'])  # Store in a set for fast lookup

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Parse incoming JSON data
#         data = request.get_json()
#         url = data.get("url")

#         if not url:
#             return jsonify({"error": "No URL provided"}), 400

#         # Check if the URL exists in the dataset
#         result = "Phishing" if url in known_phishing_urls else "Legitimate"

#         response = {"url": url, "prediction": result}
#         print(response)  # Log the response for debugging
#         return jsonify(response)

#     except Exception as e:
#         print("Error occurred:", e)  # Log error for debugging
#         return jsonify({"error": "Internal Server Error"}), 500

# if __name__ == "__main__":
#     app.run(debug=True)


# from flask import Flask, request, render_template, jsonify
# from flask_cors import CORS
# import pandas as pd

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # Load dataset
# df = pd.read_csv("dataset_updated (1).csv")
# known_phishing_urls = set(df['url'])  # Store in a set for fast lookup

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Parse incoming JSON data
#         data = request.get_json()
#         url = data.get("url")

#         if not url:
#             return jsonify({"error": "No URL provided"}), 400

#         # Check if the URL exists in the dataset
#         result = "Phishing" if url in known_phishing_urls else "Legitimate"

#         response = {"url": url, "prediction": result}
#         print(response)  # Log the response for debugging
#         return jsonify(response)

#     except Exception as e:
#         print("Error occurred:", e)  # Log error for debugging
#         return jsonify({"error": "Internal Server Error"}), 500

# if __name__ == "__main__":
    
#     app.run(debug=True)


# from flask import Flask, request, render_template, jsonify
# from flask_cors import CORS
# import pandas as pd
# import re

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # Load dataset
# df = pd.read_csv("balanced_dataset.csv")
# known_phishing_urls = set(df['url'])  # Store in a set for fast lookup

# # URL validation function
# def is_valid_url(url):
#     url_pattern = re.compile(
#         r'^(https?:\/\/)?'  # http:// or https://
#         r'(([A-Za-z0-9-]+\.)+[A-Za-z]{2,6}|'  # Domain
#         r'localhost|'  # Localhost
#         r'\d{1,3}(\.\d{1,3}){3})'  # IP address
#         r'(:\d+)?'  # Optional port
#         r'(\/.*)?$'  # Path
#     )
#     return re.match(url_pattern, url) is not None

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Parse incoming JSON data
#         data = request.get_json()
#         url = data.get("url")

#         if not url:
#             return jsonify({"error": "No URL provided"}), 400

#         if not is_valid_url(url):
#             return jsonify({"error": "Invalid URL format"}), 400

#         # Check if the URL exists in the dataset
#         result = "Phishing" if url in known_phishing_urls else "Legitimate"

#         response = {"url": url, "prediction": result}
#         print(response)  # Log the response for debugging
#         return jsonify(response)

#     except Exception as e:
#         print("Error occurred:", e)  # Log error for debugging
#         return jsonify({"error": "Internal Server Error"}), 500

# if __name__ == "__main__":
#     app.run(debug=True)



# pyth

from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import pandas as pd
import re

app = Flask(__name__)
CORS(app) 

# Load dataset
df = pd.read_csv("dataset_updated (1).csv")
known_phishing_urls = set(df['url'])  

# Updated URL validation function
def is_valid_url(url):
    url_pattern = re.compile(
        r'^(https?:\/\/)'  
        r'(([A-Za-z0-9-]+\.)+[A-Za-z]{2,6}|'  
        r'localhost|'  
        r'\d{1,3}(\.\d{1,3}){3})'  
        r'(:\d+)?'  
        r'(\/[^\s]*)?'  
        r'(\?[^\s]*)?$'  
    )
    if url.isdigit():
        return False
    return re.match(url_pattern, url) is not None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse incoming JSON data
        data = request.get_json()
        url = data.get("url")

        if not url:
            return jsonify({"error": "No URL provided"}), 400

        if not is_valid_url(url):
            return jsonify({"error": "Invalid URL format"}), 400

        # Check if the URL exists in the dataset
        result = "Phishing" if url in known_phishing_urls else "Legitimate"

        response = {"url": url, "prediction": result}
        print(response)  # Log the response for debugging
        return jsonify(response)

    except Exception as e:
        print("Error occurred:", e)  # Log error for debugging
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(debug=True)







    # URL PARSING 
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import pandas as pd
import re
import joblib
import sqlite3
import logging
from urllib.parse import urlparse, parse_qs
import whois
import socket

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes to allow frontend interaction
limiter = Limiter(get_remote_address, app=app, default_limits=["100 per hour", "10 per minute"])

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Load dataset and handle exceptions
try:
    df = pd.read_csv("balanced_dataset.csv")
    known_phishing_urls = set(df['url'])
except Exception as e:
    logging.error("Error loading dataset: %s", e)
    known_phishing_urls = set()

# Load trained ML model
try:
    model = joblib.load("phishing_model.pkl")
except Exception as e:
    logging.error("Error loading ML model: %s", e)
    model = None

# Initialize SQLite database for logging detected URLs
conn = sqlite3.connect('phishing_urls.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS detected_urls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT,
        result TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
""")
conn.commit()

def save_to_db(url, result):
    cursor.execute("INSERT INTO detected_urls (url, result) VALUES (?, ?)", (url, result))
    conn.commit()

# URL validation function
def is_valid_url(url):
    url_pattern = re.compile(r'^(https?:\/\/)' +
                             r'(([A-Za-z0-9-]+\.)+[A-Za-z]{2,6}|' +
                             r'localhost|' +
                             r'\d{1,3}(\.\d{1,3}){3})' +
                             r'(:\d+)?(\/[^\s]*)?(\?[^\s]*)?$')
    return re.match(url_pattern, url) is not None

# Extract domain from URL
def extract_domain(url):
    return urlparse(url).netloc

# Count number of subdomains
def count_subdomains(url):
    return len(urlparse(url).netloc.split('.')) - 1

# Check if URL contains an IP address
def contains_ip(url):
    return bool(re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', url))

# Detect phishing patterns in URL
def detect_phishing_patterns(url):
    phishing_keywords = ["secure", "login", "update", "bank", "verify", "account", "confirm", "password"]
    return any(keyword in url for keyword in phishing_keywords)

# Fetch WHOIS information
def get_whois_info(url):
    try:
        domain = extract_domain(url)
        whois_info = whois.whois(domain)
        return whois_info.creation_date
    except:
        return None

# Resolve DNS for a given URL
def resolve_dns(url):
    try:
        return socket.gethostbyname(extract_domain(url))
    except:
        return None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@limiter.limit("10 per minute")
def predict():
    try:
        data = request.get_json()
        url = data.get("url")
        if not url or not is_valid_url(url):
            return jsonify({"error": "Invalid URL format"}), 400

        # Determine if the URL is in the known phishing dataset
        result = "Phishing" if url in known_phishing_urls else "Legitimate"
        domain = extract_domain(url)
        subdomain_count = count_subdomains(url)
        has_ip = contains_ip(url)
        phishing_patterns = detect_phishing_patterns(url)
        whois_info = get_whois_info(url)
        dns_info = resolve_dns(url)
        url_length = len(url)
        https_present = url.startswith("https")

        # Extracting features for ML model prediction
        features = [url_length, subdomain_count, int(has_ip), int(https_present)]
        ml_prediction = "Unknown"
        if model:
            ml_prediction = "Phishing" if model.predict([features])[0] == 1 else "Legitimate"

        # Save result in database
        save_to_db(url, result)

        response = {
            "url": url,
            "prediction": result,
            "domain": domain,
            "subdomain_count": subdomain_count,
            "contains_ip": has_ip,
            "phishing_pattern_detected": phishing_patterns,
            "whois_info": whois_info,
            "dns_info": dns_info,
            "https_present": https_present,
            "ml_prediction": ml_prediction
        }
        logging.info("Processed URL: %s | Result: %s", url, result)
        return jsonify(response)
    except Exception as e:
        logging.error("Error processing URL: %s", e)
        return jsonify({"error": "Internal Server Error"}), 500

@app.route("/history", methods=["GET"])
def history():
    cursor.execute("SELECT url, result, timestamp FROM detected_urls ORDER BY id DESC LIMIT 50")
    urls = cursor.fetchall()
    return jsonify(urls)

@app.route("/bulk_predict", methods=["POST"])
def bulk_predict():
    try:
        data = request.get_json()
        urls = data.get("urls", [])
        results = []
        for url in urls:
            if not is_valid_url(url):
                results.append({"url": url, "error": "Invalid URL format"})
                continue
            result = "Phishing" if url in known_phishing_urls else "Legitimate"
            save_to_db(url, result)
            results.append({"url": url, "prediction": result})
        return jsonify(results)
    except Exception as e:
        logging.error("Bulk processing error: %s", e)
        return jsonify({"error": "Internal Server Error"}), 500

# if __name__ == "__main__":
#     app.run(debug=True)
