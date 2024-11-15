from flask import Flask, render_template, request, jsonify
from Recommender import AnimeRecommenderSystem, train_and_save_model
from functools import lru_cache
from flask_caching import Cache
import requests
import time
import os
import numpy as np
import json
import socket
import pandas as pd
from typing import Dict, List, Optional
import traceback
import math

app = Flask(__name__)

# Configure Flask-Caching
cache = Cache(app, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 300  # 5 minutes
})

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj) if not math.isnan(float(obj)) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app.json_encoder = CustomJSONEncoder

# Initialize the recommender system
recommender = AnimeRecommenderSystem(n_clusters=30)

# Check if model exists, if not train and save it
if not os.path.exists('model/kmeans_model.pkl'):
    print("No saved model found. Training new model...")
    recommender = train_and_save_model(limit=50000)
else:
    print("Loading saved model...")
    success = recommender.load_model()
    if not success:
        print("Error loading model. Training new model...")
        recommender = train_and_save_model(limit=50000)

@lru_cache(maxsize=1000)
def get_anime_poster(mal_id):
    """Cached version of poster fetching"""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            url = f"https://api.jikan.moe/v4/anime/{mal_id}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return data['data']['images']['jpg']['large_image_url']
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return None
    return None

@app.route('/')
def home():
    return render_template('index.html')

def safe_convert(value: any, convert_type: type, default: any = None) -> any:
    """Safely convert a value to the specified type"""
    try:
        if pd.isna(value):
            return default
        return convert_type(value)
    except (ValueError, TypeError):
        return default

# Cache search results
@cache.memoize(timeout=300)
def cached_search(query: str, top_k: int) -> List[Dict]:
    """Cache search results for 5 minutes"""
    return recommender.search_anime(query, top_k)

@app.route('/search', methods=['POST'])
def search():
    try:
        query = request.form.get('query')
        if not query:
            return jsonify({'error': 'No search query provided'})
        
        print(f"Searching for query: {query}")
        
        # Use cached search instead of direct search
        search_results = cached_search(query, 20)
        if not search_results:
            return jsonify({'results': []})
            
        print(f"Found {len(search_results)} results")
        
        processed_results = []
        for result in search_results:
            try:
                print(f"Processing result: {result}")
                
                if not isinstance(result, dict):
                    print(f"Unexpected result type: {type(result)}")
                    continue
                    
                processed_result = {
                    'mal_id': safe_convert(result.get('mal_id'), int),
                    'title': safe_convert(result.get('title'), str),
                    'title_english': safe_convert(result.get('title_english'), str),
                    'year': safe_convert(result.get('year'), int),
                    'score': safe_convert(result.get('score'), float),
                    'type': safe_convert(result.get('type'), str),
                    'similarity_score': safe_convert(result.get('similarity_score'), float)
                }
                
                if processed_result['mal_id'] is not None:
                    processed_results.append(processed_result)
                    
            except Exception as e:
                print(f"Error processing individual result: {str(e)}")
                print(traceback.format_exc())
                continue
        
        if not processed_results:
            return jsonify({'results': []})
            
        return jsonify({'results': processed_results})
        
    except Exception as e:
        print(f"Search error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': 'An error occurred while searching. Please try again.'})

# Cache recommendation results
@cache.memoize(timeout=300)
def cached_recommendations(mal_id: int, n_recommendations: int) -> List[Dict]:
    """Cache recommendation results for 5 minutes"""
    return recommender.get_recommendations(mal_id, n_recommendations)

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        mal_id = request.form.get('mal_id')
        n_recommendations = int(request.form.get('n_recommendations', 10))
        
        if not mal_id:
            return jsonify({'error': 'No anime ID provided'})

        print(f"Getting recommendations for anime ID: {mal_id}")
        # Use cached recommendations
        recommendations = cached_recommendations(int(mal_id), n_recommendations)
        
        if isinstance(recommendations, list) and recommendations and 'error' in recommendations[0]:
            return jsonify({'error': recommendations[0]['error']})
        
        processed_recommendations = []
        for rec in recommendations:
            try:
                if pd.isna(rec.get('mal_id')):
                    continue
                    
                processed_rec = {
                    'mal_id': safe_convert(rec.get('mal_id'), int),
                    'title': safe_convert(rec.get('title'), str),
                    'score': safe_convert(rec.get('score'), float),
                    'year': safe_convert(rec.get('year'), int),
                    'type': safe_convert(rec.get('type'), str)
                }
                
                if processed_rec['mal_id'] is not None:
                    time.sleep(1)  # Respect Jikan API rate limits
                    poster_url = get_anime_poster(processed_rec['mal_id'])
                    processed_rec['poster_url'] = poster_url
                    processed_recommendations.append(processed_rec)
                    
            except Exception as e:
                print(f"Error processing recommendation: {str(e)}")
                print(traceback.format_exc())
                continue
        
        return jsonify({'recommendations': processed_recommendations})
        
    except Exception as e:
        print(f"Recommendation error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'An error occurred while getting recommendations: {str(e)}'})

@app.route('/diagnostic')
def diagnostic():
    """Diagnostic endpoint to check data loading"""
    try:
        if recommender.df is None:
            return jsonify({'error': 'No data loaded'})
            
        sample_data = recommender.df.head(5).to_dict('records')
        stats = {
            'total_entries': len(recommender.df),
            'columns': list(recommender.df.columns),
            'null_counts': recommender.df.isnull().sum().to_dict(),
            'sample_data': sample_data
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': f'Diagnostic error: {str(e)}'})

def find_free_port():
    """Find a free port on the system"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=True)
    except OSError:
        port = find_free_port()
        print(f"Port 5000 is in use, using port {port} instead")
        app.run(host='0.0.0.0', port=port, debug=True)