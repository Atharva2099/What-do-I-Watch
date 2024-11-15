import requests
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import time
import pickle
import os
from fuzzywuzzy import fuzz
import traceback

class AnimeRecommenderSystem:
    def __init__(self, n_clusters=30):
        self.base_url = "https://api.jikan.moe/v4"
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.df = None
        self.features = None
        try:
            self.bert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            print("BERT model initialized successfully")
        except Exception as e:
            print(f"Error initializing BERT model: {e}")
            self.bert_model = None
        self.title_embeddings = None

    def fetch_top_anime(self, limit: int = 50000) -> List[Dict]:
        """Fetch anime data from Jikan API with improved pagination"""
        anime_list = []
        page = 1
        
        while len(anime_list) < limit:
            try:
                url = f"{self.base_url}/top/anime"
                params = {
                    'page': page,
                    'limit': 25
                }
                
                response = requests.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                if not data['data']:
                    break
                    
                anime_list.extend(data['data'])
                
                if len(anime_list) >= limit:
                    anime_list = anime_list[:limit]
                    break
                
                page += 1
                print(f"Fetched {len(anime_list)}/{limit} anime... (Page {page})")
                
                if page % 3 == 0:
                    time.sleep(4)
                else:
                    time.sleep(1)
                    
            except requests.exceptions.RequestException as e:
                print(f"Error fetching data: {e}")
                time.sleep(5)
                continue
                
        print(f"Successfully fetched {len(anime_list)} anime!")
        return anime_list

    def process_anime_data(self, anime_list: List[Dict]) -> pd.DataFrame:
        """Process raw anime data into a DataFrame"""
        processed_data = []
        
        for anime in anime_list:
            try:
                genres = [g['name'] for g in anime.get('genres', [])]
                studios = [s['name'] for s in anime.get('studios', [])]
                themes = [t['name'] for t in anime.get('themes', [])]
                
                processed_anime = {
                    'mal_id': anime['mal_id'],
                    'title': anime['title'],
                    'title_english': anime.get('title_english', ''),
                    'type': anime.get('type', ''),
                    'episodes': anime.get('episodes', 0),
                    'status': anime.get('status', ''),
                    'score': anime.get('score', 0),
                    'scored_by': anime.get('scored_by', 0),
                    'members': anime.get('members', 0),
                    'genres': ','.join(genres),
                    'studios': ','.join(studios),
                    'themes': ','.join(themes),
                    'year': anime.get('year', 0),
                    'synopsis': anime.get('synopsis', ''),
                    'rating': anime.get('rating', '')
                }
                processed_data.append(processed_anime)
            except Exception as e:
                print(f"Error processing anime: {e}")
                continue
            
        return pd.DataFrame(processed_data)

    def prepare_features(self):
        """Prepare features for clustering"""
        if self.df is None:
            raise ValueError("No data loaded. Call fetch_and_prepare_data first.")
            
        genre_dummies = self.df['genres'].str.get_dummies(sep=',')
        theme_dummies = self.df['themes'].str.get_dummies(sep=',')
        type_dummies = pd.get_dummies(self.df['type'], prefix='type')
        
        numerical_features = self.df[['score', 'members', 'episodes', 'year']].copy()
        numerical_features = numerical_features.fillna(numerical_features.mean())
        
        scaled_numerical = self.scaler.fit_transform(numerical_features)
        scaled_numerical_df = pd.DataFrame(
            scaled_numerical, 
            columns=numerical_features.columns
        )
        
        self.features = pd.concat([
            scaled_numerical_df,
            genre_dummies,
            theme_dummies,
            type_dummies
        ], axis=1)

    def search_anime(self, query: str, top_k: int = 10) -> List[Dict]:
        """Semantic search for anime titles with improved error handling"""
        try:
            print(f"Starting search for query: {query}")
            
            if self.df is None:
                raise ValueError("No data loaded. Call load_model first.")
            
            if self.title_embeddings is None:
                print("Creating title embeddings...")
                self._create_title_embeddings()
            
            try:
                query_embedding = self.bert_model.encode([query])
                similarities = cosine_similarity(query_embedding, self.title_embeddings)[0]
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                
            except Exception as e:
                print(f"BERT search failed, falling back to fuzzy matching: {e}")
                similarities = self.df['title'].apply(lambda x: fuzz.ratio(query.lower(), str(x).lower()))
                top_indices = similarities.nlargest(top_k).index
            
            results = []
            for idx in top_indices:
                anime = self.df.iloc[idx]
                result = {
                    'mal_id': int(anime['mal_id']),
                    'title': str(anime['title']),
                    'title_english': str(anime['title_english']) if pd.notna(anime.get('title_english')) else None,
                    'year': int(anime['year']) if pd.notna(anime.get('year')) else None,
                    'score': float(anime['score']) if pd.notna(anime.get('score')) else None,
                    'type': str(anime['type']) if pd.notna(anime.get('type')) else None,
                    'similarity_score': float(similarities[idx])
                }
                results.append(result)
            
            print(f"Found {len(results)} results")
            return results
            
        except Exception as e:
            print(f"Search error: {str(e)}")
            print(traceback.format_exc())
            raise Exception(f"Search failed: {str(e)}")

    def get_recommendations(self, mal_id: int, n_recommendations: int = 10) -> List[Dict]:
        """Get anime recommendations based on MAL ID"""
        try:
            input_anime = self.df[self.df['mal_id'] == mal_id].iloc[0]
        except IndexError:
            return [{"error": f"Anime with ID {mal_id} not found in database"}]
        
        input_anime_cluster = input_anime['Cluster']
        cluster_anime = self.df[self.df['Cluster'] == input_anime_cluster]
        
        if len(cluster_anime) < n_recommendations + 1:
            cluster_centers = self.kmeans.cluster_centers_
            input_center = cluster_centers[input_anime_cluster]
            distances = [np.linalg.norm(input_center - center) for center in cluster_centers]
            closest_clusters = np.argsort(distances)[:3]
            cluster_anime = self.df[self.df['Cluster'].isin(closest_clusters)]
        
        input_features = self.features.iloc[self.df[self.df['mal_id'] == mal_id].index[0]]
        cluster_features = self.features.iloc[cluster_anime.index]
        
        similarities = cosine_similarity([input_features], cluster_features)[0]
        similar_indices = np.argsort(similarities)[-n_recommendations-1:][::-1]
        recommendations = cluster_anime.iloc[similar_indices]
        
        recommendations = recommendations[recommendations['mal_id'] != mal_id]
        
        return recommendations.head(n_recommendations).to_dict('records')

    def _create_title_embeddings(self):
        """Create embeddings for all anime titles with error handling"""
        try:
            if self.bert_model is None:
                raise ValueError("BERT model not initialized")
                
            print("Creating title embeddings...")
            titles = self.df['title'].fillna('').tolist()
            self.title_embeddings = self.bert_model.encode(titles, show_progress_bar=True)
            print(f"Created embeddings for {len(titles)} titles")
            
        except Exception as e:
            print(f"Error creating title embeddings: {e}")
            print(traceback.format_exc())
            raise

    def save_model(self, model_dir='model'):
        """Save trained model and preprocessed data"""
        try:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
                
            with open(f'{model_dir}/kmeans_model.pkl', 'wb') as f:
                pickle.dump(self.kmeans, f)
                
            with open(f'{model_dir}/scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
                
            self.df.to_csv(f'{model_dir}/processed_data.csv', index=False)
            self.features.to_csv(f'{model_dir}/features.csv', index=False)
            
            if self.title_embeddings is not None:
                np.save(f'{model_dir}/title_embeddings.npy', self.title_embeddings)
            
            print("Model and data saved successfully!")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def load_model(self, model_dir='model'):
        """Load trained model and preprocessed data"""
        try:
            with open(f'{model_dir}/kmeans_model.pkl', 'rb') as f:
                self.kmeans = pickle.load(f)
                
            with open(f'{model_dir}/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
                
            self.df = pd.read_csv(f'{model_dir}/processed_data.csv')
            self.features = pd.read_csv(f'{model_dir}/features.csv')
            
            if os.path.exists(f'{model_dir}/title_embeddings.npy'):
                self.title_embeddings = np.load(f'{model_dir}/title_embeddings.npy')
            else:
                self._create_title_embeddings()
            
            print("Model and data loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def fetch_and_prepare_data(self, limit: int = 50000):
        """Fetch data and prepare it for the recommender system"""
        print(f"Fetching {limit} anime from Jikan API...")
        anime_list = self.fetch_top_anime(limit)
        self.df = self.process_anime_data(anime_list)
        
        print("Preparing features...")
        self.prepare_features()
        
        print("Creating embeddings for semantic search...")
        self._create_title_embeddings()
        
        print("Training clustering model...")
        self.kmeans.fit(self.features)
        self.df['Cluster'] = self.kmeans.labels_

def train_and_save_model(limit=50000, use_existing_data=True):
    """Train model using existing data or fetch new data"""
    print("Initializing recommender system...")
    recommender = AnimeRecommenderSystem(n_clusters=30)
    
    if use_existing_data and os.path.exists('model/processed_data.csv'):
        print("Loading existing data...")
        recommender.df = pd.read_csv('model/processed_data.csv')
        print(f"Loaded {len(recommender.df)} anime entries")
        
        print("Preparing features...")
        recommender.prepare_features()
        
        print("Creating embeddings for semantic search...")
        recommender._create_title_embeddings()
        
        print("Training clustering model...")
        recommender.kmeans.fit(recommender.features)
        recommender.df['Cluster'] = recommender.kmeans.labels_
        
        print("Saving model...")
        recommender.save_model()
        print("Model training and saving completed!")
    else:
        print(f"No existing data found. Training new model with {limit} anime...")
        recommender.fetch_and_prepare_data(limit=limit)
        recommender.save_model()
        
    return recommender

if __name__ == "__main__":
    # Set use_existing_data=True to use saved CSV
    recommender = train_and_save_model(limit=50000, use_existing_data=True)