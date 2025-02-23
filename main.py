import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_dataset(csv_path):
    """
    Loads and preprocesses the dataset from a CSV file.
    Returns a dictionary of {title: combined_plot}.
    """
    df = pd.read_csv(csv_path)
    df.fillna("", inplace=True)  # Handle missing values if any

    # Combine wiki_plot and imdb_plot into one text field
    df['combined_plot'] = df['wiki_plot'].astype(str) + " " + df['imdb_plot'].astype(str)

    # Build a dictionary: { "Title": "combined_plot" }
    data_dict = {}
    for idx, row in df.iterrows():
        title = row['Title']
        combined_plot = row['combined_plot']
        data_dict[title] = combined_plot

    return data_dict

def build_tfidf_vectorizer(all_texts):
    """
    Builds and fits a TF-IDF vectorizer on all plot data.
    Returns:
     - A fitted TfidfVectorizer instance
     - The TF-IDF matrix for all plots
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_texts)  # Fit + Transform
    return vectorizer, tfidf_matrix

def vectorize_query(query, vectorizer):
    """
    Vectorizes the user query using the same TF-IDF vectorizer.
    Returns the TF-IDF vector for the query.
    """
    query_vector = vectorizer.transform([query])
    return query_vector

def get_top_n_matches(query_vector, tfidf_matrix, titles, n=5):
    """
    Computes cosine similarity between the query vector and each plot.
    Returns the top N matching titles (sorted by descending similarity).
    """
    # Calculate cosine similarity
    # returns a 2D array of shape (1, num_movies).
    # Flatten it to get a 1D array of similarity scores.
    similarity_array = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Enumerate the similarity scores
    indexed_similarities = list(enumerate(similarity_array))
    
    # 3. Sort the (index, score) pairs by descending similarity score
    sorted_by_score_desc = sorted(indexed_similarities, key=lambda x: x[1], reverse=True)

    # 4. Extract the top N items
    top_n_indexed_scores = sorted_by_score_desc[:n]
    
    # Build list of (title, similarity_score)
    top_matches = []
    for index_score_tuple in top_n_indexed_scores:
        # Destructure the tuple into index and score
        idx, score = index_score_tuple

        # Find the corresponding title in the 'titles' list
        title = titles[idx]

        # Append (title, score) to our result list
        top_matches.append((title, score))

    return top_matches

def main():
    # Prompt user for input
    user_query = input("Enter your movie description: ")
    try:
        n_matches = int(input("Number of matches to return (N): "))
    except ValueError:
        n_matches = 5  # default if user doesn't enter a valid integer

    # Load dataset
    csv_path = "data/movies.csv"
    data_dict = load_dataset(csv_path)

    # Prepare data for TF-IDF
    titles = list(data_dict.keys())
    plots = list(data_dict.values())

    # Build TF-IDF vectorizer and transform plots
    vectorizer, tfidf_matrix = build_tfidf_vectorizer(plots)

    # Vectorize user query
    query_vector = vectorize_query(user_query, vectorizer)

    # Compute similarity and get top matches
    top_matches = get_top_n_matches(query_vector, tfidf_matrix, titles, n=n_matches)

    # Display results
    print("\nTop {} Matches for Query: '{}'".format(n_matches, user_query))
    for idx, (title, score) in enumerate(top_matches, start=1):
        print("{}. {} (Similarity: {:.4f})".format(idx, title, score))

if __name__ == "__main__":
    main()
