import requests

# Languages you want to include (ISO 639-1 codes)
languages = ["en", "fr", "de", "es", "it", "zh", "ru"]  # English, French, German, Spanish, Italian, Chinese, Russian

# File to store training data
output_file = "training_data.txt"

def fetch_books(language, num_books=5):
    """Fetch books from Gutenberg in a given language"""
    url = f"https://gutendex.com/books/?languages={language}&mime_type=text/plain&limit={num_books}"
    response = requests.get(url)
    
    if response.status_code == 200:
        books = response.json()["results"]
        return books
    else:
        print(f"Failed to fetch books for {language}")
        return []

def append_to_file(text):
    """Append text to the training_data.txt file"""
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("\n" + "-" * 80 + "\n")  # Separator
        f.write(text + "\n")

# Fetch and append books in multiple languages
for lang in languages:
    books = fetch_books(lang)
    for book in books:
        print(f"Fetching: {book['title']} ({lang})")
        text_url = book["formats"].get("text/plain; charset=utf-8") or book["formats"].get("text/plain")
        if text_url:
            book_text = requests.get(text_url).text
            append_to_file(f"Title: {book['title']} (Lang: {lang})\n{book_text[:5000]}...")  # Limiting to 5000 chars for each book

print(f"Training data updated in {output_file} with multiple languages!")
