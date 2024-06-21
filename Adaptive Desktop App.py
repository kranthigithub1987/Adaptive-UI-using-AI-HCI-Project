import sys
import logging
import pandas as pd
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QSizePolicy, QSpacerItem, QGridLayout
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QFont
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel, GPTNeoForCausalLM
from collections import Counter

# Initialize logging configuration
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ContentRecommender:
    def __init__(self):
        # Sample content database
        self.content_db = pd.DataFrame({
            'id': range(1, 6),
            'title': ['Python Basics', 'Machine Learning', 'Web Development', 'Data Science', 'AI Fundamentals'],
            'content': ['Python programming language basics', 'Introduction to machine learning algorithms', 
                        'Web development using HTML, CSS, and JavaScript', 'Data analysis and visualization', 
                        'Artificial intelligence and neural networks']
        })
        self.vectorizer = TfidfVectorizer()
        self.content_vectors = self.vectorizer.fit_transform(self.content_db['content'])

        # Initialize BERT model for semantic understanding
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

    def get_recommendations(self, query):
        # Use BERT embeddings for query and content
        query_tokens = self.tokenizer.encode(query, add_special_tokens=True, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
        content_tokens = self.tokenizer.encode(self.content_db['content'].tolist(), add_special_tokens=True, max_length=128, truncation=True, padding='max_length', return_tensors='pt')

        with torch.no_grad():
            query_embeddings = self.bert_model(query_tokens)[0][:, 0, :].squeeze(1)
            content_embeddings = self.bert_model(content_tokens)[0][:, 0, :].squeeze(1)

        similarities = cosine_similarity(query_embeddings.cpu().numpy(), content_embeddings.cpu().numpy()).flatten()
        recommended_indices = similarities.argsort()[-3:][::-1]  # Top 3 recommendations
        return self.content_db.iloc[recommended_indices]['title'].tolist()

class UserPreferenceLearner:
    def __init__(self):
        self.user_searches = []

    def add_search(self, query):
        self.user_searches.append(query)

    def get_top_interests(self):
        words = [word for query in self.user_searches for word in query.lower().split()]
        return [word for word, _ in Counter(words).most_common(3)]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Adaptive Desktop App")
        self.setGeometry(100, 100, 800, 600)

        main_layout = QVBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Title
        title_label = QLabel("Adaptive Desktop App")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 20px 0;")
        title_label.setFont(QFont("Arial", 24))

        # Instructions for the user
        instruction_label = QLabel(
            "Users should search for topics related to Tea, Coffee, Milk, Java, Python, Machine Learning, Web Development, Data Science, and AI.\n"
            "They can enter specific queries within these fields to receive personalized content recommendations and generated content based on their interests."
        )
        instruction_label.setAlignment(Qt.AlignCenter)
        instruction_label.setWordWrap(True)
        instruction_label.setStyleSheet("font-size: 16px; margin-bottom: 20px; padding: 10px;")
        instruction_label.setFont(QFont("Arial", 16))

        # Search input and button
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter your search query")
        self.search_input.setFixedHeight(40)
        self.search_input.setStyleSheet("font-size: 14px; padding: 5px; border: 1px solid #ccc; border-radius: 5px;")
        self.search_button = QPushButton("Search")
        self.search_button.setFixedSize(100, 40)
        self.search_button.setStyleSheet("font-size: 14px; background-color: #4CAF50; color: white; border: none; border-radius: 5px;")
        self.search_button.setIcon(QIcon.fromTheme("search"))
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_button)

        # Results area
        self.result_label = QLabel("Results will appear here")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-size: 16px; margin: 20px; min-height: 100px; background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        self.result_label.setFont(QFont("Arial", 16))

        # Additional description
        description_label = QLabel(
            "This app provides content recommendations based on user search queries using AI and machine learning techniques. "
            "It also generates related content and learns user preferences over time to highlight the most frequently searched topics. "
        )
        description_label.setAlignment(Qt.AlignCenter)
        description_label.setWordWrap(True)
        description_label.setStyleSheet("font-size: 14px; margin-top: 20px; color: #666; padding: 10px;")
        description_label.setFont(QFont("Arial", 14))

        # Add widgets to main layout
        main_layout.addWidget(title_label)
        main_layout.addWidget(instruction_label)
        main_layout.addLayout(search_layout)
        main_layout.addWidget(self.result_label)
        main_layout.addWidget(description_label)
        main_layout.addStretch(1)  # Add stretch to push content to the top

        self.search_button.clicked.connect(self.search)

        # Initialize AI components
        self.recommender = ContentRecommender()
        self.preference_learner = UserPreferenceLearner()

        # Initialize models for content generation
        self.models = {
            'GPT-2': (GPT2Tokenizer.from_pretrained('gpt2'), GPT2LMHeadModel.from_pretrained('gpt2')),
            'GPT-Neo': (GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-125M'), GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M'))
        }

    def generate_content(self, prompt):
        generated_contents = []
        for model_name, (tokenizer, model) in self.models.items():
            inputs = tokenizer.encode(prompt, return_tensors='pt')
            attention_mask = torch.ones(inputs.shape, dtype=torch.long)  # Add attention mask for padding
            outputs = model.generate(inputs, attention_mask=attention_mask, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_contents.append(f"{model_name}:\n{generated_text}\n")
        return "\n\n".join(generated_contents)

    def search(self):
        query = self.search_input.text().strip()  # Ensure query is cleaned and trimmed
        if not query:
            self.result_label.setText("Please enter a search query.")
            return
        
        try:
            # Get recommendations and user preferences
            recommendations = self.recommender.get_recommendations(query)
            self.preference_learner.add_search(query)
            top_interests = self.preference_learner.get_top_interests()

            # Generate content related to the query
            generated_content = self.generate_content(query)
            
            # Update UI with results
            result_text = (
                f"Recommendations:\n{', '.join(recommendations)}\n\n"
                f"Top Interests:\n{', '.join(top_interests)}\n\n"
                f"Generated Content:\n{generated_content}"
            )
            self.result_label.setText(result_text)
        except Exception as e:
            self.result_label.setText(f"Error: {str(e)}")
            logging.error(f"Error in search: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    
    # Additional styling and adjustments
    app.setStyleSheet("QMainWindow { background-color: #e6ffe6; }")
    window.setStyleSheet("QLabel { color: #333; }")

    window.show()
    sys.exit(app.exec_())

