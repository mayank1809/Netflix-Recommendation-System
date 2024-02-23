import numpy as np
import pandas as pd
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from PIL import Image, ImageTk  # For handling images

# Load the dataset
netflix_dataset = pd.read_csv(r"C:\college books and notes\0folder\!college books and course\PLACEMENT\PROJECTS\netflix\coding_part\netflix_dataset.csv")

# Content Based Recommendations - Plot description based Recommender
tfidf = TfidfVectorizer(stop_words='english')
netflix_dataset['description'] = netflix_dataset['description'].fillna('')
tfidf_matrix = tfidf.fit_transform(netflix_dataset['description'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(netflix_dataset.index, index=netflix_dataset['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return netflix_dataset.iloc[movie_indices]

def on_select():
    selected_title = combobox.get()
    if selected_title:
        recommended_titles = get_recommendations(selected_title)
        show_recommendations(recommended_titles)
    else:
        messagebox.showinfo("Error", "Please select or enter a movie title.")

def autocomplete(event):
    input_text = event.widget.get()
    if input_text:
        suggestions = [title for title in netflix_dataset['title'] if title.lower().startswith(input_text.lower())]
        event.widget['values'] = suggestions

def show_recommendations(recommended_titles):
    if len(recommended_titles) == 0:
        messagebox.showinfo("No Recommendations", "No recommendations found for the selected movie.")
    else:
        # Create new window for recommendations
        recommendation_window = Toplevel()
        recommendation_window.title("Recommended Movies")
        recommendation_window.configure(bg='lightblue')

        # Display recommendations
        for i, (index, row) in enumerate(recommended_titles.iterrows(), start=1):
            recommendation_label = Label(recommendation_window, text=f"{i}. {row['title']}", font=("Helvetica", 14, 'bold'), bg="lightblue", fg="red")
            recommendation_label.pack(pady=5)

# Create Tkinter GUI
root = Tk()
root.title("Netflix Movie Recommender")
root.configure(bg='blue')

# Set window icon using raw string literals
root.iconbitmap(r"C:\college books and notes\0folder\!college books and course\PLACEMENT\PROJECTS\netflix\coding_part\netflix_icon_161073.ico")

# Add label with custom styling
label = Label(root, text="Enter the title of the movie:", font=("Helvetica", 16, 'bold'), bg="blue", fg="white")
label.pack()

# Add combobox with custom styling
combobox = ttk.Combobox(root, font=("Helvetica", 14), width=40)
combobox.pack()
combobox.bind("<KeyRelease>", autocomplete)
combobox['values'] = list(netflix_dataset['title'])

# Add button with custom styling
button = Button(root, text="Get Recommendations", command=on_select, font=("Helvetica", 14, 'bold'), bg="lightgreen", fg="black")
button.pack()

root.mainloop()
